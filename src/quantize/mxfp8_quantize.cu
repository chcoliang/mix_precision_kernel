#include "mxfp8_utils.h"
#include <cfloat>
#include <cmath>
#include <algorithm>

// GPU kernel: quantize float -> MXFP8 (block size = 32 along K)
__global__ void mxfp8_quantize_kernel(const float* __restrict__ input,
                                       uint8_t* __restrict__ output_data,
                                       uint8_t* __restrict__ output_scales,
                                       int M, int K, int num_blocks_k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = blockIdx.y;

    if (row >= M || block_idx >= num_blocks_k) return;

    int k_start = block_idx * MXFP8_BLOCK_SIZE;
    int k_end = min(k_start + MXFP8_BLOCK_SIZE, K);

    // Step 1: find amax in this block
    float amax = 0.0f;
    for (int k = k_start; k < k_end; k++) {
        float val = fabsf(input[row * K + k]);
        amax = fmaxf(amax, val);
    }

    // Step 2: compute E8M0 scale = round_up(amax / 448.0f)
    const float max_fp8_e4m3 = 448.0f;
    float scale_val = amax / max_fp8_e4m3;
    if (scale_val < 1.1754944e-38f) scale_val = 1.1754944e-38f; // min normal float
    uint8_t e8m0_scale = float_to_e8m0(scale_val);
    float actual_scale = e8m0_to_float(e8m0_scale);

    // Store scale
    output_scales[row * num_blocks_k + block_idx] = e8m0_scale;

    // Step 3: quantize each element
    float inv_scale = (actual_scale > 0.0f) ? (1.0f / actual_scale) : 0.0f;
    for (int k = k_start; k < k_end; k++) {
        float val = input[row * K + k] * inv_scale;
        output_data[row * K + k] = float_to_fp8_e4m3(val);
    }
}

// GPU kernel: quantize bf16 -> FP8 with per-tensor scaling
__global__ void bf16_to_fp8_kernel(const __nv_bfloat16* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = __bfloat162float(input[idx]) * scale;
    output[idx] = float_to_fp8_e4m3(val);
}

// Reduce kernel to find max abs value
__global__ void find_amax_kernel(const __nv_bfloat16* __restrict__ input,
                                  float* __restrict__ amax_out, int size) {
    __shared__ float shared_max[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(input[i])));
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(amax_out),
                  __float_as_int(shared_max[0]));
    }
}

void mxfp8_quantize_gpu(const float* d_input, uint8_t* d_output_data, uint8_t* d_output_scales,
                         int M, int K, cudaStream_t stream) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    dim3 block(256);
    dim3 grid((M + block.x - 1) / block.x, num_blocks_k);
    mxfp8_quantize_kernel<<<grid, block, 0, stream>>>(d_input, d_output_data, d_output_scales, M, K, num_blocks_k);
}

void bf16_to_fp8_per_tensor_gpu(const __nv_bfloat16* d_input, uint8_t* d_output,
                                 float* d_scale, int size, cudaStream_t stream) {
    // Step 1: find amax
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(d_scale, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = min((size + threads - 1) / threads, 1024);
    find_amax_kernel<<<blocks, threads, 0, stream>>>(d_input, d_scale, size);

    // Step 2: compute scale and quantize
    float amax;
    CUDA_CHECK(cudaMemcpyAsync(&amax, d_scale, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float scale = (amax > 0.0f) ? (448.0f / amax) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store inverse scale for dequantization
    CUDA_CHECK(cudaMemcpyAsync(d_scale, &inv_scale, sizeof(float), cudaMemcpyHostToDevice, stream));

    blocks = (size + threads - 1) / threads;
    bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(d_input, d_output, scale, size);
}

// Host reference implementation
void mxfp8_quantize_host(const float* input, uint8_t* output_data, uint8_t* output_scales,
                          int M, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    for (int row = 0; row < M; row++) {
        for (int blk = 0; blk < num_blocks_k; blk++) {
            int k_start = blk * MXFP8_BLOCK_SIZE;
            int k_end = std::min(k_start + MXFP8_BLOCK_SIZE, K);

            float amax = 0.0f;
            for (int k = k_start; k < k_end; k++) {
                amax = std::max(amax, std::abs(input[row * K + k]));
            }

            float scale_val = amax / 448.0f;
            if (scale_val < 1.1754944e-38f) scale_val = 1.1754944e-38f;
            uint8_t e8m0_scale = float_to_e8m0(scale_val);
            float actual_scale = e8m0_to_float(e8m0_scale);

            output_scales[row * num_blocks_k + blk] = e8m0_scale;

            float inv_scale = (actual_scale > 0.0f) ? (1.0f / actual_scale) : 0.0f;
            for (int k = k_start; k < k_end; k++) {
                float val = input[row * K + k] * inv_scale;
                output_data[row * K + k] = float_to_fp8_e4m3(val);
            }
        }
    }
}
