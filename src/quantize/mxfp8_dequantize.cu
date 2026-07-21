#include "mxfp8_utils.h"

// GPU kernel: dequantize MXFP8 -> float
__global__ void mxfp8_dequantize_kernel(const uint8_t* __restrict__ input_data,
                                         const uint8_t* __restrict__ input_scales,
                                         float* __restrict__ output,
                                         int M, int K, int num_blocks_k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || k >= K) return;

    int block_idx = k / MXFP8_BLOCK_SIZE;
    uint8_t e8m0_scale = input_scales[row * num_blocks_k + block_idx];
    float scale = e8m0_to_float(e8m0_scale);

    uint8_t fp8_val = input_data[row * K + k];
    float val = fp8_e4m3_to_float(fp8_val);
    output[row * K + k] = val * scale;
}

void mxfp8_dequantize_gpu(const uint8_t* d_input_data, const uint8_t* d_input_scales,
                           float* d_output, int M, int K, cudaStream_t stream) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (K + block.y - 1) / block.y);
    mxfp8_dequantize_kernel<<<grid, block, 0, stream>>>(d_input_data, d_input_scales,
                                                         d_output, M, K, num_blocks_k);
}

// Host reference implementation
void mxfp8_dequantize_host(const uint8_t* input_data, const uint8_t* input_scales,
                            float* output, int M, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int block_idx = k / MXFP8_BLOCK_SIZE;
            uint8_t e8m0_scale = input_scales[row * num_blocks_k + block_idx];
            float scale = e8m0_to_float(e8m0_scale);

            uint8_t fp8_val = input_data[row * K + k];
            float val = fp8_e4m3_to_float(fp8_val);
            output[row * K + k] = val * scale;
        }
    }
}
