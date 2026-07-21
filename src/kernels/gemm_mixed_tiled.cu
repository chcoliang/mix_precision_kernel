#include "gemm_kernels.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Variant 3: Tiled Mixed Precision GEMM using BF16 Tensor Core (via half WMMA)
// Strategy: Dequantize A (MXFP8) to half per-tile, B (bf16) to half, use WMMA
// Multi-warp version: 4 warps per block, each handling one 16x16 tile (32x32 total)

#define TC_M 16
#define TC_N 16
#define TC_K 16
#define WARPS_PER_BLOCK 4

__global__ void gemm_mixed_tiled_kernel(const uint8_t* __restrict__ A_data,
                                         const uint8_t* __restrict__ A_scales,
                                         const __nv_bfloat16* __restrict__ B,
                                         const uint8_t* __restrict__ C_data,
                                         const uint8_t* __restrict__ C_scales,
                                         float* __restrict__ D,
                                         int M, int N, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Each block handles a 2x2 grid of 16x16 tiles
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    int row_start = blockIdx.y * (2 * TC_M) + warp_row * TC_M;
    int col_start = blockIdx.x * (2 * TC_N) + warp_col * TC_N;

    if (row_start >= M || col_start >= N) return;

    wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Each warp has its own tile buffers in shared memory
    __shared__ half sA[WARPS_PER_BLOCK][TC_M * TC_K];
    __shared__ half sB[WARPS_PER_BLOCK][TC_K * TC_N];

    for (int k_tile = 0; k_tile < K; k_tile += TC_K) {
        // Load A: dequantize MXFP8 to half
        for (int idx = lane; idx < TC_M * TC_K; idx += 32) {
            int r = idx / TC_K;
            int c = idx % TC_K;
            int global_r = row_start + r;
            int global_k = k_tile + c;

            if (global_r < M && global_k < K) {
                int blk_idx = global_k / MXFP8_BLOCK_SIZE;
                uint8_t scale_e8m0 = A_scales[global_r * num_blocks_k + blk_idx];
                float scale = e8m0_to_float(scale_e8m0);
                uint8_t fp8_val = A_data[global_r * K + global_k];
                float val = fp8_e4m3_to_float(fp8_val) * scale;
                sA[warp_id][r * TC_K + c] = __float2half(val);
            } else {
                sA[warp_id][r * TC_K + c] = __float2half(0.0f);
            }
        }

        // Load B: bf16 -> half
        for (int idx = lane; idx < TC_K * TC_N; idx += 32) {
            int r = idx / TC_N;
            int c = idx % TC_N;
            int global_k = k_tile + r;
            int global_c = col_start + c;

            if (global_k < K && global_c < N) {
                float bval = __bfloat162float(B[global_k * N + global_c]);
                sB[warp_id][r * TC_N + c] = __float2half(bval);
            } else {
                sB[warp_id][r * TC_N + c] = __float2half(0.0f);
            }
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, sA[warp_id], TC_K);
        wmma::load_matrix_sync(b_frag, sB[warp_id], TC_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    // Add dequantized C
    if (C_data != nullptr && C_scales != nullptr) {
        int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
        __shared__ float sC[WARPS_PER_BLOCK][TC_M * TC_N];

        for (int idx = lane; idx < TC_M * TC_N; idx += 32) {
            int r = idx / TC_N;
            int c = idx % TC_N;
            int global_r = row_start + r;
            int global_c = col_start + c;

            if (global_r < M && global_c < N) {
                int c_blk = global_c / MXFP8_BLOCK_SIZE;
                uint8_t c_scale_e8m0 = C_scales[global_r * num_blocks_n_c + c_blk];
                float c_scale = e8m0_to_float(c_scale_e8m0);
                uint8_t c_fp8 = C_data[global_r * N + global_c];
                sC[warp_id][r * TC_N + c] = fp8_e4m3_to_float(c_fp8) * c_scale;
            } else {
                sC[warp_id][r * TC_N + c] = 0.0f;
            }
        }
        __syncwarp();

        wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> c_frag;
        wmma::load_matrix_sync(c_frag, sC[warp_id], TC_N, wmma::mem_row_major);

        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] += c_frag.x[i];
        }
    }

    // Store result
    if (row_start + TC_M <= M && col_start + TC_N <= N) {
        wmma::store_matrix_sync(D + row_start * N + col_start, acc_frag, N, wmma::mem_row_major);
    } else {
        __shared__ float sD[WARPS_PER_BLOCK][TC_M * TC_N];
        wmma::store_matrix_sync(sD[warp_id], acc_frag, TC_N, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < TC_M * TC_N; idx += 32) {
            int r = idx / TC_N;
            int c = idx % TC_N;
            int global_r = row_start + r;
            int global_c = col_start + c;
            if (global_r < M && global_c < N) {
                D[global_r * N + global_c] = sD[warp_id][r * TC_N + c];
            }
        }
    }
}

void gemm_mixed_tiled(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                       const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                       const uint8_t* d_C_scales,
                       uint8_t* d_D_data, uint8_t* d_D_scales,
                       int M, int N, int K, cudaStream_t stream) {
    float* d_D_fp32;
    CUDA_CHECK(cudaMalloc(&d_D_fp32, M * N * sizeof(float)));

    dim3 block(128);
    int grid_m = (M + 2 * TC_M - 1) / (2 * TC_M);
    int grid_n = (N + 2 * TC_N - 1) / (2 * TC_N);
    dim3 grid(grid_n, grid_m);
    gemm_mixed_tiled_kernel<<<grid, block, 0, stream>>>(d_A_data, d_A_scales, d_B,
                                                         d_C_data, d_C_scales, d_D_fp32, M, N, K);

    mxfp8_quantize_gpu(d_D_fp32, d_D_data, d_D_scales, M, N, stream);
    cudaFree(d_D_fp32);
}
