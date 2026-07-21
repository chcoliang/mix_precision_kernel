#include "gemm_kernels.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each block = one warp (32 threads) computing one 16x16 output tile
__global__ void gemm_fp8_tc_kernel(const uint8_t* __restrict__ A_data,
                                    const uint8_t* __restrict__ A_scales,
                                    const __nv_bfloat16* __restrict__ B,
                                    const uint8_t* __restrict__ C_data,
                                    const uint8_t* __restrict__ C_scales,
                                    float* __restrict__ D,
                                    int M, int N, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    int row_start = blockIdx.y * WMMA_M;
    int col_start = blockIdx.x * WMMA_N;

    if (row_start >= M || col_start >= N) return;

    int lane = threadIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half sA[WMMA_M * WMMA_K];
    __shared__ half sB[WMMA_K * WMMA_N];

    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
        // Load A tile: dequantize MXFP8 to half (256 elements, 32 threads, 8 per thread)
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += 32) {
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            int global_r = row_start + r;
            int global_k = k_tile + c;

            if (global_r < M && global_k < K) {
                int blk_idx = global_k / MXFP8_BLOCK_SIZE;
                uint8_t scale_e8m0 = A_scales[global_r * num_blocks_k + blk_idx];
                float scale = e8m0_to_float(scale_e8m0);
                uint8_t fp8_val = A_data[global_r * K + global_k];
                float val = fp8_e4m3_to_float(fp8_val) * scale;
                sA[r * WMMA_K + c] = __float2half(val);
            } else {
                sA[r * WMMA_K + c] = __float2half(0.0f);
            }
        }

        // Load B tile: bf16 -> half
        for (int idx = lane; idx < WMMA_K * WMMA_N; idx += 32) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int global_k = k_tile + r;
            int global_c = col_start + c;

            if (global_k < K && global_c < N) {
                float bval = __bfloat162float(B[global_k * N + global_c]);
                sB[r * WMMA_N + c] = __float2half(bval);
            } else {
                sB[r * WMMA_N + c] = __float2half(0.0f);
            }
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, sA, WMMA_K);
        wmma::load_matrix_sync(b_frag, sB, WMMA_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    // Add dequantized C using a temporary buffer
    if (C_data != nullptr && C_scales != nullptr) {
        int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
        __shared__ float sC[WMMA_M * WMMA_N];

        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int global_r = row_start + r;
            int global_c = col_start + c;

            if (global_r < M && global_c < N) {
                int c_blk = global_c / MXFP8_BLOCK_SIZE;
                uint8_t c_scale_e8m0 = C_scales[global_r * num_blocks_n_c + c_blk];
                float c_scale = e8m0_to_float(c_scale_e8m0);
                uint8_t c_fp8 = C_data[global_r * N + global_c];
                sC[r * WMMA_N + c] = fp8_e4m3_to_float(c_fp8) * c_scale;
            } else {
                sC[r * WMMA_N + c] = 0.0f;
            }
        }
        __syncwarp();

        // Load C into a fragment and add
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::load_matrix_sync(c_frag, sC, WMMA_N, wmma::mem_row_major);

        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] += c_frag.x[i];
        }
    }

    // Store result
    if (row_start + WMMA_M <= M && col_start + WMMA_N <= N) {
        wmma::store_matrix_sync(D + row_start * N + col_start, acc_frag, N, wmma::mem_row_major);
    } else {
        // Edge case: store to shared then copy valid elements
        __shared__ float sD[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(sD, acc_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int global_r = row_start + r;
            int global_c = col_start + c;
            if (global_r < M && global_c < N) {
                D[global_r * N + global_c] = sD[r * WMMA_N + c];
            }
        }
    }
}

void gemm_fp8_tensorcore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                          const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                          const uint8_t* d_C_scales, float* d_D,
                          int M, int N, int K, cudaStream_t stream) {
    // One warp (32 threads) per 16x16 output tile
    dim3 block(32);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    gemm_fp8_tc_kernel<<<grid, block, 0, stream>>>(d_A_data, d_A_scales, d_B,
                                                    d_C_data, d_C_scales, d_D, M, N, K);
}
