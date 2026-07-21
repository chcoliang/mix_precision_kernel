#include "gemm_kernels.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"

// Variant 2: BF16 CUDA Core GEMM
// Strategy: Dequantize A from MXFP8 to float, keep B as bf16 (cast to float),
//           compute D = A * B + C using CUDA cores with FP32 accumulation.
// This preserves B's precision fully.

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

__global__ void gemm_bf16_cudacore_kernel(const uint8_t* __restrict__ A_data,
                                           const uint8_t* __restrict__ A_scales,
                                           const __nv_bfloat16* __restrict__ B,
                                           const uint8_t* __restrict__ C_data,
                                           const uint8_t* __restrict__ C_scales,
                                           float* __restrict__ D,
                                           int M, int N, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_row = ty;
    int thread_col = tx;

    // Each thread computes a 4x4 block of output
    float acc[4][4] = {{0.0f}};

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperatively load A tile (dequantize MXFP8)
        for (int idx = ty * blockDim.x + tx; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int global_r = row_base + r;
            int global_k = k_tile + c;

            if (global_r < M && global_k < K) {
                int blk_idx = global_k / MXFP8_BLOCK_SIZE;
                uint8_t scale_e8m0 = A_scales[global_r * num_blocks_k + blk_idx];
                float scale = e8m0_to_float(scale_e8m0);
                uint8_t fp8_val = A_data[global_r * K + global_k];
                sA[r][c] = fp8_e4m3_to_float(fp8_val) * scale;
            } else {
                sA[r][c] = 0.0f;
            }
        }

        // Cooperatively load B tile (bf16 -> float)
        for (int idx = ty * blockDim.x + tx; idx < TILE_K * TILE_N; idx += blockDim.x * blockDim.y) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int global_k = k_tile + r;
            int global_c = col_base + c;

            if (global_k < K && global_c < N) {
                sB[r][c] = __bfloat162float(B[global_k * N + global_c]);
            } else {
                sB[r][c] = 0.0f;
            }
        }
        __syncthreads();

        // Compute: each thread handles 4x4 output elements
        for (int ki = 0; ki < TILE_K; ki++) {
            for (int ri = 0; ri < 4; ri++) {
                for (int ci = 0; ci < 4; ci++) {
                    int local_r = thread_row * 4 + ri;
                    int local_c = thread_col * 4 + ci;
                    if (local_r < TILE_M && local_c < TILE_N) {
                        acc[ri][ci] += sA[local_r][ki] * sB[ki][local_c];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store results with C addition
    int num_blocks_n = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    for (int ri = 0; ri < 4; ri++) {
        for (int ci = 0; ci < 4; ci++) {
            int global_r = row_base + thread_row * 4 + ri;
            int global_c = col_base + thread_col * 4 + ci;

            if (global_r < M && global_c < N) {
                float result = acc[ri][ci];

                // Add dequantized C
                if (C_data != nullptr && C_scales != nullptr) {
                    int c_blk = global_c / MXFP8_BLOCK_SIZE;
                    uint8_t c_scale_e8m0 = C_scales[global_r * num_blocks_n + c_blk];
                    float c_scale = e8m0_to_float(c_scale_e8m0);
                    uint8_t c_fp8 = C_data[global_r * N + global_c];
                    result += fp8_e4m3_to_float(c_fp8) * c_scale;
                }

                D[global_r * N + global_c] = result;
            }
        }
    }
}

void gemm_bf16_cudacore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                         const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                         const uint8_t* d_C_scales, float* d_D,
                         int M, int N, int K, cudaStream_t stream) {
    // 16x16 threads, each handling 4x4 elements = 64x64 tile
    dim3 block(16, 16);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_bf16_cudacore_kernel<<<grid, block, 0, stream>>>(d_A_data, d_A_scales, d_B,
                                                           d_C_data, d_C_scales, d_D, M, N, K);
}
