#include "gemm_kernels.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// High-performance FP8 TensorCore GEMM
// Block tile: 128x128, K-tile: 32
// 8 warps per block (256 threads), each warp computes 32x64 via 2x4 WMMA tiles
// Double-buffered shared memory for A and B
// Cooperative loading: all 256 threads load A and B tiles

#define WM 16
#define WN 16
#define WK 16

// Block tile dimensions
#define BM 128
#define BN 128
#define BK 32

// Warp layout: 2 warps in M, 4 warps in N = 8 warps total
// Each warp: 2x2 WMMA tiles = 32x32 output
#define WARPS_M 2
#define WARPS_N 4
#define NUM_WARPS (WARPS_M * WARPS_N)
#define WARP_TILE_M (BM / WARPS_M)  // 64
#define WARP_TILE_N (BN / WARPS_N)  // 32

__global__ void __launch_bounds__(256, 2)
gemm_fp8_tc_kernel_v2(const uint8_t* __restrict__ A_data,
                       const uint8_t* __restrict__ A_scales,
                       const __nv_bfloat16* __restrict__ B,
                       const uint8_t* __restrict__ C_data,
                       const uint8_t* __restrict__ C_scales,
                       float* __restrict__ D,
                       int M, int N, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Warp position in block tile
    int warp_m = warp_id / WARPS_N;  // 0..1
    int warp_n = warp_id % WARPS_N;  // 0..3

    // Each warp computes WARP_TILE_M x WARP_TILE_N = 64x32
    // Using 4x2 = 8 WMMA fragments per warp
    // Accumulators: 4x2 WMMA fragments per warp
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Double-buffered shared memory
    // A: [BM][BK] = 128x32 half, B: [BK][BN] = 32x128 half
    __shared__ half sA[2][BM * BK];  // 2 * 128*32 * 2B = 16KB
    __shared__ half sB[2][BK * BN];  // 2 * 32*128 * 2B = 16KB

    int buf = 0;

    // Preload first tile
    {
        int k_tile = 0;
        // Load A: 128*32 = 4096 elements, 256 threads -> 16 per thread
        for (int idx = tid; idx < BM * BK; idx += 256) {
            int r = idx / BK;
            int c = idx % BK;
            int global_r = block_row + r;
            int global_k = k_tile + c;

            if (global_r < M && global_k < K) {
                int blk_idx = global_k / MXFP8_BLOCK_SIZE;
                uint8_t scale_e8m0 = A_scales[global_r * num_blocks_k + blk_idx];
                float scale = e8m0_to_float(scale_e8m0);
                uint8_t fp8_val = A_data[global_r * K + global_k];
                sA[0][r * BK + c] = __float2half(fp8_e4m3_to_float(fp8_val) * scale);
            } else {
                sA[0][r * BK + c] = __float2half(0.0f);
            }
        }

        // Load B: 32*128 = 4096 elements, 256 threads -> 16 per thread
        for (int idx = tid; idx < BK * BN; idx += 256) {
            int r = idx / BN;
            int c = idx % BN;
            int global_k = k_tile + r;
            int global_c = block_col + c;

            if (global_k < K && global_c < N) {
                sB[0][r * BN + c] = __float2half(__bfloat162float(B[global_k * N + global_c]));
            } else {
                sB[0][r * BN + c] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // Main K-loop with double buffering
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        int next_k = k_tile + BK;
        int next_buf = 1 - buf;

        // Async load next tile while computing current
        if (next_k < K) {
            for (int idx = tid; idx < BM * BK; idx += 256) {
                int r = idx / BK;
                int c = idx % BK;
                int global_r = block_row + r;
                int global_k = next_k + c;

                if (global_r < M && global_k < K) {
                    int blk_idx = global_k / MXFP8_BLOCK_SIZE;
                    uint8_t scale_e8m0 = A_scales[global_r * num_blocks_k + blk_idx];
                    float scale = e8m0_to_float(scale_e8m0);
                    uint8_t fp8_val = A_data[global_r * K + global_k];
                    sA[next_buf][r * BK + c] = __float2half(fp8_e4m3_to_float(fp8_val) * scale);
                } else {
                    sA[next_buf][r * BK + c] = __float2half(0.0f);
                }
            }

            for (int idx = tid; idx < BK * BN; idx += 256) {
                int r = idx / BN;
                int c = idx % BN;
                int global_k = next_k + r;
                int global_c = block_col + c;

                if (global_k < K && global_c < N) {
                    sB[next_buf][r * BN + c] = __float2half(__bfloat162float(B[global_k * N + global_c]));
                } else {
                    sB[next_buf][r * BN + c] = __float2half(0.0f);
                }
            }
        }

        // Compute: iterate over K-dim in WMMA_K steps within BK
        for (int kk = 0; kk < BK; kk += WK) {
            // Load A and B fragments for this warp
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag[2];

            // Load A fragments: 4 rows of 16x16 for this warp
            for (int mi = 0; mi < 4; mi++) {
                int frag_row = warp_m * WARP_TILE_M + mi * WM;
                wmma::load_matrix_sync(a_frag[mi],
                    &sA[buf][frag_row * BK + kk], BK);
            }

            // Load B fragments: 2 columns of 16x16 for this warp
            for (int ni = 0; ni < 2; ni++) {
                int frag_col = warp_n * WARP_TILE_N + ni * WN;
                wmma::load_matrix_sync(b_frag[ni],
                    &sB[buf][kk * BN + frag_col], BN);
            }

            // MMA: 4x2 accumulations
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 2; ni++) {
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
        buf = next_buf;
    }

    // Epilogue: Add C and store D
    // Use per-warp region of shared memory to avoid conflicts
    int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    __shared__ float sEpilogue[NUM_WARPS][WM * WN];

    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int out_row = block_row + warp_m * WARP_TILE_M + mi * WM;
            int out_col = block_col + warp_n * WARP_TILE_N + ni * WN;

            if (out_row >= M || out_col >= N) continue;

            // Add C
            if (C_data != nullptr && C_scales != nullptr) {
                for (int idx = lane; idx < WM * WN; idx += 32) {
                    int r = idx / WN;
                    int c = idx % WN;
                    int gr = out_row + r;
                    int gc = out_col + c;
                    if (gr < M && gc < N) {
                        int c_blk = gc / MXFP8_BLOCK_SIZE;
                        float cs = e8m0_to_float(C_scales[gr * num_blocks_n_c + c_blk]);
                        sEpilogue[warp_id][idx] = fp8_e4m3_to_float(C_data[gr * N + gc]) * cs;
                    } else {
                        sEpilogue[warp_id][idx] = 0.0f;
                    }
                }
                __syncwarp();

                wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
                wmma::load_matrix_sync(c_frag, sEpilogue[warp_id], WN, wmma::mem_row_major);
                for (int i = 0; i < acc[mi][ni].num_elements; i++) {
                    acc[mi][ni].x[i] += c_frag.x[i];
                }
            }

            // Store to global memory
            if (out_row + WM <= M && out_col + WN <= N) {
                wmma::store_matrix_sync(D + out_row * N + out_col, acc[mi][ni], N, wmma::mem_row_major);
            } else {
                wmma::store_matrix_sync(sEpilogue[warp_id], acc[mi][ni], WN, wmma::mem_row_major);
                __syncwarp();
                for (int idx = lane; idx < WM * WN; idx += 32) {
                    int r = idx / WN;
                    int c = idx % WN;
                    int gr = out_row + r;
                    int gc = out_col + c;
                    if (gr < M && gc < N) {
                        D[gr * N + gc] = sEpilogue[warp_id][idx];
                    }
                }
            }
        }
    }
}

void gemm_fp8_tensorcore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                          const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                          const uint8_t* d_C_scales, float* d_D,
                          int M, int N, int K, cudaStream_t stream) {
    dim3 block(256);  // 8 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_fp8_tc_kernel_v2<<<grid, block, 0, stream>>>(d_A_data, d_A_scales, d_B,
                                                       d_C_data, d_C_scales, d_D, M, N, K);
}
