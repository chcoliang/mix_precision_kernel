#include "gemm_kernels.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace nvcuda;

#define WM 16
#define WN 16
#define WK 16
#define BM 128
#define BN 128
#define BK 32

// ============================================================================
// Ablation A: No double buffering (single buffer)
// ============================================================================
__global__ void __launch_bounds__(256, 2)
gemm_ablation_no_double_buf(const uint8_t* __restrict__ A_data,
                             const uint8_t* __restrict__ A_scales,
                             const __nv_bfloat16* __restrict__ B,
                             float* __restrict__ D,
                             int M, int N, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / 4;
    int warp_n = warp_id % 4;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Single buffer (no double buffering)
    __shared__ half sA[BM * BK];
    __shared__ half sB[BK * BN];

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Load A
        for (int idx = tid; idx < BM * BK; idx += 256) {
            int r = idx / BK, c = idx % BK;
            int gr = block_row + r, gk = k_tile + c;
            if (gr < M && gk < K) {
                int blk = gk / MXFP8_BLOCK_SIZE;
                float scale = e8m0_to_float(A_scales[gr * num_blocks_k + blk]);
                sA[r * BK + c] = __float2half(fp8_e4m3_to_float(A_data[gr * K + gk]) * scale);
            } else {
                sA[r * BK + c] = __float2half(0.0f);
            }
        }
        // Load B
        for (int idx = tid; idx < BK * BN; idx += 256) {
            int r = idx / BN, c = idx % BN;
            int gk = k_tile + r, gc = block_col + c;
            if (gk < K && gc < N) {
                sB[r * BN + c] = __float2half(__bfloat162float(B[gk * N + gc]));
            } else {
                sB[r * BN + c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int kk = 0; kk < BK; kk += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag[2];
            for (int mi = 0; mi < 4; mi++)
                wmma::load_matrix_sync(a_frag[mi], &sA[(warp_m * 64 + mi * WM) * BK + kk], BK);
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(b_frag[ni], &sB[kk * BN + warp_n * 32 + ni * WN], BN);
            for (int mi = 0; mi < 4; mi++)
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    // Store
    __shared__ float sEpilogue[8][WM * WN];
    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int out_row = block_row + warp_m * 64 + mi * WM;
            int out_col = block_col + warp_n * 32 + ni * WN;
            if (out_row + WM <= M && out_col + WN <= N)
                wmma::store_matrix_sync(D + out_row * N + out_col, acc[mi][ni], N, wmma::mem_row_major);
            else {
                wmma::store_matrix_sync(sEpilogue[warp_id], acc[mi][ni], WN, wmma::mem_row_major);
                __syncwarp();
                int lane = tid % 32;
                for (int idx = lane; idx < WM * WN; idx += 32) {
                    int r = idx / WN, c = idx % WN;
                    if (out_row + r < M && out_col + c < N)
                        D[(out_row + r) * N + out_col + c] = sEpilogue[warp_id][idx];
                }
            }
        }
    }
}

// ============================================================================
// Ablation B: Vectorized loads (uint4 = 128 bits = 8 half values at once)
// ============================================================================
__global__ void __launch_bounds__(256, 2)
gemm_ablation_vectorized(const uint8_t* __restrict__ A_data,
                          const uint8_t* __restrict__ A_scales,
                          const __nv_bfloat16* __restrict__ B,
                          float* __restrict__ D,
                          int M, int N, int K) {
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / 4;
    int warp_n = warp_id % 4;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    __shared__ half sA[2][BM * BK];
    __shared__ half sB[2][BK * BN];
    int buf = 0;

    // Preload with vectorized access (process 4 elements at a time)
    auto load_tile = [&](int k_tile, int b) {
        // Load A: each thread processes 4 consecutive elements
        for (int idx = tid * 4; idx < BM * BK; idx += 256 * 4) {
            int r = idx / BK, c = idx % BK;
            int gr = block_row + r, gk = k_tile + c;
            if (gr < M && gk + 3 < K) {
                int blk = gk / MXFP8_BLOCK_SIZE;
                float scale = e8m0_to_float(A_scales[gr * num_blocks_k + blk]);
                // Load 4 bytes at once
                uint32_t packed = *reinterpret_cast<const uint32_t*>(&A_data[gr * K + gk]);
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed);
                for (int i = 0; i < 4; i++)
                    sA[b][(r) * BK + c + i] = __float2half(fp8_e4m3_to_float(bytes[i]) * scale);
            } else {
                for (int i = 0; i < 4 && idx + i < BM * BK; i++) {
                    int rr = (idx + i) / BK, cc = (idx + i) % BK;
                    int grr = block_row + rr, gkk = k_tile + cc;
                    if (grr < M && gkk < K) {
                        int blk2 = gkk / MXFP8_BLOCK_SIZE;
                        float sc = e8m0_to_float(A_scales[grr * num_blocks_k + blk2]);
                        sA[b][rr * BK + cc] = __float2half(fp8_e4m3_to_float(A_data[grr * K + gkk]) * sc);
                    } else {
                        sA[b][(idx + i) / BK * BK + (idx + i) % BK] = __float2half(0.0f);
                    }
                }
            }
        }
        // Load B: vectorized bf16 load (2 bf16 = 1 uint32)
        for (int idx = tid * 2; idx < BK * BN; idx += 256 * 2) {
            int r = idx / BN, c = idx % BN;
            int gk = k_tile + r, gc = block_col + c;
            if (gk < K && gc + 1 < N) {
                sB[b][r * BN + c] = __float2half(__bfloat162float(B[gk * N + gc]));
                sB[b][r * BN + c + 1] = __float2half(__bfloat162float(B[gk * N + gc + 1]));
            } else {
                for (int i = 0; i < 2 && idx + i < BK * BN; i++) {
                    int rr = (idx + i) / BN, cc = (idx + i) % BN;
                    int gkk = k_tile + rr, gcc = block_col + cc;
                    if (gkk < K && gcc < N)
                        sB[b][rr * BN + cc] = __float2half(__bfloat162float(B[gkk * N + gcc]));
                    else
                        sB[b][(idx + i) / BN * BN + (idx + i) % BN] = __float2half(0.0f);
                }
            }
        }
    };

    load_tile(0, 0);
    __syncthreads();

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        int next_buf = 1 - buf;
        if (k_tile + BK < K) load_tile(k_tile + BK, next_buf);

        for (int kk = 0; kk < BK; kk += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag[2];
            for (int mi = 0; mi < 4; mi++)
                wmma::load_matrix_sync(a_frag[mi], &sA[buf][(warp_m * 64 + mi * WM) * BK + kk], BK);
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(b_frag[ni], &sB[buf][kk * BN + warp_n * 32 + ni * WN], BN);
            for (int mi = 0; mi < 4; mi++)
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }
        __syncthreads();
        buf = next_buf;
    }

    // Store (simplified: assume aligned)
    for (int mi = 0; mi < 4; mi++)
        for (int ni = 0; ni < 2; ni++) {
            int out_row = block_row + warp_m * 64 + mi * WM;
            int out_col = block_col + warp_n * 32 + ni * WN;
            if (out_row + WM <= M && out_col + WN <= N)
                wmma::store_matrix_sync(D + out_row * N + out_col, acc[mi][ni], N, wmma::mem_row_major);
        }
}

// ============================================================================
// Benchmark runner
// ============================================================================
struct AblationResult {
    float time_ms;
    float tflops;
};

AblationResult run_ablation(const char* name, int variant,
                             uint8_t* dA, uint8_t* dAs, __nv_bfloat16* dB, float* dD,
                             int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warmup
    for (int i = 0; i < 5; i++) {
        switch (variant) {
            case 0: gemm_ablation_no_double_buf<<<grid, block>>>(dA, dAs, dB, dD, M, N, K); break;
            case 1: gemm_ablation_vectorized<<<grid, block>>>(dA, dAs, dB, dD, M, N, K); break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int runs = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        switch (variant) {
            case 0: gemm_ablation_no_double_buf<<<grid, block>>>(dA, dAs, dB, dD, M, N, K); break;
            case 1: gemm_ablation_vectorized<<<grid, block>>>(dA, dAs, dB, dD, M, N, K); break;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    AblationResult r;
    r.time_ms = total_ms / runs;
    r.tflops = (2.0 * M * N * K) / (r.time_ms * 1e9);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return r;
}

// Also time the original V0 kernel (just the compute part, without output quantize)
extern void gemm_fp8_tensorcore(const uint8_t*, const uint8_t*, const __nv_bfloat16*,
                                 const uint8_t*, const uint8_t*, uint8_t*, uint8_t*,
                                 int, int, int, cudaStream_t);

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Ablation Study: Optimization Techniques on FP8 TC Kernel       ║\n");
    printf("║  Shape: 4096x4096x4096, Gaussian(0,0.02)                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    int M = 4096, N = 4096, K = 4096;
    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<float> A_float(M * K);
    for (auto& v : A_float) v = dist(rng);
    std::vector<uint8_t> A_data(M * K), A_scales(M * num_blocks_k);
    mxfp8_quantize_host(A_float.data(), A_data.data(), A_scales.data(), M, K);

    std::vector<__nv_bfloat16> B_bf16(K * N);
    for (int i = 0; i < K * N; i++) B_bf16[i] = __float2bfloat16(dist(rng));

    uint8_t *dA, *dAs;
    __nv_bfloat16* dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K));
    CUDA_CHECK(cudaMalloc(&dAs, M * num_blocks_k));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dD, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A_data.data(), M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAs, A_scales.data(), M * num_blocks_k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B_bf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Run original V0 (compute only, no output quantize)
    {
        dim3 block(256);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        // Use the kernel from gemm_fp8_tensorcore.cu directly
        // We'll just time our ablation kernels and compare
    }

    printf("│ %-40s │ Time(ms) │ TFLOPS │\n", "Optimization Variant");
    printf("├──────────────────────────────────────────┼──────────┼────────┤\n");

    AblationResult r;
    r = run_ablation("No double buffer (single buf)", 0, dA, dAs, dB, dD, M, N, K);
    printf("│ %-40s │ %8.3f │ %6.2f │\n", "No double buffer (single buf)", r.time_ms, r.tflops);

    r = run_ablation("Vectorized loads (4-element)", 1, dA, dAs, dB, dD, M, N, K);
    printf("│ %-40s │ %8.3f │ %6.2f │\n", "Vectorized loads (4-elem pack)", r.time_ms, r.tflops);

    // Time original V0 kernel (with double buffer, no vectorize)
    {
        // Reuse existing kernel via the full API (includes output quantize overhead)
        uint8_t *dD_data, *dD_scales;
        int nbn = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
        CUDA_CHECK(cudaMalloc(&dD_data, M * N));
        CUDA_CHECK(cudaMalloc(&dD_scales, M * nbn));

        for (int i = 0; i < 5; i++)
            gemm_fp8_tensorcore(dA, dAs, dB, nullptr, nullptr, dD_data, dD_scales, M, N, K, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 50; i++)
            gemm_fp8_tensorcore(dA, dAs, dB, nullptr, nullptr, dD_data, dD_scales, M, N, K, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float tflops = (2.0 * M * N * K) / ((ms / 50) * 1e9);
        printf("│ %-40s │ %8.3f │ %6.2f │\n", "Original V0 (double buf + output quant)", ms / 50, tflops);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(dD_data);
        cudaFree(dD_scales);
    }

    printf("└──────────────────────────────────────────┴──────────┴────────┘\n");

    cudaFree(dA); cudaFree(dAs); cudaFree(dB); cudaFree(dD);
    return 0;
}
