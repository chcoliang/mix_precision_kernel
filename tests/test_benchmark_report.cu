#include "gemm_kernels.h"
#include "gemm_reference.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

// GPU kernel: convert bf16 array to float array
__global__ void bf16_to_float_kernel(const __nv_bfloat16* __restrict__ input,
                                      float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = __bfloat162float(input[idx]);
}

// Reference: pure bf16 computation path (GPU-accelerated)
// 1. Dequantize A (MXFP8) -> float on GPU
// 2. B bf16 -> float on GPU
// 3. Compute D = A * B + C using GPU reference GEMM (FP32 tiled)
void compute_bf16_reference_gpu(uint8_t* d_A_data, uint8_t* d_A_scales,
                                __nv_bfloat16* d_B, uint8_t* d_C_data,
                                uint8_t* d_C_scales, float* d_D_ref,
                                int M, int N, int K) {
    int num_blocks_k_a = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    // Dequantize A on GPU
    float* d_A_float;
    CUDA_CHECK(cudaMalloc(&d_A_float, M * K * sizeof(float)));
    mxfp8_dequantize_gpu(d_A_data, d_A_scales, d_A_float, M, K);

    // Convert B to float on GPU
    float* d_B_float;
    CUDA_CHECK(cudaMalloc(&d_B_float, K * N * sizeof(float)));
    int threads = 256;
    int blocks = (K * N + threads - 1) / threads;
    bf16_to_float_kernel<<<blocks, threads>>>(d_B, d_B_float, K * N);

    // Dequantize C on GPU
    float* d_C_float = nullptr;
    if (d_C_data != nullptr && d_C_scales != nullptr) {
        CUDA_CHECK(cudaMalloc(&d_C_float, M * N * sizeof(float)));
        mxfp8_dequantize_gpu(d_C_data, d_C_scales, d_C_float, M, N);
    }

    // GPU reference GEMM
    gemm_reference_gpu(d_A_float, d_B_float, d_C_float, d_D_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_A_float);
    cudaFree(d_B_float);
    if (d_C_float) cudaFree(d_C_float);
}

struct BenchResult {
    float time_ms;
    float tflops;
    float max_rel_err;
    float avg_rel_err;
    float rmse;
};

BenchResult run_variant(int variant, const char* name,
                        uint8_t* d_A_data, uint8_t* d_A_scales,
                        __nv_bfloat16* d_B, uint8_t* d_C_data,
                        uint8_t* d_C_scales,
                        uint8_t* d_D_data, uint8_t* d_D_scales,
                        const float* D_ref, int M, int N, int K) {
    BenchResult result;

    // Warmup
    int warmup = (M >= 16384) ? 1 : (M >= 8192) ? 2 : 5;
    for (int i = 0; i < warmup; i++) {
        switch (variant) {
            case 0: gemm_fp8_tensorcore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
            case 1: gemm_bf16_cudacore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
            case 2: gemm_mixed_tiled(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_runs = (M >= 16384) ? 3 : (M >= 8192) ? 10 : 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        switch (variant) {
            case 0: gemm_fp8_tensorcore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
            case 1: gemm_bf16_cudacore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
            case 2: gemm_mixed_tiled(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    result.time_ms = total_ms / num_runs;
    result.tflops = (2.0 * M * N * K) / (result.time_ms * 1e9);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Accuracy: dequantize MXFP8 output and compare against bf16 reference
    float* d_D_deq;
    CUDA_CHECK(cudaMalloc(&d_D_deq, M * N * sizeof(float)));
    mxfp8_dequantize_gpu(d_D_data, d_D_scales, d_D_deq, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> D_result(M * N);
    CUDA_CHECK(cudaMemcpy(D_result.data(), d_D_deq, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_D_deq);

    float max_rel = 0.0f, sum_rel = 0.0f, sum_sq = 0.0f;
    int valid_count = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(D_result[i] - D_ref[i]);
        float denom = fmaxf(fabsf(D_ref[i]), 1e-6f);
        float rel = diff / denom;
        max_rel = fmaxf(max_rel, rel);
        sum_rel += rel;
        sum_sq += diff * diff;
        valid_count++;
    }
    result.max_rel_err = max_rel;
    result.avg_rel_err = sum_rel / valid_count;
    result.rmse = sqrtf(sum_sq / valid_count);

    return result;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        Mixed Precision GEMM Benchmark (H100, SM90)                                 ║\n");
    printf("║  Reference: FP32 GEMM -> quantize to MXFP8 -> dequantize (gold standard)          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════════╝\n\n");

    // Real-world LLM shapes + large M scenarios
    struct Shape { int M; int N; int K; const char* name; };
    Shape shapes[] = {
        // LLaMA-7B (hidden=4096, ffn=11008)
        {4096, 12288, 4096, "LLaMA-7B QKV"},
        {4096, 11008, 4096, "LLaMA-7B FFN-up"},
        {4096, 4096, 11008, "LLaMA-7B FFN-down"},
        // LLaMA-70B (hidden=8192, ffn=28672)
        {2048, 28672, 8192, "LLaMA-70B FFN-up"},
        {2048, 8192, 28672, "LLaMA-70B FFN-down"},
        // Large M, small N/K
        {65536, 2048, 2048, "M=64K, N=2K, K=2K"},
        {131072, 2048, 2048, "M=128K, N=2K, K=2K"},
        // Square
        {8192, 8192, 8192, "Square 8K"},
        {16384, 16384, 16384, "Square 16K"},
    };
    const char* variant_names[] = {"FP8 TensorCore (WMMA)", "BF16 CUDA Core", "Mixed Tiled (WMMA)"};

    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);
    for (int si = 0; si < num_shapes; si++) {
        int M = shapes[si].M, N = shapes[si].N, K = shapes[si].K;
        const char* shape_name = shapes[si].name;
        int num_blocks_k_a = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
        int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

        printf("┌──────────────────────────────────────────────────────────────────────────────────────────┐\n");
        printf("│ %-40s [M=%d, N=%d, K=%d]\n", shape_name, M, N, K);
        printf("├─────────────────────┬───────────┬──────────┬───────────┬───────────┬───────────────┤\n");
        printf("│ Variant             │ Time(ms)  │ TFLOPS   │ MaxRelErr │ AvgRelErr │ RMSE          │\n");
        printf("├─────────────────────┼───────────┼──────────┼───────────┼───────────┼───────────────┤\n");

        // Generate data with Gaussian distribution (mean=0, std=0.02)
        std::mt19937 rng(42 + si);
        std::normal_distribution<float> dist(0.0f, 0.02f);

        std::vector<float> A_float(M * K), C_float(M * N);
        for (auto& v : A_float) v = dist(rng);
        for (auto& v : C_float) v = dist(rng);

        // Quantize A and C to MXFP8
        std::vector<uint8_t> A_data(M * K), A_scales(M * num_blocks_k_a);
        std::vector<uint8_t> C_data(M * N), C_scales(M * num_blocks_n_c);
        mxfp8_quantize_host(A_float.data(), A_data.data(), A_scales.data(), M, K);
        mxfp8_quantize_host(C_float.data(), C_data.data(), C_scales.data(), M, N);

        // Generate B in bf16
        std::vector<__nv_bfloat16> B_bf16(K * N);
        for (int i = 0; i < K * N; i++) B_bf16[i] = __float2bfloat16(dist(rng));

        // Allocate device memory
        uint8_t *d_A_data, *d_A_scales, *d_C_data, *d_C_scales;
        uint8_t *d_D_data, *d_D_scales;
        __nv_bfloat16* d_B;

        int num_blocks_n_d = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

        CUDA_CHECK(cudaMalloc(&d_A_data, M * K));
        CUDA_CHECK(cudaMalloc(&d_A_scales, M * num_blocks_k_a));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_C_data, M * N));
        CUDA_CHECK(cudaMalloc(&d_C_scales, M * num_blocks_n_c));
        CUDA_CHECK(cudaMalloc(&d_D_data, M * N));
        CUDA_CHECK(cudaMalloc(&d_D_scales, M * num_blocks_n_d));

        CUDA_CHECK(cudaMemcpy(d_A_data, A_data.data(), M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A_scales, A_scales.data(), M * num_blocks_k_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B_bf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_data, C_data.data(), M * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_scales, C_scales.data(), M * num_blocks_n_c, cudaMemcpyHostToDevice));

        // Reference: FP32 full-precision GEMM → quantize to MXFP8 → dequantize as gold standard
        float* d_D_ref_fp32;
        CUDA_CHECK(cudaMalloc(&d_D_ref_fp32, M * N * sizeof(float)));
        compute_bf16_reference_gpu(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_ref_fp32, M, N, K);

        // Quantize FP32 result to MXFP8, then dequantize → this is the gold standard
        uint8_t *d_ref_mxfp8_data, *d_ref_mxfp8_scales;
        CUDA_CHECK(cudaMalloc(&d_ref_mxfp8_data, M * N));
        CUDA_CHECK(cudaMalloc(&d_ref_mxfp8_scales, M * num_blocks_n_d));
        mxfp8_quantize_gpu(d_D_ref_fp32, d_ref_mxfp8_data, d_ref_mxfp8_scales, M, N);

        float* d_D_ref_deq;
        CUDA_CHECK(cudaMalloc(&d_D_ref_deq, M * N * sizeof(float)));
        mxfp8_dequantize_gpu(d_ref_mxfp8_data, d_ref_mxfp8_scales, d_D_ref_deq, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> D_ref(M * N);
        CUDA_CHECK(cudaMemcpy(D_ref.data(), d_D_ref_deq, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_D_ref_fp32);
        cudaFree(d_ref_mxfp8_data);
        cudaFree(d_ref_mxfp8_scales);
        cudaFree(d_D_ref_deq);

        // Skip BF16 CUDA Core when compute is too large (would take > 60s)
        for (int v = 0; v < 3; v++) {
            double flops = 2.0 * M * N * (double)K;
            if (v == 1 && flops > 2.0 * 16384.0 * 16384.0 * 16384.0) {
                printf("│ %-19s │  (skipped - too slow)                                       │\n",
                       variant_names[v]);
                continue;
            }
            BenchResult r = run_variant(v, variant_names[v],
                                        d_A_data, d_A_scales, d_B, d_C_data, d_C_scales,
                                        d_D_data, d_D_scales,
                                        D_ref.data(), M, N, K);
            printf("│ %-19s │ %9.3f │ %8.2f │ %9.6f │ %9.6f │ %13.6e │\n",
                   variant_names[v], r.time_ms, r.tflops, r.max_rel_err, r.avg_rel_err, r.rmse);
        }

        printf("└─────────────────────┴───────────┴──────────┴───────────┴───────────┴───────────────┘\n\n");
        cudaFree(d_A_data);
        cudaFree(d_A_scales);
        cudaFree(d_B);
        cudaFree(d_C_data);
        cudaFree(d_C_scales);
        cudaFree(d_D_data);
        cudaFree(d_D_scales);
    }

    printf("Notes:\n");
    printf("  - Reference: FP32 full-precision GEMM -> quantize to MXFP8 -> dequantize\n");
    printf("  - MaxRelErr/AvgRelErr: kernel MXFP8 output vs ideal FP32->MXFP8 output\n");
    printf("  - Error measures computation precision loss (not output quantization loss)\n");
    printf("  - Timing includes output MXFP8 quantization overhead\n");
    printf("  - Average of 50 runs after 5 warmup iterations\n");

    return 0;
}
