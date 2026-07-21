#include "gemm_kernels.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct BenchCase {
    int M, N, K;
    const char* name;
};

void benchmark_variant(const char* name, int variant,
                       uint8_t* d_A_data, uint8_t* d_A_scales,
                       __nv_bfloat16* d_B, uint8_t* d_C_data,
                       uint8_t* d_C_scales,
                       uint8_t* d_D_data, uint8_t* d_D_scales,
                       int M, int N, int K) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; i++) {
        switch (variant) {
            case 0: gemm_fp8_tensorcore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
            case 1: gemm_bf16_cudacore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
            case 2: gemm_mixed_tiled(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K); break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int num_runs = 100;
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
    float avg_ms = total_ms / num_runs;

    TimingResult tr = time_kernel(M, N, K, avg_ms);
    printf("  %-18s: %8.3f ms  %8.2f TFLOPS\n", name, avg_ms, tr.tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_benchmark(const BenchCase& bc) {
    int M = bc.M, N = bc.N, K = bc.K;
    printf("\n--- %s: [%d x %d x %d] ---\n", bc.name, M, N, K);

    int num_blocks_k_a = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    int num_blocks_n_d = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate and quantize data
    std::vector<float> A_float(M * K), C_float(M * N);
    for (auto& v : A_float) v = dist(rng);
    for (auto& v : C_float) v = dist(rng);

    std::vector<uint8_t> A_data(M * K), A_scales(M * num_blocks_k_a);
    std::vector<uint8_t> C_data(M * N), C_scales(M * num_blocks_n_c);
    mxfp8_quantize_host(A_float.data(), A_data.data(), A_scales.data(), M, K);
    mxfp8_quantize_host(C_float.data(), C_data.data(), C_scales.data(), M, N);

    std::vector<__nv_bfloat16> B_bf16(K * N);
    for (int i = 0; i < K * N; i++) B_bf16[i] = __float2bfloat16(dist(rng));

    // Device allocation
    uint8_t *d_A_data, *d_A_scales, *d_C_data, *d_C_scales;
    uint8_t *d_D_data, *d_D_scales;
    __nv_bfloat16* d_B;

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

    benchmark_variant("FP8 TensorCore", 0, d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K);
    benchmark_variant("BF16 CUDACore", 1, d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K);
    benchmark_variant("Mixed Tiled", 2, d_A_data, d_A_scales, d_B, d_C_data, d_C_scales, d_D_data, d_D_scales, M, N, K);

    cudaFree(d_A_data);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_C_data);
    cudaFree(d_C_scales);
    cudaFree(d_D_data);
    cudaFree(d_D_scales);
}

int main() {
    printf("=== GEMM Performance Benchmark ===\n");
    printf("GPU: H100, Warmup: 10, Runs: 100\n");

    BenchCase cases[] = {
        {512, 512, 512, "512 cube"},
        {1024, 1024, 1024, "1K cube"},
        {2048, 2048, 2048, "2K cube"},
        {4096, 4096, 4096, "4K cube"},
        {2048, 8192, 4096, "LLM-like (FFN)"},
        {4096, 4096, 11008, "LLaMA-7B FFN"},
    };

    for (const auto& bc : cases) {
        run_benchmark(bc);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
