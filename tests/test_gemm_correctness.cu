#include "gemm_kernels.h"
#include "gemm_reference.h"
#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

struct TestCase {
    int M, N, K;
    const char* name;
};

void run_correctness_test(const TestCase& tc, float tolerance) {
    printf("\n--- %s: [%d x %d x %d] ---\n", tc.name, tc.M, tc.N, tc.K);

    int M = tc.M, N = tc.N, K = tc.K;
    int num_blocks_k_a = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    int num_blocks_n_c = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate random data
    std::vector<float> A_float(M * K), B_float(K * N), C_float(M * N);
    for (auto& v : A_float) v = dist(rng);
    for (auto& v : B_float) v = dist(rng);
    for (auto& v : C_float) v = dist(rng);

    // Quantize A to MXFP8
    std::vector<uint8_t> A_data(M * K), A_scales(M * num_blocks_k_a);
    mxfp8_quantize_host(A_float.data(), A_data.data(), A_scales.data(), M, K);

    // Dequantize A for reference
    std::vector<float> A_deq(M * K);
    mxfp8_dequantize_host(A_data.data(), A_scales.data(), A_deq.data(), M, K);

    // Convert B to bf16
    std::vector<__nv_bfloat16> B_bf16(K * N);
    for (int i = 0; i < K * N; i++) B_bf16[i] = __float2bfloat16(B_float[i]);

    // B in float (from bf16 for reference)
    std::vector<float> B_ref(K * N);
    for (int i = 0; i < K * N; i++) B_ref[i] = __bfloat162float(B_bf16[i]);

    // Quantize C to MXFP8
    std::vector<uint8_t> C_data(M * N), C_scales(M * num_blocks_n_c);
    mxfp8_quantize_host(C_float.data(), C_data.data(), C_scales.data(), M, N);

    // Dequantize C for reference
    std::vector<float> C_deq(M * N);
    mxfp8_dequantize_host(C_data.data(), C_scales.data(), C_deq.data(), M, N);

    // Reference: D_ref = A_deq * B_ref + C_deq
    std::vector<float> D_ref_fp32(M * N);
    gemm_reference_host(A_deq.data(), B_ref.data(), C_deq.data(), D_ref_fp32.data(), M, N, K);

    // Quantize reference to MXFP8 then dequantize (to match output format)
    int num_blocks_n_d = (N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    std::vector<uint8_t> D_ref_data(M * N), D_ref_scales(M * num_blocks_n_d);
    mxfp8_quantize_host(D_ref_fp32.data(), D_ref_data.data(), D_ref_scales.data(), M, N);
    std::vector<float> D_ref(M * N);
    mxfp8_dequantize_host(D_ref_data.data(), D_ref_scales.data(), D_ref.data(), M, N);

    // Allocate device memory
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

    std::vector<float> D_result(M * N);

    // Test each variant
    const char* variant_names[] = {"FP8 TensorCore", "BF16 CUDACore", "Mixed Tiled"};
    auto run_variant = [&](int variant) {
        CUDA_CHECK(cudaMemset(d_D_data, 0, M * N));
        CUDA_CHECK(cudaMemset(d_D_scales, 0, M * num_blocks_n_d));

        switch (variant) {
            case 0:
                gemm_fp8_tensorcore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales,
                                    d_D_data, d_D_scales, M, N, K);
                break;
            case 1:
                gemm_bf16_cudacore(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales,
                                   d_D_data, d_D_scales, M, N, K);
                break;
            case 2:
                gemm_mixed_tiled(d_A_data, d_A_scales, d_B, d_C_data, d_C_scales,
                                 d_D_data, d_D_scales, M, N, K);
                break;
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Dequantize output MXFP8 to float for comparison
        float* d_D_deq;
        CUDA_CHECK(cudaMalloc(&d_D_deq, M * N * sizeof(float)));
        mxfp8_dequantize_gpu(d_D_data, d_D_scales, d_D_deq, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(D_result.data(), d_D_deq, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_D_deq);

        float max_rel_err = compute_relative_error(D_result.data(), D_ref.data(), M * N);
        bool pass = max_rel_err < tolerance;
        printf("  %-18s: max_rel_err=%.6f  %s\n", variant_names[variant], max_rel_err,
               pass ? "PASS" : "FAIL");
    };

    for (int v = 0; v < 3; v++) {
        run_variant(v);
    }

    cudaFree(d_A_data);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_C_data);
    cudaFree(d_C_scales);
    cudaFree(d_D_data);
    cudaFree(d_D_scales);
}

int main() {
    printf("=== GEMM Correctness Tests ===\n");
    printf("Tolerance: 0.15 (15%% max relative error, accounts for fp16 intermediate + MXFP8 output)\n");

    TestCase tests[] = {
        {128, 128, 128, "Small"},
        {256, 256, 256, "Medium"},
        {512, 512, 512, "Large"},
        {256, 512, 1024, "Rectangular"},
        {1024, 1024, 1024, "1K cube"},
    };

    float tolerance = 0.15f;

    for (const auto& tc : tests) {
        run_correctness_test(tc, tolerance);
    }

    printf("\n=== Done ===\n");
    return 0;
}
