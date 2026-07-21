#include "mxfp8_utils.h"
#include "gemm_common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

bool test_quantize_roundtrip(int M, int K, float tolerance) {
    printf("  Test quantize roundtrip [%d x %d] tolerance=%.4f ... ", M, K, tolerance);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> input(M * K);
    for (auto& v : input) v = dist(rng);

    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
    std::vector<uint8_t> quant_data(M * K);
    std::vector<uint8_t> quant_scales(M * num_blocks_k);
    std::vector<float> output(M * K);

    // Host quantize -> dequantize
    mxfp8_quantize_host(input.data(), quant_data.data(), quant_scales.data(), M, K);
    mxfp8_dequantize_host(quant_data.data(), quant_scales.data(), output.data(), M, K);

    // Check relative error
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * K; i++) {
        if (fabsf(input[i]) > 1e-4f) {
            float rel_err = fabsf(output[i] - input[i]) / fabsf(input[i]);
            max_rel_err = fmaxf(max_rel_err, rel_err);
        }
    }

    if (max_rel_err < tolerance) {
        printf("PASS (max_rel_err=%.6f)\n", max_rel_err);
        return true;
    } else {
        printf("FAIL (max_rel_err=%.6f)\n", max_rel_err);
        return false;
    }
}

bool test_quantize_gpu_vs_host(int M, int K) {
    printf("  Test GPU vs Host quantize [%d x %d] ... ", M, K);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> input(M * K);
    for (auto& v : input) v = dist(rng);

    int num_blocks_k = (K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

    // Host version
    std::vector<uint8_t> host_data(M * K);
    std::vector<uint8_t> host_scales(M * num_blocks_k);
    mxfp8_quantize_host(input.data(), host_data.data(), host_scales.data(), M, K);

    // GPU version
    float* d_input;
    uint8_t* d_data;
    uint8_t* d_scales;
    CUDA_CHECK(cudaMalloc(&d_input, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data, M * K));
    CUDA_CHECK(cudaMalloc(&d_scales, M * num_blocks_k));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    mxfp8_quantize_gpu(d_input, d_data, d_scales, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> gpu_data(M * K);
    std::vector<uint8_t> gpu_scales(M * num_blocks_k);
    CUDA_CHECK(cudaMemcpy(gpu_data.data(), d_data, M * K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_scales.data(), d_scales, M * num_blocks_k, cudaMemcpyDeviceToHost));

    // Compare scales
    int scale_mismatches = 0;
    for (int i = 0; i < M * num_blocks_k; i++) {
        if (host_scales[i] != gpu_scales[i]) scale_mismatches++;
    }

    // Compare data (allow +-1 due to rounding)
    int data_mismatches = 0;
    for (int i = 0; i < M * K; i++) {
        int diff = abs((int)host_data[i] - (int)gpu_data[i]);
        if (diff > 1) data_mismatches++;
    }

    cudaFree(d_input);
    cudaFree(d_data);
    cudaFree(d_scales);

    if (scale_mismatches == 0 && data_mismatches == 0) {
        printf("PASS\n");
        return true;
    } else {
        printf("FAIL (scale_mismatches=%d, data_mismatches=%d)\n", scale_mismatches, data_mismatches);
        return false;
    }
}

bool test_edge_cases() {
    printf("  Test edge cases ... ");

    float zeros[32] = {0};
    uint8_t data[32], scales[1];
    float output[32];

    mxfp8_quantize_host(zeros, data, scales, 1, 32);
    mxfp8_dequantize_host(data, scales, output, 1, 32);

    for (int i = 0; i < 32; i++) {
        if (output[i] != 0.0f) {
            printf("FAIL (zero roundtrip)\n");
            return false;
        }
    }

    // Test max values
    float maxvals[32];
    for (int i = 0; i < 32; i++) maxvals[i] = 400.0f;
    mxfp8_quantize_host(maxvals, data, scales, 1, 32);
    mxfp8_dequantize_host(data, scales, output, 1, 32);

    float rel_err = fabsf(output[0] - 400.0f) / 400.0f;
    if (rel_err > 0.1f) {
        printf("FAIL (max value rel_err=%.4f)\n", rel_err);
        return false;
    }

    printf("PASS\n");
    return true;
}

int main() {
    printf("=== MXFP8 Quantization Tests ===\n\n");

    int passed = 0, total = 0;

    total++; if (test_quantize_roundtrip(64, 128, 0.25f)) passed++;
    total++; if (test_quantize_roundtrip(256, 512, 0.25f)) passed++;
    total++; if (test_quantize_roundtrip(128, 100, 0.25f)) passed++;  // non-aligned K
    total++; if (test_quantize_gpu_vs_host(128, 256)) passed++;
    total++; if (test_quantize_gpu_vs_host(512, 1024)) passed++;
    total++; if (test_edge_cases()) passed++;

    printf("\n=== Results: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
