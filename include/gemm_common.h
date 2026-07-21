#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                         \
    do {                                                                          \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

struct GemmDims {
    int M;
    int N;
    int K;
};

struct TimingResult {
    float elapsed_ms;
    float tflops;
    float bandwidth_gb_s;
};

inline TimingResult time_kernel(int M, int N, int K, float elapsed_ms) {
    TimingResult result;
    result.elapsed_ms = elapsed_ms;
    double flops = 2.0 * M * N * K;
    result.tflops = static_cast<float>(flops / (elapsed_ms * 1e9));
    result.bandwidth_gb_s = 0.0f;
    return result;
}

inline float compute_relative_error(const float* result, const float* reference, int size) {
    float max_rel_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float ref_val = reference[i];
        float res_val = result[i];
        if (fabsf(ref_val) > 1e-6f) {
            float rel_err = fabsf(res_val - ref_val) / fabsf(ref_val);
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
    }
    return max_rel_err;
}

constexpr int WARP_SIZE = 32;
constexpr int MXFP8_BLOCK_SIZE = 32;
