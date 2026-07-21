#pragma once

#include <cuda_runtime.h>

// D = A * B + C, all in FP32 (for correctness reference)
// A: [M, K], B: [K, N], C: [M, N], D: [M, N]
void gemm_reference_gpu(const float* d_A, const float* d_B, const float* d_C,
                         float* d_D, int M, int N, int K, cudaStream_t stream = 0);

// Host reference
void gemm_reference_host(const float* A, const float* B, const float* C,
                          float* D, int M, int N, int K);
