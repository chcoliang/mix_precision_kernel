#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Variant 1: FP8 Tensor Core GEMM
// A (MXFP8) -> dequant to FP8 per-tensor, B (bf16) -> quantize to FP8, use FP8 tensor core
void gemm_fp8_tensorcore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                          const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                          const uint8_t* d_C_scales, float* d_D,
                          int M, int N, int K, cudaStream_t stream = 0);

// Variant 2: BF16 CUDA Core GEMM
// A (MXFP8) -> dequant to bf16, B stays bf16, compute with CUDA cores in FP32
void gemm_bf16_cudacore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                         const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                         const uint8_t* d_C_scales, float* d_D,
                         int M, int N, int K, cudaStream_t stream = 0);

// Variant 3: Tiled Mixed Precision GEMM
// A (MXFP8) -> dequant to bf16 per-tile, B (bf16), use BF16 tensor core
void gemm_mixed_tiled(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                       const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                       const uint8_t* d_C_scales, float* d_D,
                       int M, int N, int K, cudaStream_t stream = 0);
