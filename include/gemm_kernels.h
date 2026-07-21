#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// All kernels: D = A * B + C
// Input:  A [M,K] MXFP8, B [K,N] bf16, C [M,N] MXFP8
// Output: D [M,N] MXFP8 (d_D_data: FP8 E4M3, d_D_scales: E8M0, block_size=32 along N)

// Variant 1: FP8 Tensor Core GEMM (8-warp, 128x128 block tile, double-buffered)
void gemm_fp8_tensorcore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                          const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                          const uint8_t* d_C_scales,
                          uint8_t* d_D_data, uint8_t* d_D_scales,
                          int M, int N, int K, cudaStream_t stream = 0);

// Variant 2: BF16 CUDA Core GEMM
void gemm_bf16_cudacore(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                         const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                         const uint8_t* d_C_scales,
                         uint8_t* d_D_data, uint8_t* d_D_scales,
                         int M, int N, int K, cudaStream_t stream = 0);

// Variant 3: Tiled Mixed Precision GEMM (4-warp, 32x32 block tile)
void gemm_mixed_tiled(const uint8_t* d_A_data, const uint8_t* d_A_scales,
                       const __nv_bfloat16* d_B, const uint8_t* d_C_data,
                       const uint8_t* d_C_scales,
                       uint8_t* d_D_data, uint8_t* d_D_scales,
                       int M, int N, int K, cudaStream_t stream = 0);
