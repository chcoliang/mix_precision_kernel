#pragma once

#include "fp8_types.h"
#include "gemm_common.h"
#include <cstdint>

// E8M0 format: pure exponent, value = 2^(byte - 127)
__host__ __device__ inline float e8m0_to_float(uint8_t e8m0) {
    uint32_t f32_bits = static_cast<uint32_t>(e8m0) << 23;
    return *reinterpret_cast<float*>(&f32_bits);
}

__host__ __device__ inline uint8_t float_to_e8m0(float val) {
    if (val <= 0.0f) return 0;
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    uint8_t exponent = static_cast<uint8_t>((bits >> 23) & 0xFF);
    // Round up: if mantissa bits are non-zero, increment exponent
    if (bits & 0x007FFFFF) {
        exponent = (exponent < 255) ? exponent + 1 : 255;
    }
    return exponent;
}

struct MxFp8Data {
    uint8_t* data;       // FP8 E4M3 quantized values
    uint8_t* scales;     // E8M0 scale factors (1 per 32 elements along K)
    int M;
    int K;
    int num_blocks_k;    // ceil(K / 32)
};

// Quantize: float[M][K] -> MXFP8 (FP8 data + E8M0 scales)
// Block scaling along K dimension, block_size = 32
void mxfp8_quantize_host(const float* input, uint8_t* output_data, uint8_t* output_scales,
                          int M, int K);

// GPU quantize kernel
void mxfp8_quantize_gpu(const float* d_input, uint8_t* d_output_data, uint8_t* d_output_scales,
                         int M, int K, cudaStream_t stream = 0);

// Dequantize: MXFP8 -> float[M][K]
void mxfp8_dequantize_host(const uint8_t* input_data, const uint8_t* input_scales,
                            float* output, int M, int K);

// GPU dequantize kernel
void mxfp8_dequantize_gpu(const uint8_t* d_input_data, const uint8_t* d_input_scales,
                           float* d_output, int M, int K, cudaStream_t stream = 0);

// Quantize bf16 to FP8 E4M3 with per-tensor scaling
void bf16_to_fp8_per_tensor_gpu(const __nv_bfloat16* d_input, uint8_t* d_output,
                                 float* d_scale, int size, cudaStream_t stream = 0);
