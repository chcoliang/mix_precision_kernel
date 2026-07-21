#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__host__ __device__ inline float fp8_e4m3_to_float(uint8_t val) {
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<uint8_t*>(&fp8) = val;
    return static_cast<float>(static_cast<__half>(fp8));
}

__host__ __device__ inline uint8_t float_to_fp8_e4m3(float val) {
    if (val != val) return 0x7F;
    const float max_fp8 = 448.0f;
    if (val > max_fp8) val = max_fp8;
    if (val < -max_fp8) val = -max_fp8;
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(val);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

__host__ __device__ inline float fp8_e5m2_to_float(uint8_t val) {
    __nv_fp8_e5m2 fp8;
    *reinterpret_cast<uint8_t*>(&fp8) = val;
    return static_cast<float>(static_cast<__half>(fp8));
}

__host__ __device__ inline uint8_t float_to_fp8_e5m2(float val) {
    __nv_fp8_e5m2 fp8 = __nv_fp8_e5m2(val);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

__host__ __device__ inline float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__host__ __device__ inline __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}
