cmake_minimum_required(VERSION 3.20)
project(mix_precision_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3")

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

# Quantize library
add_library(mxfp8_quantize STATIC
    src/quantize/mxfp8_quantize.cu
    src/quantize/mxfp8_dequantize.cu
)

# Reference GEMM
add_library(gemm_reference STATIC
    src/reference/gemm_reference.cu
)

# Kernel variants
add_library(gemm_kernels STATIC
    src/kernels/gemm_fp8_tensorcore.cu
    src/kernels/gemm_bf16_cudacore.cu
    src/kernels/gemm_mixed_tiled.cu
)

# Tests
add_executable(test_mxfp8_quantize tests/test_mxfp8_quantize.cu)
target_link_libraries(test_mxfp8_quantize mxfp8_quantize CUDA::cublas)

add_executable(test_gemm_correctness tests/test_gemm_correctness.cu)
target_link_libraries(test_gemm_correctness gemm_kernels mxfp8_quantize gemm_reference CUDA::cublas)

add_executable(test_gemm_perf tests/test_gemm_perf.cu)
target_link_libraries(test_gemm_perf gemm_kernels mxfp8_quantize gemm_reference CUDA::cublas)

add_executable(test_benchmark_report tests/test_benchmark_report.cu)
target_link_libraries(test_benchmark_report gemm_kernels mxfp8_quantize gemm_reference CUDA::cublas)
