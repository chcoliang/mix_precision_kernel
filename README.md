# Mixed Precision GEMM Kernels (MXFP8 + BF16) for NVIDIA H100

混合精度 GEMM CUDA 算子实现，支持 MXFP8 (Microscaling FP8) 输入和 BF16 权重，针对 NVIDIA H100 (SM90, Hopper) GPU 优化。

## 概述

实现 `D = A * B + C` 运算，其中：
- **A** (输入激活): MXFP8 格式 (FP8 E4M3 数据 + E8M0 block scaling, 每32个元素一个 scale)
- **B** (权重): BF16 格式
- **C** (偏置): MXFP8 格式
- **D** (输出): FP32 格式

## 三种 Kernel 实现方案

| 方案 | 策略 | Tensor Core | 精度保留 |
|------|------|-------------|----------|
| **FP8 TensorCore** | A 反量化为 half，B 转 half，WMMA fp16 MMA | ✅ FP16 WMMA | A/B 均经过 fp16 转换 |
| **BF16 CUDA Core** | A 反量化为 float，B 转 float，FMA 累加 | ❌ 纯 CUDA Core | 最高精度，B 完全保留 |
| **Mixed Tiled** | 多 warp 分块，A 反量化为 half，B 转 half，WMMA | ✅ FP16 WMMA (4-warp) | 同 FP8 TensorCore |

## Benchmark 结果 (H100 80GB HBM3)

精度标准：**纯 BF16 计算路径**（A 反量化到 float → B float → FP32 累加）作为 ground truth。

### 512 x 512 x 512

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 0.082 | 3.26 | 0.000945 | 0.000000 | 9.09e-07 |
| BF16 CUDA Core | 0.311 | 0.86 | 0.000000 | 0.000000 | 0.00e+00 |
| Mixed Tiled (WMMA) | 0.057 | 4.71 | 0.000945 | 0.000000 | 9.09e-07 |

### 1024 x 1024 x 1024

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 0.515 | 4.17 | 0.017094 | 0.000001 | 2.37e-06 |
| BF16 CUDA Core | 1.425 | 1.51 | 0.000000 | 0.000000 | 0.00e+00 |
| Mixed Tiled (WMMA) | 0.316 | 6.80 | 0.017094 | 0.000001 | 2.37e-06 |

### 2048 x 2048 x 2048

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 3.394 | 5.06 | 0.953674 | 0.000002 | 6.92e-06 |
| BF16 CUDA Core | 9.772 | 1.76 | 0.000000 | 0.000000 | 0.00e+00 |
| Mixed Tiled (WMMA) | 2.218 | 7.75 | 0.953674 | 0.000002 | 6.92e-06 |

### 4096 x 4096 x 4096

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 27.089 | 5.07 | 2.384186 | 0.000004 | 2.25e-05 |
| BF16 CUDA Core | 74.593 | 1.84 | 0.000000 | 0.000000 | 0.00e+00 |
| Mixed Tiled (WMMA) | 17.169 | 8.01 | 2.384186 | 0.000004 | 2.25e-05 |

### 结果分析

1. **性能排序**: Mixed Tiled (8 TFLOPS) > FP8 TensorCore (5 TFLOPS) > BF16 CUDA Core (1.8 TFLOPS)
2. **精度排序**: BF16 CUDA Core (完全精确) > WMMA 变体 (fp16 转换引入少量舍入误差)
3. **MaxRelErr 说明**: WMMA 变体的大 MaxRelErr 仅出现在个别极小值元素处（除法放大），平均相对误差 (AvgRelErr) 极小 (< 0.0004%)
4. **Mixed Tiled 最优**: 利用 4-warp 并行和共享内存分块，WMMA 利用率高，比单 warp 版本快 ~60%

## MXFP8 格式说明

```
MXFP8 = FP8 E4M3 数据 + E8M0 Block Scaling

Block Size: 32 elements (沿 K 维度)
Scale Format: E8M0 (纯指数字节, value = 2^(byte - 127))

量化流程:
  1. 计算 block 内 amax = max(|vals[0:31]|)
  2. scale = round_up_to_power_of_2(amax / 448.0)  → 存为 E8M0
  3. output[i] = round_to_nearest_even(vals[i] / scale) → 存为 FP8 E4M3

反量化:
  output[i] = fp8_to_float(data[i]) * e8m0_to_float(scale)
```

## 项目结构

```
mix_precision_kernel/
├── CMakeLists.txt                      # CMake 构建 (SM90a, CUDA 12.4)
├── include/
│   ├── fp8_types.h                     # FP8 E4M3/E5M2 类型转换
│   ├── gemm_common.h                   # 通用宏、工具函数
│   ├── gemm_kernels.h                  # Kernel 接口声明
│   ├── gemm_reference.h               # FP32 参考实现接口
│   └── mxfp8_utils.h                  # MXFP8 量化/反量化接口
├── src/
│   ├── kernels/
│   │   ├── gemm_fp8_tensorcore.cu      # 方案1: FP16 WMMA TensorCore
│   │   ├── gemm_bf16_cudacore.cu       # 方案2: CUDA Core FP32 累加
│   │   └── gemm_mixed_tiled.cu         # 方案3: 4-warp 分块 WMMA
│   ├── quantize/
│   │   ├── mxfp8_quantize.cu           # MXFP8 量化 (GPU + Host)
│   │   └── mxfp8_dequantize.cu         # MXFP8 反量化 (GPU + Host)
│   └── reference/
│       └── gemm_reference.cu           # FP32 tiled GPU 参考 GEMM
├── tests/
│   ├── test_mxfp8_quantize.cu          # 量化正确性单元测试
│   ├── test_gemm_correctness.cu        # 三种方案正确性测试
│   ├── test_gemm_perf.cu              # 性能基准测试
│   └── test_benchmark_report.cu        # 精度+性能对比报告
└── scripts/
    ├── build.sh                        # 一键构建
    └── run_tests.sh                    # 一键测试
```

## 构建与运行

### 环境要求
- NVIDIA H100 GPU (SM90)
- CUDA Toolkit 12.4+
- CMake 3.20+
- GCC 11+

### 构建
```bash
bash scripts/build.sh
```

### 运行测试
```bash
# 运行全部测试
bash scripts/run_tests.sh

# 单独运行精度+性能对比报告
./build/test_benchmark_report
```

### 测试说明
- `test_mxfp8_quantize`: MXFP8 量化/反量化 roundtrip 正确性 (GPU vs Host)
- `test_gemm_correctness`: 三种 Kernel 变体与 FP32 参考结果对比
- `test_gemm_perf`: 性能基准 (512~11K 矩阵, 含 LLM 典型 shape)
- `test_benchmark_report`: 以 bf16 计算路径为标准的精度+性能完整报告

## 后续优化方向

1. 使用 PTX 内联 `wgmma.mma_async` 指令替代 WMMA API (预期 2-3x 提升)
2. 实现 TMA (Tensor Memory Accelerator) 异步数据加载
3. 添加 persistent kernel + stream-K 分解
4. 集成 cuBLAS FP8 接口作为对比基线
5. 在 Blackwell (SM100) 上使用硬件原生 MXFP8 支持
