# Mixed Precision GEMM Kernels (MXFP8 + BF16) for NVIDIA H100

混合精度 GEMM CUDA 算子实现，支持 MXFP8 (Microscaling FP8) 输入和 BF16 权重，针对 NVIDIA H100 (SM90, Hopper) GPU 优化。

## 概述

实现 `D = A * B + C` 运算，其中：
- **A** (输入激活): MXFP8 格式 (FP8 E4M3 数据 + E8M0 block scaling, 每32个元素一个 scale)
- **B** (权重): BF16 格式
- **C** (偏置): MXFP8 格式
- **D** (输出): FP32 格式

## 三种 Kernel 实现方案

| 方案 | 策略 | Tensor Core | 特点 |
|------|------|-------------|------|
| **FP8 TensorCore** | 8-warp, 128×128 block tile, 双缓冲 K-loop, 每 warp 4×2 WMMA | ✅ FP16 WMMA | 最高吞吐，大矩阵最优 |
| **BF16 CUDA Core** | A 反量化为 float，B 转 float，FMA 累加 | ❌ 纯 CUDA Core | 最高精度，B 完全保留 |
| **Mixed Tiled** | 4-warp 分块，A 反量化为 half，B 转 half，WMMA | ✅ FP16 WMMA (4-warp) | 均衡方案 |

## Benchmark 结果 (H100 80GB HBM3)

**输入**: A [M,K] MXFP8, B [K,N] BF16, C [M,N] MXFP8  
**输出**: D [M,N] MXFP8 (含输出量化开销)  
**精度标准**: 纯 BF16 计算路径（A 反量化→FP32 matmul→FP32 累加，不做输出量化）作为 ground truth

### 512 x 512 x 512

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 0.278 | 0.97 | 1.000000 | 0.022569 | 2.01e-01 |
| BF16 CUDA Core | 0.484 | 0.55 | 1.000000 | 0.022569 | 2.01e-01 |
| Mixed Tiled (WMMA) | 0.231 | 1.16 | 1.000000 | 0.022569 | 2.01e-01 |

### 1024 x 1024 x 1024

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 0.438 | 4.90 | 1.000000 | 0.022533 | 2.83e-01 |
| BF16 CUDA Core | 1.638 | 1.31 | 1.000000 | 0.022533 | 2.83e-01 |
| Mixed Tiled (WMMA) | 0.536 | 4.01 | 1.000000 | 0.022533 | 2.83e-01 |

### 2048 x 2048 x 2048

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 2.595 | 6.62 | 1.048000 | 0.022501 | 4.00e-01 |
| BF16 CUDA Core | 10.204 | 1.68 | 1.000000 | 0.022501 | 4.00e-01 |
| Mixed Tiled (WMMA) | 2.568 | 6.69 | 1.048000 | 0.022501 | 4.00e-01 |

### 4096 x 4096 x 4096

| Variant | Time(ms) | TFLOPS | MaxRelErr | AvgRelErr | RMSE |
|---------|----------|--------|-----------|-----------|------|
| FP8 TensorCore (WMMA) | 10.306 | **13.34** | 1.197425 | 0.022175 | 5.56e-01 |
| BF16 CUDA Core | 75.693 | 1.82 | 1.000000 | 0.022175 | 5.56e-01 |
| Mixed Tiled (WMMA) | 18.203 | 7.55 | 1.197425 | 0.022175 | 5.56e-01 |

### 结果分析

**性能:**
1. **FP8 TensorCore** 在大矩阵(4K)达到 **13.34 TFLOPS**，是最快方案
2. **Mixed Tiled** 达到 7.55 TFLOPS，在中小矩阵(512~2K)与 FP8 TensorCore 接近
3. **BF16 CUDA Core** 约 1.8 TFLOPS，不使用 tensor core 故最慢

**精度:**
1. **三种方案精度完全一致** (AvgRelErr ≈ 2.25%)，误差来源是 MXFP8 输出量化本身的固有精度损失
2. **MaxRelErr ≈ 1.0** 出现在 FP32 结果接近零的元素处：量化后为零，relative error = 1.0
3. **AvgRelErr ≈ 2.25%** 是 FP8 E4M3 (3-bit mantissa) 的理论量化误差水平
4. WMMA 变体在 2K/4K 有轻微额外 MaxRelErr (1.05~1.20) 来自 fp16 中间精度舍入

**结论: 精度由输出 MXFP8 格式决定（而非计算路径），选择 FP8 TensorCore 可获得最佳性能且不牺牲精度。**

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
│   │   ├── gemm_fp8_tensorcore.cu      # 方案1: 8-warp 128×128 双缓冲 WMMA
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
