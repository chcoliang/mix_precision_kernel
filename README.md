# Mixed Precision GEMM Kernels (MXFP8 + BF16) for NVIDIA H100

混合精度 GEMM CUDA 算子实现，支持 MXFP8 (Microscaling FP8) 输入和 BF16 权重，针对 NVIDIA H100 (SM90, Hopper) GPU 优化。

## 概述

实现 `D = A * B + C` 运算，其中：
- **A** (输入激活): MXFP8 格式 (FP8 E4M3 数据 + E8M0 block scaling, 每32个元素一个 scale)
- **B** (权重): BF16 格式
- **C** (偏置): MXFP8 格式
- **D** (输出): MXFP8 格式 (FP8 E4M3 数据 + E8M0 block scaling, 每32个元素一个 scale)

## 三种 Kernel 实现方案

| 方案 | 策略 | Tensor Core | 特点 |
|------|------|-------------|------|
| **FP8 TensorCore** | 8-warp, 128×128 block tile, 双缓冲 K-loop, 每 warp 4×2 WMMA | ✅ FP16 WMMA | 最高吞吐，大矩阵最优 |
| **BF16 CUDA Core** | A 反量化为 float，B 转 float，FMA 累加 | ❌ 纯 CUDA Core | 最高精度，B 完全保留 |
| **Mixed Tiled** | 4-warp 分块，A 反量化为 half，B 转 half，WMMA | ✅ FP16 WMMA (4-warp) | 均衡方案 |

## Benchmark 结果 (H100 80GB HBM3)

**输入**: A [M,K] MXFP8, B [K,N] BF16, C [M,N] MXFP8  
**输出**: D [M,N] MXFP8 (含输出量化开销)  
**精度标准**: **FP32 全精度计算 → 量化到 MXFP8** 作为 gold standard（消除输出量化噪声，仅测量计算精度损失）

### LLM 实际 Shape 测试

#### LLaMA-7B (hidden=4096, ffn=11008, seq=4096)

| Shape | M | N | K | FP8 TC (ms) | FP8 TC TFLOPS | Mixed Tiled TFLOPS | BF16 Core TFLOPS | AvgRelErr |
|-------|---|---|---|------------|---------------|-------------------|-----------------|-----------|
| QKV projection | 4096 | 12288 | 4096 | 29.4 | **14.01** | 7.46 | 1.83 | 0.004% |
| O-projection | 4096 | 4096 | 4096 | 10.5 | **13.13** | 7.57 | 1.82 | 0.000% |
| FFN gate-up | 4096 | 11008 | 4096 | 26.1 | **14.16** | 7.49 | 1.84 | 0.004% |
| FFN down | 4096 | 4096 | 11008 | 26.4 | **14.01** | 7.64 | 1.82 | 0.010% |

#### LLaMA-70B (hidden=8192, ffn=28672, seq=2048)

| Shape | M | N | K | FP8 TC (ms) | FP8 TC TFLOPS | Mixed Tiled TFLOPS | BF16 Core TFLOPS | AvgRelErr |
|-------|---|---|---|------------|---------------|-------------------|-----------------|-----------|
| QKV projection | 2048 | 8192 | 8192 | 19.7 | **13.98** | 7.50 | 1.82 | 0.017% |
| FFN gate-up | 2048 | 28672 | 8192 | 64.7 | **14.87** | 7.65 | 1.84 | 0.016% |
| FFN down | 2048 | 8192 | 28672 | 65.7 | **14.65** | 7.65 | 1.83 | 0.140% |

#### 大 Batch 推理 (batch_size × seq_len = 32768)

| Shape | M | N | K | FP8 TC (ms) | FP8 TC TFLOPS | Mixed Tiled TFLOPS | BF16 Core TFLOPS | AvgRelErr |
|-------|---|---|---|------------|---------------|-------------------|-----------------|-----------|
| Linear (h=4K) | 32768 | 4096 | 4096 | 78.8 | **13.95** | 7.67 | 1.84 | 0.003% |
| FFN gate-up | 32768 | 11008 | 4096 | 206.0 | **14.35** | 7.56 | 1.84 | 0.005% |
| FFN down | 32768 | 4096 | 11008 | 200.7 | **14.72** | 7.66 | 1.86 | 0.022% |

### 方阵扩展性测试

| Size | FP8 TC (ms) | FP8 TC TFLOPS | Mixed Tiled TFLOPS | BF16 Core TFLOPS | AvgRelErr |
|------|------------|---------------|-------------------|-----------------|-----------|
| 4096³ | 10.5 | **13.15** | 7.33 | 1.81 | 0.005% |
| 8192³ | 76.0 | **14.48** | 7.66 | 1.84 | 0.011% |
| 16384³ | 565.3 | **15.56** | 7.79 | 1.85 | 0.039% |
| 32768³ | 4521.0 | **15.56** | 7.78 | (skipped) | 0.153% |

### 结果分析

**性能:**
1. **FP8 TensorCore** 在 LLM 实际 shape 稳定达到 **13-15 TFLOPS**，是最优方案
2. LLaMA-70B FFN-up (M=2048, N=28672, K=8192) 达到 **14.87 TFLOPS**，大 N 有利于 128×128 tile
3. **FP8 TC vs BF16 CUDA Core**: 7.5x~8.5x 加速
4. **FP8 TC vs Mixed Tiled**: 稳定 1.8x~2.0x 加速
5. 方阵性能在 16K+ 饱和于 **~15.6 TFLOPS**

### 大 M 小 N/K 场景 (长序列/大批量, 小隐藏维度)

| Shape | M | N | K | FP8 TC (ms) | FP8 TC TFLOPS | Mixed Tiled TFLOPS | BF16 Core TFLOPS | AvgRelErr |
|-------|---|---|---|------------|---------------|-------------------|-----------------|-----------|
| M=64K, N=1K, K=1K | 65536 | 1024 | 1024 | 11.6 | **11.83** | 6.79 | 1.72 | 0.001% |
| M=64K, N=2K, K=1K | 65536 | 2048 | 1024 | 21.9 | **12.54** | 6.65 | 1.78 | 0.000% |
| M=64K, N=1K, K=2K | 65536 | 1024 | 2048 | 20.4 | **13.49** | 7.41 | 1.81 | 0.001% |
| M=64K, N=2K, K=2K | 65536 | 2048 | 2048 | 45.8 | **11.99** | 7.54 | 1.80 | 0.002% |
| M=128K, N=1K, K=1K | 131072 | 1024 | 1024 | 22.6 | **12.16** | 6.67 | 1.78 | 0.001% |
| M=128K, N=2K, K=1K | 131072 | 2048 | 1024 | 43.1 | **12.76** | 6.95 | 1.79 | 0.001% |
| M=128K, N=1K, K=2K | 131072 | 1024 | 2048 | 40.0 | **13.74** | 7.59 | 1.82 | 0.001% |
| M=128K, N=2K, K=2K | 131072 | 2048 | 2048 | 79.5 | **13.83** | 7.59 | 1.82 | 0.001% |

**大 M 场景分析:**
- FP8 TensorCore 在大 M (64K~128K) + 小 N/K (1K~2K) 下仍达到 **11.8-13.8 TFLOPS**
- 性能受 N/K 小导致 block tile (128×128) 在 N 方向 grid 过少影响（N=1024 时 grid_n=8）
- K=2048 比 K=1024 性能更好，因为 K-loop 迭代更多可更好隐藏 memory latency
- 精度极高: AvgRelErr ≤ 0.002%，因 K 较小时 fp16 舍入累积少

**精度 (相对 FP32→MXFP8 gold standard):**
1. **BF16 CUDA Core**: 与 gold standard **完全一致** (AvgRelErr = 0)
2. **WMMA 变体**: LLM shape 中 AvgRelErr 通常 < 0.02%
3. 大 K 场景 (K=28672) 误差稍大 (0.14%)，因 fp16 舍入沿 K 累积
4. 所有测试中 AvgRelErr < 0.16%，对模型推理精度无实质影响

**结论: FP8 TensorCore 在所有 LLM 典型 shape 中均为最优选择——14+ TFLOPS 性能，< 0.02% 精度损失（大多数场景）。**

## MXFP8 格式说明 (OCP Microscaling Spec)

```
MXFP8 = FP8 E4M3 数据 + E8M0 Block Scaling (OCP Microscaling Standard)

Block Size: 32 elements (一维, 沿连续维度)
  - 输入 A [M,K]: block 沿 K 方向, scales shape = [M, ceil(K/32)]
  - 输出 D [M,N]: block 沿 N 方向, scales shape = [M, ceil(N/32)]

Data Format: FP8 E4M3 (1-sign, 4-exponent, 3-mantissa, max=448.0)
Scale Format: E8M0 (8-bit exponent only, bias=127, value = 2^(byte-127))
  - 范围: 2^-127 ~ 2^128, 仅为 2 的幂次

量化流程 (per block of 32 elements):
  1. amax = max(|vals[0:31]|)
  2. scale = round_up_to_power_of_2(amax / 448.0)  → 存为 E8M0
  3. output[i] = round_to_nearest_even(vals[i] / scale) → 存为 FP8 E4M3

反量化:
  output[i] = fp8_to_float(data[i]) * e8m0_to_float(scale)

约束:
  - 张量最后一维必须是 32 的倍数 (不足补零)
  - Block 为一维 (不做 2D scaling)
  - 硬件原生支持: NVIDIA Blackwell (SM100+)
  - 本项目在 H100 (SM90) 上通过软件实现
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
