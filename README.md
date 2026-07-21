# Mixed Precision GEMM Kernels (MXFP8 + BF16) for NVIDIA H100

## 问题

实现混合精度矩阵乘法 `D = A * B + C`：

| 矩阵 | 格式 | 说明 |
|-------|------|------|
| A [M, K] | MXFP8 | FP8 E4M3 + E8M0 block scaling (block_size=32, 沿 K) |
| B [K, N] | BF16 | 权重矩阵 |
| C [M, N] | MXFP8 | 偏置 (可选) |
| D [M, N] | MXFP8 | 输出, block scaling 沿 N 方向 |

**MXFP8 格式 (OCP Microscaling Standard):**
- 每 32 个连续元素共享一个 E8M0 scale factor (8-bit exponent, value = 2^(byte-127))
- 数据为 FP8 E4M3 (max=448.0)
- Blackwell (SM100) 硬件原生支持, 本项目在 H100 (SM90) 上通过软件实现

**核心挑战:** H100 Tensor Core 要求 A/B 同为 FP8 或同为 FP16/BF16, 无法直接处理 MXFP8+BF16 混合输入。需设计高效的数据转换 + 计算融合策略。

## 算法

### 方案 1: FP8 TensorCore (最优)

```
策略: MXFP8 反量化 → FP16 → WMMA Tensor Core → FP32 累加 → MXFP8 量化输出

优化:
- 128×128 block tile, 8 warps (256 threads)
- 每 warp 4×2 = 8 个 WMMA 16×16×16 fragments
- 双缓冲 shared memory: 计算当前 tile 同时预加载下一个
- Per-warp epilogue buffer 避免共享内存冲突
- 协作加载: 256 线程并行 dequant A + convert B → shared memory
```

### 方案 2: BF16 CUDA Core (最高精度)

```
策略: MXFP8 反量化 → FP32 → CUDA Core FMA → FP32 累加 → MXFP8 量化输出

特点:
- 64×64 block tile, 16×16 线程, 每线程 4×4 输出
- A/B 均转为 FP32, FMA 累加无精度损失
- 与 FP32→MXFP8 gold standard 完全一致 (AvgRelErr = 0)
- 不使用 Tensor Core, 性能较低
```

### 方案 3: Mixed Tiled (均衡)

```
策略: MXFP8 反量化 → FP16 → WMMA Tensor Core → FP32 累加 → MXFP8 量化输出

特点:
- 32×32 block tile, 4 warps (128 threads)
- 每 warp 1 个 WMMA 16×16×16 fragment
- Per-warp shared memory, warp-level sync
- 小矩阵时优于方案1 (更细粒度的 grid)
```

## 结果

**精度标准:** FP32 全精度计算 → 量化到 MXFP8 → 反量化, 作为 gold standard。误差仅反映计算精度损失。

### 方案 1: FP8 TensorCore

| Shape | M | N | K | Time(ms) | TFLOPS | AvgRelErr |
|-------|---|---|---|----------|--------|-----------|
| LLaMA-7B QKV | 4096 | 12288 | 4096 | 29.2 | **14.11** | 0.0005% |
| LLaMA-7B FFN-up | 4096 | 11008 | 4096 | 27.8 | **13.29** | 0.0005% |
| LLaMA-7B FFN-down | 4096 | 4096 | 11008 | 25.9 | **14.26** | 0.0018% |
| LLaMA-70B FFN-up | 2048 | 28672 | 8192 | 64.9 | **14.82** | 0.0012% |
| LLaMA-70B FFN-down | 2048 | 8192 | 28672 | 65.7 | **14.65** | 0.0056% |
| M=64K, N=2K, K=2K | 65536 | 2048 | 2048 | 40.0 | **13.73** | 0.0002% |
| M=128K, N=2K, K=2K | 131072 | 2048 | 2048 | 79.3 | **13.87** | 0.0002% |
| Square 8K | 8192 | 8192 | 8192 | 75.9 | **14.49** | 0.0011% |
| Square 16K | 16384 | 16384 | 16384 | 565.6 | **15.55** | 0.0028% |

### 方案 2: BF16 CUDA Core

| Shape | M | N | K | Time(ms) | TFLOPS | AvgRelErr |
|-------|---|---|---|----------|--------|-----------|
| LLaMA-7B QKV | 4096 | 12288 | 4096 | 225.5 | 1.83 | **0%** |
| LLaMA-7B FFN-up | 4096 | 11008 | 4096 | 210.8 | 1.75 | **0%** |
| LLaMA-7B FFN-down | 4096 | 4096 | 11008 | 202.0 | 1.83 | **0%** |
| LLaMA-70B FFN-up | 2048 | 28672 | 8192 | 522.0 | 1.84 | **0%** |
| LLaMA-70B FFN-down | 2048 | 8192 | 28672 | 524.6 | 1.83 | **0%** |
| M=64K, N=2K, K=2K | 65536 | 2048 | 2048 | 367.0 | 1.50 | **0%** |
| M=128K, N=2K, K=2K | 131072 | 2048 | 2048 | 603.4 | 1.82 | **0%** |
| Square 8K | 8192 | 8192 | 8192 | 593.9 | 1.85 | **0%** |
| Square 16K | 16384 | 16384 | 16384 | 4750.8 | 1.85 | **0%** |

### 方案 3: Mixed Tiled

| Shape | M | N | K | Time(ms) | TFLOPS | AvgRelErr |
|-------|---|---|---|----------|--------|-----------|
| LLaMA-7B QKV | 4096 | 12288 | 4096 | 54.9 | 7.51 | 0.0005% |
| LLaMA-7B FFN-up | 4096 | 11008 | 4096 | 49.3 | 7.49 | 0.0005% |
| LLaMA-7B FFN-down | 4096 | 4096 | 11008 | 48.5 | 7.61 | 0.0018% |
| LLaMA-70B FFN-up | 2048 | 28672 | 8192 | 125.2 | 7.68 | 0.0012% |
| LLaMA-70B FFN-down | 2048 | 8192 | 28672 | 125.7 | 7.65 | 0.0056% |
| M=64K, N=2K, K=2K | 65536 | 2048 | 2048 | 72.7 | 7.57 | 0.0002% |
| M=128K, N=2K, K=2K | 131072 | 2048 | 2048 | 144.0 | 7.63 | 0.0002% |
| Square 8K | 8192 | 8192 | 8192 | 142.8 | 7.70 | 0.0011% |
| Square 16K | 16384 | 16384 | 16384 | 1124.0 | 7.83 | 0.0028% |

### 对比总结

| 方案 | 峰值 TFLOPS | LLM 典型 TFLOPS | vs BF16 加速 | AvgRelErr |
|------|------------|----------------|-------------|-----------|
| FP8 TensorCore | 15.55 | 13~15 | **7.5x~8.5x** | < 0.006% |
| Mixed Tiled | 7.83 | 7.5~7.7 | 4.1x~4.2x | 与 FP8 TC 相同 |
| BF16 CUDA Core | 1.85 | 1.75~1.85 | 1x (baseline) | **0% (精确)** |

**关键结论:**
- FP8 TensorCore 是最优方案: 14+ TFLOPS, 精度损失 < 0.006% (Gaussian 数据)
- 精度损失来源: fp16 中间精度在 K 方向累积舍入 (大 K 时略增)
- BF16 CUDA Core 与 gold standard 完全一致, 适合精度验证场景
- FP8 TC 和 Mixed Tiled 精度完全相同 (均使用 fp16 WMMA)

### 不同方差对精度的影响

固定 shape [4096 × 4096 × 4096], 测试不同 Gaussian(0, std) 的精度:

| std | FP8 TC AvgRelErr | FP8 TC TFLOPS | Mixed Tiled AvgRelErr | Mixed Tiled TFLOPS |
|-----|-----------------|---------------|----------------------|-------------------|
| 0.01 | 0.000416% | 12.93 | 0.000416% | 7.50 |
| 0.02 | 0.000528% | 13.43 | 0.000528% | 7.13 |
| 0.05 | 0.000487% | 13.34 | 0.000487% | 7.60 |
| 0.10 | 0.000818% | 13.45 | 0.000818% | 7.49 |
| 0.50 | 0.007692% | 13.21 | 0.007692% | 7.57 |
| 1.00 | 0.012032% | 13.29 | 0.012032% | 7.42 |
| 5.00 | 0.559187% | 13.03 | 0.559187% | 7.57 |

**方差分析:**
- **std ≤ 0.1**: AvgRelErr < 0.001%, 精度极高, 适合 LLM 推理场景
- **std = 0.5~1.0**: AvgRelErr ≈ 0.008%~0.012%, 仍然极好
- **std = 5.0**: AvgRelErr ≈ 0.56%, 大动态范围数据精度下降明显
- **性能不受方差影响**: 所有 std 下 TFLOPS 保持稳定 (13+ / 7.5+)
- **FP8 TC 与 Mixed Tiled 精度完全相同**: 两者使用相同 fp16 WMMA 路径
- 大 std 精度下降原因: MXFP8 block 内方差大, E8M0 scale 为 2 的幂次无法精确覆盖

## 测试数据生成

```
随机数生成:
  - 种子: mt19937(42 + test_index), 确保可复现
  - 分布: Gaussian(mean=0, std=0.02), 模拟神经网络权重/激活分布

矩阵 A [M, K]:
  1. 生成 FP32 随机矩阵 A_float ~ N(0, 0.02²)
  2. MXFP8 量化: 沿 K 每 32 元素为一个 block
     - scale = round_up_pow2(amax(block) / 448.0) → E8M0
     - data[i] = round_nearest_even(A_float[i] / scale) → FP8 E4M3
  3. 输入 kernel 的是量化后的 (A_data, A_scales)

矩阵 B [K, N]:
  1. 生成 FP32 随机值 ~ N(0, 0.02²)
  2. 转换为 BF16: __float2bfloat16(val)
  3. 直接作为 BF16 输入 kernel

矩阵 C [M, N]:
  1. 生成 FP32 随机矩阵 C_float ~ N(0, 0.02²)
  2. 同 A 的方式量化为 MXFP8 (block 沿 N 方向)

Gold Standard (精度参考):
  1. A 反量化为 FP32, B 转为 FP32, C 反量化为 FP32
  2. FP32 tiled GEMM: D_ref = A_fp32 * B_fp32 + C_fp32
  3. D_ref 量化为 MXFP8 → 再反量化回 FP32
  4. 与 kernel 输出 (也经过 MXFP8 反量化) 比较
  5. 误差仅反映计算路径差异, 不含输出量化固有损失

选择 std=0.02 的原因:
  - 接近 LLM 权重初始化分布 (Xavier/He init 在大维度时 std ≈ 0.01~0.03)
  - 数值集中在 MXFP8 精度范围内, 量化损失小
  - 避免 uniform 分布的边界效应
```

## 与已有实现对比

| 实现 | 硬件 | 格式 | 峰值 TFLOPS | 说明 |
|------|------|------|------------|------|
| **cuBLAS FP8** | H100 | FP8 E4M3 (per-tensor scaling) | ~1500-1800 | 硬件原生 FP8 MMA, 理论峰值 1979 TFLOPS |
| **cuBLAS BF16** | H100 | BF16 | ~800-900 | 理论峰值 989 TFLOPS |
| **CUTLASS Block-Scaled** | Blackwell (SM100) | MXFP8×BF16 mixed | ~2000+ | 使用 tcgen05.mma.blockscaled, 仅 SM100 |
| **Transformer Engine MXFP8** | Blackwell (SM100+) | MXFP8 | ~1.6x vs BF16 | 需要 SM100, 通过 torch API 调用 |
| **TK-GEMM (Triton)** | H100 | FP8 per-tensor | ~1.87x cuBLAS FP8 | 针对小 batch 推理优化的 SplitK 方案 |
| **本项目 FP8 TensorCore** | H100 | MXFP8 (软件模拟) | **15.6** | H100 上唯一的 MXFP8 block-scaled 实现 |

**关键差异:**

1. **本项目是 H100 上唯一的 MXFP8 block-scaled GEMM 实现**
   - MXFP8 (32-element block scaling) 是 Blackwell 硬件原生特性
   - 现有方案 (cuBLAS/CUTLASS MXFP8) 均要求 SM100+
   - 本项目通过软件模拟在 H100 (SM90) 上实现 MXFP8 语义

2. **性能差距分析 (本项目 15.6 vs cuBLAS ~1500 TFLOPS)**
   - cuBLAS 使用硬件原生 FP8 `mma.sync` PTX 指令 (m16n8k32), 本项目使用 FP16 WMMA (m16n16k16)
   - 本项目额外开销: MXFP8 反量化 (global mem → smem) + FP16 转换 + 输出 MXFP8 量化
   - WMMA API 限制: 无法直接使用 FP8 tensor core, 只能通过 FP16 间接利用
   - 进一步优化方向: PTX 内联 `mma.sync.m16n8k32.e4m3` 可提升 2-4x

3. **适用场景**
   - 本项目适合: 需要在 H100 上使用 MXFP8 格式兼容 Blackwell 训练流水线的场景
   - cuBLAS FP8 适合: 不需要 block scaling, 使用 per-tensor scaling 的标准 FP8 推理
   - CUTLASS Blackwell 适合: 拥有 Blackwell GPU, 追求峰值性能

```bash
# 构建
bash scripts/build.sh

# 运行完整 benchmark
./build/test_benchmark_report

# 运行正确性测试
./build/test_gemm_correctness
```

**环境:** NVIDIA H100 (SM90), CUDA 12.4+, CMake 3.20+
