#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Building Mixed Precision GEMM Kernels ==="
echo "Project: $PROJECT_DIR"
echo "Build:   $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_DIR" \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build . -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Binaries in: $BUILD_DIR"
echo "  - test_mxfp8_quantize"
echo "  - test_gemm_correctness"
echo "  - test_gemm_perf"
