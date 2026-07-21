#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(dirname "$SCRIPT_DIR")/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Run build.sh first."
    exit 1
fi

echo "=========================================="
echo "  Mixed Precision GEMM Test Suite"
echo "=========================================="

echo ""
echo "[1/3] MXFP8 Quantization Tests"
echo "------------------------------------------"
"$BUILD_DIR/test_mxfp8_quantize"

echo ""
echo "[2/3] GEMM Correctness Tests"
echo "------------------------------------------"
"$BUILD_DIR/test_gemm_correctness"

echo ""
echo "[3/3] GEMM Performance Benchmark"
echo "------------------------------------------"
"$BUILD_DIR/test_gemm_perf"

echo ""
echo "=========================================="
echo "  All Tests Complete"
echo "=========================================="
