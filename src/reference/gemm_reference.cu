#include "gemm_reference.h"
#include "gemm_common.h"

// Simple GPU GEMM kernel: D = A * B + C
// A [M,K] row-major, B [K,N] row-major, C [M,N], D [M,N]
__global__ void gemm_reference_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       const float* __restrict__ C,
                                       float* __restrict__ D,
                                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    if (C != nullptr) {
        sum += C[row * N + col];
    }
    D[row * N + col] = sum;
}

// Tiled version for better performance on larger matrices
__global__ void gemm_reference_tiled_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             const float* __restrict__ C,
                                             float* __restrict__ D,
                                             int M, int N, int K) {
    const int TILE = 32;
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (C != nullptr) sum += C[row * N + col];
        D[row * N + col] = sum;
    }
}

void gemm_reference_gpu(const float* d_A, const float* d_B, const float* d_C,
                         float* d_D, int M, int N, int K, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_reference_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, M, N, K);
}

void gemm_reference_host(const float* A, const float* B, const float* C,
                          float* D, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            if (C != nullptr) sum += C[i * N + j];
            D[i * N + j] = sum;
        }
    }
}
