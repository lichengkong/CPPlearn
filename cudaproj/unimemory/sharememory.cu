#include <iostream>

#define TILE_SIZE 32

__global__ void matrixMultiply(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (Row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[Row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (t * TILE_SIZE + ty < N && Col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + Col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    // 主机端数据
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // 设备端数据
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义网格和线程块
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // 调用内核函数
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // 将结果从设备复制到主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}