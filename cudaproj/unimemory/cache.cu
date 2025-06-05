#include <iostream>

__global__ void vectorSum(const float *a, float *sum, int n) {
    __shared__ float partialSum[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    partialSum[tid] = (idx < n) ? a[idx] : 0.0;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partialSum[tid] += partialSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, partialSum[0]);
    }
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    // 主机端数据
    float *h_a, h_sum = 0.0;
    h_a = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
    }

    // 设备端数据
    float *d_a, *d_sum;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // 定义网格和线程块
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 调用内核函数
    vectorSum<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_sum, n);

    // 将结果从设备复制到主机
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    float expected_sum = 0.0;
    for (int i = 0; i < n; i++) {
        expected_sum += h_a[i];
    }
    if (h_sum != expected_sum) {
        std::cout << "Error: Expected sum = " << expected_sum << ", Got sum = " << h_sum << std::endl;
    }

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_sum);
    free(h_a);

    return 0;
}