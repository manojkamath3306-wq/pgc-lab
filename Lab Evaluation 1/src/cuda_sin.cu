#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void compute_heavy_kernel(const float* X, const float* Y, float* Z, float alpha, size_t N, int K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = X[idx];
        float y = Y[idx];
        float sum = 0.0f;

        #pragma unroll 4
        for (int k = 1; k <= K; ++k) {
            sum += (sinf(k * x) + cosf(k * y)) / (float)k;
        }

        Z[idx] = sum + alpha * x;
    }
}

int main(int argc, char* argv[]) {
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 10000000;
    int K = (argc > 2) ? std::stoi(argv[2]) : 50; // 50 iterations per element
    int blockSize = 256;
    float alpha = 2.5f;
    size_t bytes = N * sizeof(float);

    std::vector<float> h_X(N, 1.25f), h_Y(N, 0.75f), h_Z(N, 0.0f);

    float *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);
    cudaMalloc(&d_Z, bytes);

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // Total Pipeline Start (includes PCIe Transfers)
    cudaEventRecord(start_total);
    cudaMemcpy(d_X, h_X.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y.data(), bytes, cudaMemcpyHostToDevice);

    size_t gridSize = (N + blockSize - 1) / blockSize;

    // Kernel Compute
    cudaEventRecord(start_kernel);
    compute_heavy_kernel<<<gridSize, blockSize>>>(d_X, d_Y, d_Z, alpha, N, K);
    cudaEventRecord(stop_kernel);

    cudaMemcpy(h_Z.data(), d_Z, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_total);

    cudaEventSynchronize(stop_total);

    float total_ms = 0.0f, kernel_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start_total, stop_total);
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);

    std::cout << "CUDA_Heavy,N=" << N << ",K=" << K 
              << ",Kernel_ms=" << kernel_ms 
              << ",Total_ms=" << total_ms << std::endl;

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    return 0;
}
