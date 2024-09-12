#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

#include <chrono>

void vecAdd(float* A, float* B, float* C, int N) {
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy host memory to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Kernel invocation with N blocks and 256 threads per block
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize; // This ensures enough blocks

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 1 << 20; // 2^20 elements
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Call vector addition function
    vecAdd(A, B, C, N);

    // Verify the result
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
