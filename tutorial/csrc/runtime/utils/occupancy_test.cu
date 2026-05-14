#include <cuda_runtime.h>
#include <iostream>

__global__ void squareKernel(float *d_out, float *d_in, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    d_out[i] = d_in[i] * d_in[i];
  }
}

__global__ void squareKernelBack(float *d_out, float *d_in, int N) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; ++i) {
    d_out[i] = d_in[i] * d_in[i];
  }
}

int main() {
  int N = 1000000000;
  int blockSize = 256;

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId);
  int smCount = prop.multiProcessorCount;

  int maxBlocksPerSM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, squareKernel,
                                                blockSize, 0);
  int numBlocks = maxBlocksPerSM * smCount;

  std::cout << "--- GPU Occupancy Info ---" << std::endl;
  std::cout << "GPU Name: " << prop.name << std::endl;
  std::cout << "Device Index: " << deviceId << std::endl;
  std::cout << "SM Count: " << smCount << std::endl;
  std::cout << "Max Active Blocks per SM: " << maxBlocksPerSM << std::endl;
  std::cout << "Optimal Grid Size (Total Blocks): " << numBlocks << std::endl;

  //
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // prepare input * output
  float *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  squareKernel<<<numBlocks, blockSize>>>(d_out, d_in, N);

  cudaEventRecord(start);
  squareKernel<<<numBlocks, blockSize>>>(d_out, d_in, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaDeviceSynchronize();

  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Kernel executed successfully with optimized parameters!"
            << std::endl;

  return 0;
}
