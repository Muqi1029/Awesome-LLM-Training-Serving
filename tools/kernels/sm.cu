#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    printf("GPU %d: %s\n", i, devProp.name);
    printf("  Total Streaming Multiprocessors (SMs): %d\n",
           devProp.multiProcessorCount);
  }
  return 0;
}
