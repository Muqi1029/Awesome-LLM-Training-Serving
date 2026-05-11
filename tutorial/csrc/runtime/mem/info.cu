#include <cstdio>
#include <cuda_runtime_api.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    if (cudaSetDevice(i) != cudaSuccess) {
      printf("Error: Cannot set device %d\n", i);
      return 1;
    }
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("GPU[%d] total memory %0.2f GB (%0.2f GiB), available "
           "memory %0.2f GB (%0.2f GiB)\n",
           i, (static_cast<double>(total) / 1e9),
           (static_cast<double>(total) / (1 << 30)),
           (static_cast<double>(free) / 1e9),
           (static_cast<double>(free) / (1 << 30)));
  }
  return 0;
}
