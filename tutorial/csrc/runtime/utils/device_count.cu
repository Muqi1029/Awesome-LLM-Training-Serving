#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  printf("device_count: %d\n", device_count);
}
