#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  void *ptr = nullptr;
  cudaMallocAsync(&ptr, 0, 0);
  bool result = (ptr == nullptr);
  printf("ptr == nullptr: %d\n", result);
  if (result) {
      printf("ptr: %p\n", ptr);
  }
}
