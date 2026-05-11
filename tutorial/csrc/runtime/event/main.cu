#include <cuda_runtime.h>
#include <iostream>

void time_kernel() {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  // kernel launch
  //
  cudaEventRecord(end);

  // let cpu sync to wait all gpu kernels to be completed
  cudaEventSynchronize(end);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, end);
  std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

int main() {
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, 0);
  cudaEventSynchronize(event);
  cudaEventDestroy(event);

  time_kernel();
  return 0;
}
