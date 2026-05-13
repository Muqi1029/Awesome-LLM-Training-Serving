#include <cuda_runtime.h>
#include <stdio.h>

inline int getMultiProcessorCount() {
  int nSM;
  int deviceID;
  cudaGetDevice(&deviceID);
  // cudaSetDevice()
  cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID);
  printf("deviceID: %d\n", deviceID);
  return nSM;
}

int main() {
  printf("nSM: %d\n", getMultiProcessorCount());
  return 0;
}
