#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <typename T> void find_argmax_modern(T *h_in, int num_items) {
  T *d_in;
  cudaMalloc(&d_in, sizeof(T) * num_items);
  cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice);

  int *d_out_index;
  T *d_out_value;
  cudaMalloc(&d_out_index, sizeof(int));
  cudaMalloc(&d_out_value, sizeof(T));

  // fake run: to get the size of temp_storage_bytes
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes;
  cub::DeviceReduce::ArgMax(NULL, temp_storage_bytes, d_in, d_out_value,
                            d_out_index, num_items);

  printf("Need temp_storage_bytes: %zu\n", temp_storage_bytes);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in,
                            d_out_value, d_out_index, num_items);

  int h_index;
  T h_value;
  cudaMemcpy(&h_index, d_out_index, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_value, d_out_value, sizeof(T), cudaMemcpyDeviceToHost);

  printf("The maximum value: %d, index: %d\n", (int)h_value, h_index);

  cudaFree(d_in);
  cudaFree(d_out_index);
  cudaFree(d_out_value);
  cudaFree(d_temp_storage);
}

int main() {
  int input[] = {12, 32, 1001, 21, 301};
  int num_items = sizeof(input) / sizeof(int);
  find_argmax_modern<int>(input, num_items);
  return 0;
}
