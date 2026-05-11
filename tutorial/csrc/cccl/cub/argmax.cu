#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <typename T> void find_argmax(T *h_in, int num_items) {
  using KVPair = cub::KeyValuePair<int, T>;

  // prepare input
  T *d_in;
  cudaMalloc(&d_in, sizeof(T) * num_items);
  cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice);

  // prepare output
  KVPair *d_out;
  cudaMalloc(&d_out, sizeof(KVPair));

  // get temp storage by a fake run
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes;
  cub::DeviceReduce::ArgMax(NULL, temp_storage_bytes, d_in, d_out, num_items);

  // formal run
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items);

  // copy final result d2h
  KVPair h_out;
  cudaMemcpy(&h_out, d_out, sizeof(KVPair), cudaMemcpyDeviceToHost);

  // print
  printf("The maximum value: %d, index: %d\n", h_out.value, h_out.key);

  // clean up resources
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_temp_storage);
}

int main() {
  // define h_in
  int input[] = {12, 32, 1001, 21, 301};
  int num_items = sizeof(input) / sizeof(int);

  find_argmax<int>(input, num_items);

  return 0;
}
