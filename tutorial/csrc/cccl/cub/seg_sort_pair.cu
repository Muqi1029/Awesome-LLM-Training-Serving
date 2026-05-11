#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename KeyT, typename ValueT>
void segmented_sort_pairs_demo(KeyT *h_keys, ValueT *h_values, int num_items,
                               int *h_offsets, int num_segments) {
  // device input
  KeyT *d_keys_in;
  ValueT *d_values_in;

  cudaMalloc(&d_keys_in, sizeof(KeyT) * num_items);
  cudaMalloc(&d_values_in, sizeof(ValueT) * num_items);

  cudaMemcpy(d_keys_in, h_keys, sizeof(KeyT) * num_items,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_values_in, h_values, sizeof(ValueT) * num_items,
             cudaMemcpyHostToDevice);

  // device output
  KeyT *d_keys_out;
  ValueT *d_values_out;

  cudaMalloc(&d_keys_out, sizeof(KeyT) * num_items);
  cudaMalloc(&d_values_out, sizeof(ValueT) * num_items);

  // segment offsets
  int *d_offsets;

  cudaMalloc(&d_offsets, sizeof(int) * (num_segments + 1));

  cudaMemcpy(d_offsets, h_offsets, sizeof(int) * (num_segments + 1),
             cudaMemcpyHostToDevice);

  // temp storage query
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceSegmentedSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  printf("Need temp_storage_bytes: %zu\n", temp_storage_bytes);

  // allocate temp storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // real run
  cub::DeviceSegmentedSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  // copy back
  KeyT *h_keys_out = new KeyT[num_items];
  ValueT *h_values_out = new ValueT[num_items];

  cudaMemcpy(h_keys_out, d_keys_out, sizeof(KeyT) * num_items,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_values_out, d_values_out, sizeof(ValueT) * num_items,
             cudaMemcpyDeviceToHost);

  // print result
  printf("\nSorted result:\n");

  for (int i = 0; i < num_items; i++) {
    printf("(%d, %c) ", h_keys_out[i], h_values_out[i]);
  }

  printf("\n");

  // free
  delete[] h_keys_out;
  delete[] h_values_out;

  cudaFree(d_keys_in);
  cudaFree(d_values_in);

  cudaFree(d_keys_out);
  cudaFree(d_values_out);

  cudaFree(d_offsets);

  cudaFree(d_temp_storage);
}

int main() {
  int keys[] = {9, 3, 7, 5, 1, 8, 2};

  char values[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g'};

  // segments:
  //
  // [0,3)
  // [3,7)

  int offsets[] = {0, 3, 7};

  int num_items = sizeof(keys) / sizeof(int);

  int num_segments = 2;

  segmented_sort_pairs_demo<int, char>(keys, values, num_items, offsets,
                                       num_segments);

  return 0;
}
