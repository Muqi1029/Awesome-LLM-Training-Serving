#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAdd(float *A, float *B, float *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  // 2. 设置参数
  int N = 65536; // 向量总长度 (256 * 256)
  size_t size = N * sizeof(float);

  // 3. 分配主机内存 (Host Memory - CPU)
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // 初始化主机数据
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // 4. 分配设备内存 (Device Memory - GPU)
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // 5. 将数据从主机拷贝到设备 (H2D)
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // 6. 设置执行配置 (Grid and Block)
  // 根据你的要求：grid 大小为 256
  // 我们设定每个 block 含有 256 个线程，总线程数 = 256 * 256 = 65536
  dim3 grid(256);
  dim3 block(256);

  // 7. 启动内核
  // 这里使用了最简形式 <<<grid, block>>>
  vecAdd<<<grid, block>>>(d_A, d_B, d_C, N);

  // 等待 GPU 完成工作 (可选，cudaMemcpy 内部带同步效果)
  cudaDeviceSynchronize();

  // 8. 将结果从设备拷贝回主机 (D2H)
  cudaMemcpy(h_C, h_C, size,
             cudaMemcpyDeviceToHost); // 注意：这里通常拷贝回 h_C
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // 9. 验证结果
  bool success = true;
  for (int i = 0; i < N; i++) {
    if (h_C[i] != 3.0f) {
      success = false;
      break;
    }
  }

  if (success) {
    printf("Success! 结果正确 (1.0 + 2.0 = 3.0)\n");
  } else {
    printf("Error! 结果错误\n");
  }

  // 10. 释放内存
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
