#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

typedef float V;

constexpr int expand_times = 64;
constexpr int DIM = 64;
struct Vector {
  V values[DIM];
};

__global__ void d2h_hbm_data(Vector* __restrict dst, Vector* __restrict src,
                             int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    dst[vec_index * expand_times].values[dim_index] =
        src[vec_index].values[dim_index];
  }
}

int main() {
  constexpr int V_NUM = 1024 * 1024;
  constexpr int N = V_NUM * DIM;
  constexpr int TEST_TIMES = 1;

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  Vector* src;
  Vector* dst;
  cudaMalloc(&src, V_NUM * sizeof(Vector));
  cudaMallocHost(&dst, expand_times * V_NUM * sizeof(Vector),
                 cudaHostAllocWriteCombined);

  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;

  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data d2h speed = %.2f GB/s\n",
         V_NUM * DIM * sizeof(float) / (diff_test.count()) / (1 << 30));

  cudaFreeHost(dst);
  cudaFree(src);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
