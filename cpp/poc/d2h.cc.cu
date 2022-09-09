#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <thread>
#include <unordered_set>

typedef float V;

constexpr int DIM = 64;
struct Vector {
  V values[DIM];
};

void create_random_offset(int* offset, int num, int range) {
  std::set<int> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<int> distr;
  int i = 0;

  while (numbers.size() < num) {
    numbers.insert(distr(eng) % range);
  }

  for (const int num : numbers) {
    offset[i++] = num;
  }
}

__global__ void d2h_const_data(const Vector* __restrict src,
                               Vector** __restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = 0.1f;
  }
}

__global__ void d2h_hbm_data(
    Vector* __restrict src, Vector** __restrict dst,
    int N) {  // dst is a set of Vector* in the pinned memory
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = src[vec_index].values[dim_index];

    //    src[vec_index].values[dim_index] =
    //    (*(dst[vec_index])).values[dim_index];
  }
}

void mthd_memcpy_one(Vector** __restrict dst, Vector* __restrict src,
                     int handled_size, int trunk_size) {
  for (int i = handled_size; i < handled_size + trunk_size; i++) {
    memcpy(dst[i], src + i, sizeof(Vector));
  }
}

void mthd_memcpy(Vector** __restrict dst, Vector* __restrict src, size_t N,
                 int n_worker = 8) {
  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;

  start_test = std::chrono::steady_clock::now();

  std::vector<std::thread> thds;
  if (n_worker < 1) std::cerr << "at least 1 worker for memcpy\n";

  size_t trunk_size = N / n_worker;
  size_t handled_size = 0;
  for (int i = 0; i < n_worker - 1; i++) {
    thds.push_back(
        std::thread(mthd_memcpy_one, dst, src, handled_size, trunk_size));
    handled_size += trunk_size;
  }

  size_t remaining = N - handled_size;
  thds.push_back(
      std::thread(mthd_memcpy_one, dst, src, handled_size, remaining));

  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] mthd_memcpy create threads %f.2ms\n",
         diff_test.count() * 1000);

  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < n_worker; i++) {
    thds[i].join();
  }
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] mthd_memcpy execute threads %f.2ms\n",
         diff_test.count() * 1000);
}

void d2h_hbm_data_cpu(Vector** __restrict dst, Vector* __restrict src, int N) {
  mthd_memcpy(dst, src, N);
}

__global__ void create_fake_ptr(const Vector* __restrict dst,
                                Vector** __restrict vectors, int* offset,
                                int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    vectors[tid] = (Vector*)((Vector*)dst + offset[tid]);
  }
}

int main() {
  constexpr int KEY_NUM = 1024 * 1024;
  constexpr int INIT_SIZE = KEY_NUM * 1024;
  constexpr int N = KEY_NUM * DIM;
  constexpr int TEST_TIMES = 1;
  constexpr size_t vectors_size = INIT_SIZE * sizeof(Vector);

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  int* h_offset;
  int* d_offset;

  cudaMallocHost(&h_offset, sizeof(int) * KEY_NUM);
  cudaMalloc(&d_offset, sizeof(int) * KEY_NUM);
  cudaMemset(&h_offset, 0, sizeof(int) * KEY_NUM);
  cudaMemset(&d_offset, 0, sizeof(int) * KEY_NUM);

  Vector* src;
  Vector* dst;
  Vector** dst_ptr;
  cudaMallocHost(&src, KEY_NUM * sizeof(Vector));
  cudaMallocHost(&dst_ptr, KEY_NUM * sizeof(Vector*));
  cudaMallocHost(&dst, vectors_size);

  create_random_offset(h_offset, KEY_NUM, INIT_SIZE);
  cudaMemcpy(d_offset, h_offset, sizeof(int) * KEY_NUM, cudaMemcpyHostToDevice);
  create_fake_ptr<<<1024, 1024>>>(dst, dst_ptr, d_offset, KEY_NUM);
  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;

  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    // d2h_const_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] Constant d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    // d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  printf("[timing] HBM data d2h=%.2fms\n",
         KEY_NUM * DIM * sizeof(float) / (diff_test.count()) / (1 << 30));

  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_hbm_data_cpu(dst_ptr, src, KEY_NUM);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data cpu h2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  printf("[timing] HBM data cpu h2h=%.2f GB/s\n",
         KEY_NUM * DIM * sizeof(float) / (diff_test.count()) / (1 << 30));

  cudaFreeHost(dst);
  cudaFreeHost(h_offset);
  cudaFreeHost(dst_ptr);
  cudaFreeHost(src);
  cudaFreeHost(d_offset);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
