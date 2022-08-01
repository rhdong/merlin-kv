#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>

typedef float V;

constexpr int DIM = 64;
struct Vector {
  V values[DIM];
};

void create_random_offset(int *offset, int num, int range) {
  std::unordered_set<int> numbers;
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

__global__ void d2h_const_data(const Vector *__restrict src,
                               Vector **__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = 0.1f;
  }
}

__global__ void d2h_hbm_data(
    Vector *__restrict src, Vector **__restrict dst,
    int N) {  // dst is a set of Vector* in the pinned memory
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    //     (*(dst[vec_index])).values[dim_index] =
    //     src[vec_index].values[dim_index];

    src[vec_index].values[dim_index] = (*(dst[vec_index])).values[dim_index];
  }
}

__global__ void create_fake_ptr(const Vector *__restrict dst,
                                Vector **__restrict vectors, int *offset,
                                int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    vectors[tid] = (Vector *)((Vector *)dst + offset[tid]);
  }
}

template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void upsert_kernel(const K *__restrict keys, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int rank = g.thread_rank();

    int key_pos = -1;
    bool found_or_empty = false;
    const size_t bucket_max_size = 128;
    size_t key_idx = t / TILE_SIZE;
    K insert_key = keys[key_idx];
    K hashed_key = insert_key;  // Murmur3HashDevice(insert_key);
    size_t bkt_idx = hashed_key & (524288 - 1);
    size_t start_idx = hashed_key & (bucket_max_size - 1);

    Bucket<K, V, M, DIM> *bucket = table->buckets + bkt_idx;

#pragma unroll
    for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      size_t key_offset = (start_idx + tile_offset + rank) % bucket_max_size;
      K current_key = *(bucket->keys + key_offset);
      auto const found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      if (found_or_empty_vote) {
        found_or_empty = true;
        key_pos = (start_idx + tile_offset + __ffs(found_or_empty_vote) - 1) &
                  bucket_max_size;
        break;
      }
    }
  }
}
int main() {
  constexpr int KEY_NUM = 1024 * 1024;
  constexpr int INIT_SIZE = KEY_NUM * 32;
  constexpr int N = KEY_NUM * DIM;
  constexpr int TEST_TIMES = 1;
  constexpr size_t vectors_size = INIT_SIZE * sizeof(Vector);

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  int *h_offset;
  int *d_offset;

  cudaMallocHost(&h_offset, sizeof(int) * KEY_NUM);
  cudaMalloc(&d_offset, sizeof(int) * KEY_NUM);
  cudaMemset(&h_offset, 0, sizeof(int) * KEY_NUM);
  cudaMemset(&d_offset, 0, sizeof(int) * KEY_NUM);

  Vector *src;
  Vector *dst;
  Vector **dst_ptr;
  cudaMalloc(&src, KEY_NUM * sizeof(Vector));
  cudaMalloc(&dst_ptr, KEY_NUM * sizeof(Vector *));
  cudaMallocHost(&dst, vectors_size);

  create_random_offset(h_offset, KEY_NUM, INIT_SIZE);
  cudaMemcpy(d_offset, h_offset, sizeof(int) * KEY_NUM, cudaMemcpyHostToDevice);
  create_fake_ptr<<<1024, 1024>>>(dst, dst_ptr, d_offset, KEY_NUM);
  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;

  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_const_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] Constant d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  cudaFreeHost(dst);
  cudaFreeHost(h_offset);
  cudaFree(dst_ptr);
  cudaFree(src);
  cudaFree(d_offset);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
