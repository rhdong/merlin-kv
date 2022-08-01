#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>

typedef uint64_t K;

template <class K>
void create_continuous_keys(K *h_keys, int KEY_NUM, K start = 0) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
  }
}

template <class K>
struct Bucket {
  K *keys;
};

constexpr K EMPTY_KEY = std::numeric_limits<K>::max();
constexpr int KEY_NUM = 1024 * 1024;
constexpr int INIT_SIZE = KEY_NUM * 64;
constexpr int MAX_BUCKET_SIZE = 128;
constexpr const size_t BLOCK_SIZE = 128;
constexpr int TILE_SIZE = 8;
constexpr const size_t N = KEY_NUM * TILE_SIZE;
constexpr const size_t GRID_SIZE = ((N)-1) / BLOCK_SIZE + 1;
constexpr int BUCKETS_NUM = INIT_SIZE / MAX_BUCKET_SIZE;

template <class K>
__global__ void upsert_kernel(const K *__restrict keys,
                              const Bucket<K> *__restrict buckets, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int rank = g.thread_rank();

    int key_pos = -1;
    bool found_or_empty = false;
    const size_t bucket_max_size = MAX_BUCKET_SIZE;
    size_t key_idx = t / TILE_SIZE;
    K insert_key = keys[key_idx];
    K hashed_key = insert_key;  // Murmur3HashDevice(insert_key);
    size_t bkt_idx = hashed_key & (BUCKETS_NUM - 1);
    size_t start_idx = hashed_key & (bucket_max_size - 1);

    const Bucket<K> *bucket = buckets + bkt_idx;

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

  K *h_keys;
  K *d_keys;

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));
  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));
  Bucket<K> *buckets;
  cudaMallocManaged(&buckets, sizeof(Bucket<K>) * BUCKETS_NUM);
  for (int i = 0; i < BUCKETS_NUM; i++) {
    cudaMalloc(&(buckets[i].keys), sizeof(K) * MAX_BUCKET_SIZE);
    cudaMemset(&(buckets[i].keys), 0xFF, sizeof(K) * MAX_BUCKET_SIZE);
  }
  create_continuous_keys<K>(h_keys, KEY_NUM, 0);
  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);

  upsert_kernel<K><<<GRID_SIZE, BLOCK_SIZE>>>(d_keys, buckets, N);

  for (int i = 0; i < BUCKETS_NUM; i++) {
    cudaFree(&(buckets[i].keys));
  }
  cudaFree(buckets);
  cudaFreeHost(h_keys);
  cudaFree(d_keys);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
