/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "merlin/initializers.cuh"
#include "merlin/optimizers.cuh"
#include "merlin_hashtable.cuh"

uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
template <class K, class M>
void create_random_keys(K *h_keys, M *h_metas, int KEY_NUM) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_metas[i] = getTimestamp();
    i++;
  }
}

template <class K, class M>
void create_continuous_keys(K *h_keys, M *h_metas, int KEY_NUM, K start = 0) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

template <class V, size_t DIM>
struct ValueArray {
  V value[DIM];
};

constexpr uint64_t INIT_SIZE = 64 * 1024 * 1024UL;
constexpr uint64_t MAX_SIZE = INIT_SIZE;
constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
constexpr uint64_t DIM = 4;

template <class K, class M>
__forceinline__ __device__ bool erase_if_pred(const K &key, const M &meta) {
  return ((key % 2) == 1);
}

using K = uint64_t;
using M = uint64_t;
using Vector = ValueArray<float, DIM>;
using Table = nv::merlin::HashTable<K, float, M, DIM>;

/* A demo of Pred for erase_if */
template <class K, class M>
__device__ Table::Pred pred = erase_if_pred<K, M>;

int test_main() {
  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  bool *h_found;

  std::unique_ptr<Table> table_ =
      std::make_unique<Table>(INIT_SIZE,          /* init_size */
                              MAX_SIZE,           /* max_size */
                              nv::merlin::GB(16), /* max_hbm_for_vectors */
                              0.75,               /* max_load_factor */
                              128,                /* buckets_max_size */
                              nullptr,            /* initializer */
                              true,               /* primary */
                              1024                /* block_size */
      );

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, KEY_NUM * sizeof(M));         // 8MB
  cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_found, KEY_NUM * sizeof(bool));      // 4MB

  cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector));

  K *d_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;

  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));          // 8MB
  cudaMalloc(&d_metas, KEY_NUM * sizeof(M));         // 8MB
  cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector));  // 256MB

  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  K start = 0UL;
  float cur_load_factor = table_->load_factor();

  create_continuous_keys<K, M>(h_keys, h_metas, KEY_NUM, start);
  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                           d_metas, KEY_NUM, false, stream);

  auto start_insert_or_assign = std::chrono::steady_clock::now();
  table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                           d_metas, KEY_NUM, false, stream);
  auto end_insert_or_assign = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_insert_or_assign =
      end_insert_or_assign - start_insert_or_assign;
  printf("[prepare] insert_or_assign=%.2fms\n",
         diff_insert_or_assign.count() * 1000);
  cudaStreamDestroy(stream);

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);

  std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

  return 0;
}

int main() {
  test_main();
  return 0;
}