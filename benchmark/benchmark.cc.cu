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
void create_random_keys(K *h_keys, M *h_metas, int key_num_per_op) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < key_num_per_op) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_metas[i] = getTimestamp();
    i++;
  }
}

template <class K, class M>
void create_continuous_keys(K *h_keys, M *h_metas, int key_num_per_op,
                            K start = 0) {
  for (K i = 0; i < key_num_per_op; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

template <class V, size_t DIM>
struct ValueArray {
  V value[DIM];
};

template <class K, class M, size_t DIM>
int test_main(size_t init_capacity = 64 * 1024 * 1024UL,
              size_t key_num_per_op = 1 * 1024 * 1024UL,
              size_t max_hbm_for_vectors_by_gb = 16,
              float target_load_factor = 1.0) {
  using Vector = ValueArray<float, DIM>;
  using Table = nv::merlin::HashTable<K, float, M, DIM>;

  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  bool *h_found;

  std::unique_ptr<Table> table_ =
      std::make_unique<Table>(init_capacity,      /* init_capacity */
                              init_capacity,      /* max_size */
                              nv::merlin::GB(16), /* max_hbm_for_vectors */
                              0.75,               /* max_load_factor */
                              128,                /* buckets_max_size */
                              nullptr,            /* initializer */
                              true,               /* primary */
                              1024                /* block_size */
      );

  cudaMallocHost(&h_keys, key_num_per_op * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, key_num_per_op * sizeof(M));         // 8MB
  cudaMallocHost(&h_vectors, key_num_per_op * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_found, key_num_per_op * sizeof(bool));      // 4MB

  cudaMemset(h_vectors, 0, key_num_per_op * sizeof(Vector));

  create_random_keys<K, M>(h_keys, h_metas, key_num_per_op);

  K *d_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;
  Vector *d_def_val;
  Vector **d_vectors_ptr;
  bool *d_found;
  size_t dump_counter = 0;

  cudaMalloc(&d_keys, key_num_per_op * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, key_num_per_op * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, key_num_per_op * sizeof(Vector));        // 256MB
  cudaMalloc(&d_def_val, key_num_per_op * sizeof(Vector));        // 256MB
  cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_found, key_num_per_op * sizeof(bool));            // 4MB

  cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
             cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, key_num_per_op * sizeof(Vector));
  cudaMemset(d_def_val, 2, key_num_per_op * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, key_num_per_op * sizeof(Vector *));
  cudaMemset(d_found, 0, key_num_per_op * sizeof(bool));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  K start = 0UL;
  float cur_load_factor = table_->load_factor();
  auto start_insert_or_assign = std::chrono::steady_clock::now();
  auto end_insert_or_assign = std::chrono::steady_clock::now();
  auto start_find = std::chrono::steady_clock::now();
  auto end_find = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff_insert_or_assign;
  std::chrono::duration<double> diff_find;

  while (cur_load_factor < target_load_factor) {
    create_continuous_keys<K, M>(h_keys, h_metas, key_num_per_op, start);
    cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
               cudaMemcpyHostToDevice);

    start_insert_or_assign = std::chrono::steady_clock::now();
    table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                             d_metas, key_num_per_op, false, stream);
    end_insert_or_assign = std::chrono::steady_clock::now();
    diff_insert_or_assign = end_insert_or_assign - start_insert_or_assign;

    start_find = std::chrono::steady_clock::now();
    table_->find(d_keys, reinterpret_cast<float *>(d_vectors), d_found,
                 key_num_per_op, nullptr, stream);
    end_find = std::chrono::steady_clock::now();
    diff_find = end_find - start_find;

    cur_load_factor = table_->load_factor();

    start += key_num_per_op;
  }

  size_t hmem_for_vectors_by_gb =
      init_capacity * DIM * sizeof(float) / 1024 / 1024 / 1024 -
      max_hbm_for_vectors_by_gb;
  printf(
      "|\t%d |\t%lu |\t%.2f |\t%lu |\t%lu "
      "|\t%.3f |\t%.3f |\n",
      dim, key_num_per_op, target_load_factor, max_hbm_for_vectors_by_gb,
      hmem_for_vectors_by_gb,
      key_num_per_op / diff_insert_or_assign.count() / (1024 * 1024 * 1024),
      key_num_per_op / diff_find.count() / (1024 * 1024 * 1024),
      cur_load_factor);

  cudaStreamDestroy(stream);

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_def_val);
  cudaFree(d_vectors_ptr);
  cudaFree(d_found);

  return 0;
}

int main() {
  printf(
      "|\tdim |\tkeys_num_per_op |\tload_factor |\t HBM(GB) |\t HMEM(GB) "
      "|\tinsert_or_assign(G-KV/s) |\tfind(G-KV/s) |\n");
  printf(
      "|\t---:|\t---------------:|\t-----------:|\t--------:|\t----------------"
      "-:|\t---"
      "--------------:|\t----:|\n");
  test_main<uint64_t, uint64_t, 4>(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 16,
                                   0.5);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 4, 16, 0.75);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 4, 16, 1.0);
  //
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 4, 0, 0.5);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 4, 0, 0.75);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 4, 0, 1.0);

  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 128, 64, 0.5);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 128, 64, 0.75);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 128, 64, 1.0);
  //
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 128, 64, 0.5);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 128, 64, 0.75);
  //  test_main(64 * 1024 * 1024UL, 1 * 1024 * 1024UL, 128, 64, 1.0);
  return 0;
}
