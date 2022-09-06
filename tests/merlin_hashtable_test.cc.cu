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

#include <gtest/gtest.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
#include "merlin/memory.cuh"
#include "merlin_hashtable.cuh"

using namespace nv::merlin::memory;

uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <class K, class M, class V, size_t DIM>
void create_random_keys(K* h_keys, M* h_metas, V* h_vectors, int KEY_NUM) {
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
    if (h_metas != nullptr) {
      h_metas[i] = getTimestamp();
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < DIM; j++) {
        *(h_vectors + i * DIM + j) = static_cast<float>(num * 0.00001);
      }
    }
    i++;
  }
}

template <class K, class M>
void create_continuous_keys(K* h_keys, M* h_metas, int KEY_NUM, K start = 0) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

template <class K, class M, class V, size_t DIM>
void create_keys_in_one_buckets(K* h_keys, M* h_metas, V* h_vectors,
                                int KEY_NUM, int capacity,
                                int bucket_max_size = 128, int bucket_idx = 0) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  K candidate;
  K hashed_key;
  size_t global_idx;
  size_t bkt_idx;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    candidate = distr(eng) % 100000;
    hashed_key = Murmur3HashHost(candidate);
    global_idx = hashed_key & (capacity - 1);
    bkt_idx = global_idx / bucket_max_size;
    if (bkt_idx == bucket_idx) {
      numbers.insert(candidate);
    }
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_metas != nullptr) {
      h_metas[i] = num;
    }
    for (size_t j = 0; j < DIM; j++) {
      *(h_vectors + i * DIM + j) = static_cast<float>(num * 0.00001);
    }
    i++;
  }
}

template <class V, size_t DIM>
struct ValueArray {
  V value[DIM];
};

constexpr uint64_t DIM = 2;

using K = uint64_t;
using M = uint64_t;
using Vector = ValueArray<float, DIM>;
using Table = nv::merlin::HashTable<K, float, M, DIM>;
using TableOptions = nv::merlin::HashTableOptions;

template <class K, class M>
__forceinline__ __device__ bool erase_if_pred(const K& key, M& meta,
                                              const K& pattern,
                                              const M& threshold) {
  return ((key & 0x7f > pattern) && (meta > threshold));
}

template <class K, class M>
__device__ Table::Pred EraseIfPred = erase_if_pred<K, M>;

template <class K, class M>
__forceinline__ __device__ bool export_if_pred(const K& key, M& meta,
                                               const K& pattern,
                                               const M& threshold) {
  return meta > threshold;
}

template <class K, class M>
__device__ Table::Pred ExportIfPred = export_if_pred<K, M>;

void test_basic() {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  M* h_metas;
  Vector* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector)));

  create_random_keys<K, M, float, DIM>(h_keys, h_metas, nullptr, KEY_NUM);

  K* d_keys;
  M* d_metas = nullptr;
  Vector* d_vectors;
  Vector* d_def_val;
  Vector** d_vectors_ptr;
  bool* d_found;

  //  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_def_val, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector*)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMemset(d_def_val, 2, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector*)));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  DeviceMemory<size_t> dump_counter = DeviceMemory<size_t>(1, stream);
  dump_counter.memset(0);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);

    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == KEY_NUM);

    CUDA_CHECK(cudaMemset(d_vectors, 2, KEY_NUM * sizeof(Vector)));
    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == KEY_NUM);

    table->find(KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_found,
                nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
    }
    ASSERT_TRUE(found_num == KEY_NUM);

    table->accum_or_assign(KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors),
                           d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == KEY_NUM);

    size_t erase_num = table->erase(KEY_NUM >> 1, d_keys, stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(erase_num == total_size);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);

    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);

    table->export_batch(table->capacity(), 0, dump_counter.get(), d_keys,
                        reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //    ASSERT_TRUE(dump_counter == KEY_NUM);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas))
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_erase_if_pred() {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr uint64_t BUCKET_MAX_SIZE = 128;

  K* h_keys;
  M* h_metas;
  Vector* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  K* d_keys;
  M* d_metas = nullptr;
  Vector* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    create_keys_in_one_buckets<K, M, float, DIM>(
        h_keys, h_metas, reinterpret_cast<float*>(h_vectors), KEY_NUM,
        INIT_CAPACITY);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);

    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == BUCKET_MAX_SIZE);

    K pattern = 100;
    M threshold = 0;
    size_t erase_num =
        table->erase_if(EraseIfPred<K, M>, pattern, threshold, stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE((erase_num + total_size) == BUCKET_MAX_SIZE);

    table->find(KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_found,
                d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector)));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) {
        found_num++;
        ASSERT_TRUE(h_metas[i] == h_keys[i]);
        for (int j = 0; j < DIM; j++) {
          ASSERT_TRUE(h_vectors[i].value[j] ==
                      static_cast<float>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_TRUE(found_num == (BUCKET_MAX_SIZE - erase_num));

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas))
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_rehash() {
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = 2 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE * 2;
  constexpr uint64_t TEST_TIMES = 100;
  K* h_keys;
  M* h_metas;
  Vector* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  K* d_keys;
  M* d_metas = nullptr;
  Vector* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  DeviceMemory<size_t> dump_counter = DeviceMemory<size_t>(1, stream);
  PinnedMemory<size_t> p_dump_counter = PinnedMemory<size_t>(1, stream);

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    create_keys_in_one_buckets<K, M, float, DIM>(
        h_keys, h_metas, reinterpret_cast<float*>(h_vectors), KEY_NUM,
        INIT_CAPACITY, BUCKET_MAX_SIZE);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);

    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_TRUE(total_size == BUCKET_MAX_SIZE);

    table->export_batch(table->capacity(), 0, dump_counter.get(), d_keys,
                        reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //    p_dump_counter.copy_from(&dump_counter);
    //    ASSERT_TRUE(p_dump_counter[0] == BUCKET_MAX_SIZE);

    table->reserve(MAX_CAPACITY, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(table->capacity() == MAX_CAPACITY);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == BUCKET_MAX_SIZE);

    table->find(BUCKET_MAX_SIZE, d_keys, reinterpret_cast<float*>(d_vectors),
                d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector)));
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < BUCKET_MAX_SIZE; i++) {
      if (h_found[i]) {
        found_num++;
        ASSERT_TRUE(h_metas[i] == h_keys[i]);
        for (int j = 0; j < DIM; j++) {
          ASSERT_TRUE(h_vectors[i].value[j] ==
                      static_cast<float>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_TRUE(found_num == BUCKET_MAX_SIZE);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas))
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_dynamic_rehash_on_multi_threads() {
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = 1 * 1024;
  constexpr uint64_t MAX_CAPACITY = 128 * 1024 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 256;
  constexpr uint64_t THREAD_N = 4;

  std::vector<std::thread> threads;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_load_factor = 0.50f;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  options.evict_strategy = nv::merlin::EvictStrategy::kLru;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  auto worker_function = [&table](int task_n) {
    K* h_keys;
    Vector* h_vectors;
    bool* h_found;

    size_t current_capacity = table->capacity();

    CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector)));
    CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

    K* d_keys;
    Vector* d_vectors;
    bool* d_found;

    CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    while (table->capacity() < MAX_CAPACITY) {
      create_random_keys<K, M, float, DIM>(
          h_keys, nullptr, reinterpret_cast<float*>(h_vectors), KEY_NUM);
      CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(Vector),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

      table->insert_or_assign(KEY_NUM, d_keys,
                              reinterpret_cast<float*>(d_vectors), nullptr,
                              stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      table->find(BUCKET_MAX_SIZE, d_keys, reinterpret_cast<float*>(d_vectors),
                  d_found, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      int found_num = 0;

      CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
      CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector)));
      CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                            cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < BUCKET_MAX_SIZE; i++) {
        if (h_found[i]) {
          found_num++;
          for (int j = 0; j < DIM; j++) {
            ASSERT_TRUE(h_vectors[i].value[j] ==
                        static_cast<float>(h_keys[i] * 0.00001));
          }
        }
      }
      if (task_n == 0 && current_capacity != table->capacity()) {
        std::cout << "[test_dynamic_rehash_on_multi_threads] The capacity "
                     "changed from "
                  << current_capacity << " to " << table->capacity()
                  << std::endl;
        current_capacity = table->capacity();
      }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFreeHost(h_found));

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaCheckError();
  };

  for (int i = 0; i < THREAD_N; ++i)
    threads.emplace_back(std::thread(worker_function, i));

  for (auto& th : threads) {
    th.join();
  }
  ASSERT_TRUE(table->capacity() == MAX_CAPACITY);
}

void test_export_if_batch() {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 1;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  PinnedMemory<K> h_keys = PinnedMemory<K>(KEY_NUM);
  PinnedMemory<M> h_metas = PinnedMemory<M>(KEY_NUM);
  PinnedMemory<Vector> h_vectors = PinnedMemory<Vector>(KEY_NUM);

  PinnedMemory<size_t> dump_counter = PinnedMemory<size_t>(1);

  DeviceMemory<K> d_keys = DeviceMemory<K>(KEY_NUM);
  DeviceMemory<M> d_metas = DeviceMemory<M>(KEY_NUM);
  DeviceMemory<Vector> d_vectors = DeviceMemory<Vector>(KEY_NUM);

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    create_random_keys<K, M, float, DIM>(
        h_keys.get(), h_metas.get(), reinterpret_cast<float*>(h_vectors.get()),
        KEY_NUM);

    d_keys.copy_from(&h_keys);
    d_metas.copy_from(&h_metas);
    d_vectors.copy_from(&h_vectors);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);

    table->insert_or_assign(KEY_NUM, d_keys.get(),
                            reinterpret_cast<float*>(d_vectors.get()),
                            d_metas.get(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == KEY_NUM);

    K pattern = 100;
    M threshold = h_metas.get()[size_t(KEY_NUM / 2)];

    table->export_if_batch(
        ExportIfPred<K, M>, pattern, threshold, table->capacity(), 0,
        dump_counter.get(), d_keys.get(),
        reinterpret_cast<float*>(d_vectors.get()), d_metas.get(), stream);

    size_t expected_export_count = 0;
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_metas[i] > threshold) expected_export_count++;
    }
    ASSERT_TRUE(expected_export_count == dump_counter[0]);

    h_metas.memset(0);
    h_vectors.memset(0);

    h_metas.copy_from(&d_metas);
    h_vectors.copy_from(&d_vectors);

    for (int i = 0; i < dump_counter[0]; i++) {
      ASSERT_TRUE(h_metas.get()[i] > threshold);
    }

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(total_size == 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaStreamDestroy(stream));

  CudaCheckError();
}

TEST(MerlinHashTableTest, test_basic) { test_basic(); }
// TEST(MerlinHashTableTest, test_erase_if_pred) { test_erase_if_pred(); }
// TEST(MerlinHashTableTest, test_rehash) { test_rehash(); }
// TEST(MerlinHashTableTest, test_dynamic_rehash_on_multi_threads) {
//  test_dynamic_rehash_on_multi_threads();
//}
// TEST(MerlinHashTableTest, test_export_if_batch) { test_export_if_batch(); }
