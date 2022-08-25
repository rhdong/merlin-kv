/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <typeindex>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

// Never
#include "merlin_kv/tensorflow/core/lib/merlin-kv/cpp/include/merlin_hashtable.cuh"
// CaseInsensitive

namespace tensorflow {
namespace merlin_kv {
namespace lookup {
namespace gpu {

using GPUDevice = Eigen::ThreadPoolDevice;

template <class V>
struct ValueArrayBase {};

template <class V, size_t DIM>
struct ValueArray : public ValueArrayBase<V> {
  V value[DIM];
};

template <class T>
using ValueType = ValueArrayBase<T>;

template <class K, class V, class M = uint64_t>
class TableWrapperBase {
 public:
  virtual ~TableWrapperBase() {}
  virtual void upsert(const K* d_keys, const ValueType<V>* d_vals, size_t len,
                      bool allow_duplicated_keys, cudaStream_t stream) {}
  virtual void upsert(const K* d_keys, const ValueType<V>* d_vals,
                      const M* d_metas, size_t len, bool allow_duplicated_keys,
                      cudaStream_t stream) {}
  virtual void accum(const K* d_keys, const ValueType<V>* d_vals_or_deltas,
                     const bool* d_exists, size_t len, cudaStream_t stream) {}
  virtual void dump(K* d_key, ValueType<V>* d_val, const size_t offset,
                    const size_t search_length, cudaStream_t stream) const {}
  virtual void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status,
                   size_t len, const ValueType<V>* d_def_val,
                   bool is_full_size_default, cudaStream_t stream) const {}

  virtual void get(const K* d_keys, ValueType<V>* d_vals, M* d_metas,
                   bool* d_status, size_t len, const ValueType<V>* d_def_val,
                   bool is_full_size_default, cudaStream_t stream) const {}
  virtual size_t get_size(cudaStream_t stream) const { return 0; }
  virtual size_t get_capacity() const { return 0; }
  virtual void remove(const K* d_keys, size_t len, cudaStream_t stream) {}
  virtual void clear(cudaStream_t stream) {}
};

template <class K, class V, size_t DIM, class M = uint64_t>
class TableWrapper final : public TableWrapperBase<K, V, M> {
 private:
  using Table = nv::merlin::HashTable<K, V, M, DIM>;
  using TableOptions = nv::merlin::HashTableOptions;

 public:
  TableWrapper(nv::merlin::HashTableOptions options) {
    options_ = options;
    table_ = new Table();
    table_->init(options_);
  }

  ~TableWrapper() override { delete table_; }

  void upsert(const K* d_keys, const ValueType<V>* d_vals, size_t len,
              bool allow_duplicated_keys, cudaStream_t stream) override {
    table_->insert_or_assign(len, d_keys, (const V*)d_vals, nullptr, stream);
  }

  void upsert(const K* d_keys, const ValueType<V>* d_vals, const M* d_metas,
              size_t len, bool allow_duplicated_keys,
              cudaStream_t stream) override {
    table_->insert_or_assign(len, d_keys, (const V*)d_vals, d_metas, stream);
  }

  void accum(const K* d_keys, const ValueType<V>* d_vals_or_deltas,
             const bool* d_exists, size_t len, cudaStream_t stream) override {
    table_->accum_or_assign(len, d_keys, (const V*)d_vals_or_deltas, d_exists,
                            nullptr, stream);
  }

  void dump(K* d_key, ValueType<V>* d_val, const size_t offset,
            const size_t search_length, cudaStream_t stream) const override {
    table_->export_batch(search_length, 0, d_key, (V*)d_val, nullptr, stream);
  }

  void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status, size_t len,
           const ValueType<V>* d_def_val, bool is_full_size_default,
           cudaStream_t stream) const override {
    if (is_full_size_default) {
      CUDA_CHECK(cudaMemcpy((void*)d_vals, (void*)d_def_val,
                            sizeof(ValueArray<V, DIM>) * len,
                            cudaMemcpyDefault));

    } else {
      const size_t N = len;
      thrust::device_ptr<ValueArray<V, DIM>> d_vals_ptr(
          reinterpret_cast<ValueArray<V, DIM>*>(d_vals));
      thrust::device_ptr<const ValueArray<V, DIM>> d_def_val_ptr(
          reinterpret_cast<const ValueArray<V, DIM>*>(d_def_val));

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::fill(policy, d_vals_ptr, d_vals_ptr + N, *d_def_val_ptr);
    }

    table_->find(len, d_keys, (V*)d_vals, d_status, nullptr, stream);
  }

  void get(const K* d_keys, ValueType<V>* d_vals, M* d_metas, bool* d_status,
           size_t len, const ValueType<V>* d_def_val, bool is_full_size_default,
           cudaStream_t stream) const override {
    if (is_full_size_default) {
      CUDA_CHECK(cudaMemcpy((void*)d_vals, (void*)d_def_val,
                            sizeof(ValueArray<V, DIM>) * len,
                            cudaMemcpyDefault));

    } else {
      const size_t N = len;
      thrust::device_ptr<ValueArray<V, DIM>> d_vals_ptr(
          reinterpret_cast<ValueArray<V, DIM>*>(d_vals));
      thrust::device_ptr<const ValueArray<V, DIM>> d_def_val_ptr(
          reinterpret_cast<const ValueArray<V, DIM>*>(d_def_val));

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::fill(policy, d_vals_ptr, d_vals_ptr + N, *d_def_val_ptr);
    }
    table_->find(len, d_keys, (V*)d_vals, d_status, d_metas, stream);
  }

  size_t get_size(cudaStream_t stream) const override {
    return table_->size(stream);
  }

  size_t get_capacity() const override { return table_->capacity(); }

  void remove(const K* d_keys, size_t len, cudaStream_t stream) override {
    table_->erase(len, d_keys, stream);
  }

  void clear(cudaStream_t stream) override { table_->clear(stream); }

 private:
  Table* table_;
  TableOptions options_;
};

#define CREATE_A_TABLE(DIM)                                  \
  do {                                                       \
    if (runtime_dim == (DIM + 1)) {                          \
      *pptable = new TableWrapper<K, V, (DIM + 1)>(options); \
    };                                                       \
  } while (0)

#define CREATE_TABLE_PARTIAL_BRANCHES(PERIFX) \
  do {                                        \
    CREATE_A_TABLE((PERIFX)*10 + 0);          \
    CREATE_A_TABLE((PERIFX)*10 + 1);          \
    CREATE_A_TABLE((PERIFX)*10 + 2);          \
    CREATE_A_TABLE((PERIFX)*10 + 3);          \
    CREATE_A_TABLE((PERIFX)*10 + 4);          \
    CREATE_A_TABLE((PERIFX)*10 + 5);          \
    CREATE_A_TABLE((PERIFX)*10 + 6);          \
    CREATE_A_TABLE((PERIFX)*10 + 7);          \
    CREATE_A_TABLE((PERIFX)*10 + 8);          \
    CREATE_A_TABLE((PERIFX)*10 + 9);          \
  } while (0)

// create branches with dim range:
// [CENTILE * 100 + (DECTILE) * 10, CENTILE * 100 + (DECTILE) * 10 + 50]
#define CREATE_TABLE_BRANCHES(CENTILE, DECTILE)              \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 0); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 1); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 2); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 3); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 4);

template <class K, class V, int centile, int dectile>
void CreateTableImpl(TableWrapperBase<K, V>** pptable, size_t runtime_dim,
                     nv::merlin::HashTableOptions options) {
  CREATE_TABLE_BRANCHES(centile, dectile);
}

#define DEFINE_CREATE_TABLE(ID, K, V, CENTILE, DECTILE)                      \
  void CreateTable##ID(size_t runtime_dim, TableWrapperBase<K, V>** pptable, \
                       nv::merlin::HashTableOptions options) {               \
    CreateTableImpl<K, V, CENTILE, DECTILE>(pptable, runtime_dim, options);  \
  }

#define DECLARE_CREATE_TABLE(K, V)                                \
  void CreateTable0(size_t runtime_dim, TableWrapperBase<K, V>**, \
                    nv::merlin::HashTableOptions);                \
  void CreateTable1(size_t runtime_dim, TableWrapperBase<K, V>**, \
                    nv::merlin::HashTableOptions);                \
  void CreateTable2(size_t runtime_dim, TableWrapperBase<K, V>**, \
                    nv::merlin::HashTableOptions);                \
  void CreateTable3(size_t runtime_dim, TableWrapperBase<K, V>**, \
                    nv::merlin::HashTableOptions);

DECLARE_CREATE_TABLE(int64, float);
DECLARE_CREATE_TABLE(int64, Eigen::half);
DECLARE_CREATE_TABLE(int64, int64);
DECLARE_CREATE_TABLE(int64, int32);
DECLARE_CREATE_TABLE(int64, int8);

#undef CREATE_A_TABLE
#undef CREATE_DEFAULT_TABLE
#undef CREATE_TABLE_PARTIAL_BRANCHES
#undef CREATE_TABLE_ALL_BRANCHES
#undef DECLARE_CREATE_TABLE

}  // namespace gpu
}  // namespace lookup
}  // namespace merlin_kv
}  // namespace tensorflow
