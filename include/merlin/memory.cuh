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

#pragma once

#include "utils.cuh"

namespace nv {
namespace merlin {
namespace memory {

template <class T>
class CudaMemory {
 public:
  CudaMemory(size_t n, cudaStream_t stream = 0) {
    n_ = n;
    stream_ = stream;
  }

  virtual ~CudaMemory() {}

  T* get() const { return ptr_; }

  void copy_from(const CudaMemory<T>* src) {
    MERLIN_CHECK(
        size() == src->size(),
        "[CudaMemory] `src` should have save size with this instance.");
    CUDA_CHECK(cudaMemcpyAsync(ptr_, src->get(), sizeof(T) * n_,
                               cudaMemcpyDefault, stream_));
  }

  void memset(int value) {
    CUDA_CHECK(cudaMemsetAsync(ptr_, value, sizeof(T) * n_, stream_));
  }

  size_t size() const { return sizeof(T) * n_; }
  cudaStream_t stream() const { return stream_; }
  void stream(cudaStream_t stream) { return stream_ = stream; }

  void stream_sync() {}

 public:
  T* ptr_ = nullptr;

 private:
  size_t n_;
  cudaStream_t stream_;
};

template <class T>
class DeviceMemory final : public CudaMemory<T> {
 public:
  DeviceMemory(size_t n, cudaStream_t stream = 0) : CudaMemory<T>(n, stream) {
    CUDA_CHECK(cudaMallocAsync((void **)&CudaMemory<T>::ptr_, CudaMemory<T>::size(),
                               CudaMemory<T>::stream()));
  };

  ~DeviceMemory() override {
    if (CudaMemory<T>::ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeAsync(CudaMemory<T>::ptr_, CudaMemory<T>::stream()));
    }
  }
};

template <class T>
class PinnedMemory final : public CudaMemory<T> {
 public:
  explicit PinnedMemory(size_t n, cudaStream_t stream = 0)
      : CudaMemory<T>(n, stream) {
    CUDA_CHECK(cudaMallocHost((void **)&CudaMemory<T>::ptr_, CudaMemory<T>::size()));
  };

  T& operator[](size_t idx) { return CudaMemory<T>::ptr_[idx]; }

  ~PinnedMemory() override {
    if (CudaMemory<T>::ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeHost(CudaMemory<T>::ptr_));
    }
  }
};

template <class T>
class ManagedMemory final : public CudaMemory<T> {
 public:
  explicit ManagedMemory(size_t n, bool need_memset = false,
                         cudaStream_t stream = 0)
      : CudaMemory<T>(n, stream) {
    CUDA_CHECK(cudaMallocManaged((void **)&CudaMemory<T>::ptr_, CudaMemory<T>::size()));
  };

  T& operator[](size_t idx) { return CudaMemory<T>::ptr_[idx]; }

  ~ManagedMemory() override {
    if (CudaMemory<T>::ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(CudaMemory<T>::ptr_));
    }
  }
};

}  // namespace memory
}  // namespace merlin
}  // namespace nv