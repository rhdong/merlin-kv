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

  virtual T* get() const { return nullptr; }

  void copy_from(const CudaMemory<T>* src) {
    MERLIN_CHECK(
        size() == src->size(),
        "[CudaMemory] `src` should have save size with this instance.");
    CUDA_CHECK(cudaMemcpyAsync(get(), src->get(), sizeof(T) * n_,
                               cudaMemcpyDefault, stream_));
  }

  void memset(int value) {
    CUDA_CHECK(cudaMemsetAsync(get(), value, sizeof(T) * n_, stream_));
  }

  size_t size() const { return sizeof(T) * n_; }
  cudaStream_t stream() const {
    return cudaStreamQuery(stream_) == cudaSuccess ? stream_ : 0;
  }
  void stream(cudaStream_t stream) { return stream_ = stream; }

  void stream_sync() {}

 private:
  size_t n_;
  cudaStream_t stream_;
};

template <class T>
class DeviceMemory final : public CudaMemory<T> {
 public:
  DeviceMemory(size_t n, cudaStream_t stream = 0) : CudaMemory<T>(n, stream) {
    CUDA_CHECK(
        cudaMallocAsync(&ptr_, CudaMemory<T>::size(), CudaMemory<T>::stream()));
  };

  ~DeviceMemory() override {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeAsync(ptr_, CudaMemory<T>::stream()));
    }
  }
  T* get() const override { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

template <class T>
class PinnedMemory final : public CudaMemory<T> {
 public:
  explicit PinnedMemory(size_t n, cudaStream_t stream = 0)
      : CudaMemory<T>(n, stream) {
    CUDA_CHECK(cudaMallocHost(&ptr_, CudaMemory<T>::size()));
  };

  T& operator[](size_t idx) { return ptr_[idx]; }

  T* get() const override { return ptr_; }

  ~PinnedMemory() override {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeHost(ptr_));
    }
  }

 private:
  T* ptr_ = nullptr;
};

template <class T>
class ManagedMemory final : public CudaMemory<T> {
 public:
  explicit ManagedMemory(size_t n, bool need_memset = false,
                         cudaStream_t stream = 0)
      : CudaMemory<T>(n, stream) {
    CUDA_CHECK(cudaMallocManaged(&ptr_, CudaMemory<T>::size()));
  };

  T& operator[](size_t idx) { return ptr_[idx]; }

  T* get() const override { return ptr_; }

  ~ManagedMemory() override {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
    }
  }

 private:
  T* ptr_ = nullptr;
};

}  // namespace memory
}  // namespace merlin
}  // namespace nv