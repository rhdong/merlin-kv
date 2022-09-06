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
class DeviceMemory {
 public:
  DeviceMemory(size_t n, bool need_memset = false, cudaStream_t stream = 0) {
    n_ = n;
    stream_ = stream;
    CUDA_CHECK(cudaMallocAsync(&ptr_, sizeof(T) * n_, stream_));
    CUDA_CHECK(cudaMemsetAsync(ptr_, 0, sizeof(T) * n_, stream_));
  };

  ~DeviceMemory() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeAsync(ptr_, stream_));
    }
  }

  inline T* get() { return ptr_; }

 private:
  size_t n_;
  T* ptr_ = nullptr;
  cudaStream_t stream_;
};

template <class T>
class PinnedMemory {
 public:
  PinnedMemory(size_t n, bool need_memset = false, cudaStream_t stream = 0) {
    n_ = n;
    stream_ = stream;
    CUDA_CHECK(cudaMallocHost(&ptr_, sizeof(T) * n_));
    CUDA_CHECK(cudaMemsetAsync(&ptr_, 0, sizeof(T) * n_, stream_));
  };

  ~PinnedMemory() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeHost(ptr_));
    }
  }

  inline T* get() { return ptr_; }

 private:
  size_t n_;
  T* ptr_ = nullptr;
  cudaStream_t stream_;
};

}  // namespace memory
}  // namespace merlin
}  // namespace nv