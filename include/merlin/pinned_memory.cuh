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

template <class T>
class cudaPinnedMemory {
 public:
  HashTable(size_t n, bool need_memset = false, cudaStream_t stream = 0) {
    stream_ = stream;
    CUDA_CHECK(cudaMallocHost(&ptr_, n));
    CUDA_CHECK(cudaMemsetAsync(&ptr_, n, stream_));
  };

  ~HashTable() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFreeHost(ptr_));
    }
  }

  T* get() { return ptr_; }

 private:
  size_t size_;
  T* ptr_ = nullptr;
  cudaStream_t stream_;
};

}  // namespace merlin
}  // namespace nv