# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)
project(merlin-hkvs LANGUAGES CXX CUDA)
find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Default to release build.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

# Some neat defaults.
set(CUDA_SEPARABLE_COMPILATION ON)

# Select target CUDA binary architecture.
foreach(cuda_arch ${sm})
  list(APPEND cuda_arch_list ${cuda_arch})
  message(STATUS "Assign GPU architecture (sm=${cuda_arch})")
endforeach()
list(LENGTH cuda_arch_list cuda_arch_list_length)
if(cuda_arch_list_length EQUAL 0)
  list(APPEND cuda_arch_list "80")
  message(STATUS "Assign default GPU architecture sm=80")
endif()
foreach(cuda_arch ${cuda_arch_list})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
endforeach()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tests/googletest/googletest/include
)

ADD_SUBDIRECTORY(tests/googletest)

link_directories(
)

file(GLOB_RECURSE merlin_hkvs_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

# TODO:
# add_library(merlin_kv STATIC ${merlin_kv_src})
# target_compile_features(merlin_kv PUBLIC cxx_std_14)
# target_link_libraries(merlin_kv PUBLIC ...)


add_executable(merlin_hashtable_benchmark benchmark/merlin_hashtable_benchmark.cc.cu)
target_compile_features(merlin_hashtable_benchmark PUBLIC cxx_std_14)
set_target_properties(merlin_hashtable_benchmark PROPERTIES  CUDA_ARCHITECTURES OFF)

add_executable(merlin_hashtable_test tests/merlin_hashtable_test.cc.cu)
target_compile_features(merlin_hashtable_test PUBLIC cxx_std_14)
set_target_properties(merlin_hashtable_test PROPERTIES  CUDA_ARCHITECTURES OFF)
TARGET_LINK_LIBRARIES(merlin_hashtable_test gtest_main)

# TODO(Lifann): remove this line after implementing all of the unit testing to gtest.
set(CMAKE_BUILD_TYPE "Debug")
add_executable(save_and_load_test tests/save_and_load_test.cc.cu)
target_compile_features(save_and_load_test PUBLIC cxx_std_14)
set_target_properties(save_and_load_test PROPERTIES  CUDA_ARCHITECTURES OFF)
TARGET_LINK_LIBRARIES(save_and_load_test gtest_main)
