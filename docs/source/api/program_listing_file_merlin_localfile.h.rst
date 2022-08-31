
.. _program_listing_file_merlin_localfile.h:

Program Listing for File merlin_localfile.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_merlin_localfile.h>` (``merlin_localfile.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

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
   
   #include <stdio.h>
   #include <string>
   #include "merlin/types.cuh"
   
   namespace nv {
   namespace merlin {
   
   template <class K, class V, class M, size_t D>
   class LocalKVFile : public BaseKVFile<K, V, M, D> {
    public:
     LocalKVFile() : keys_file_path_(nullptr), values_file_path_(nullptr) {}
   
     ~LocalKVFile() { Close(); }
   
     int Open(std::string& keyfile, std::string& valuefile, const char* mode) {
       keys_file_path_ = fopen(keyfile.c_str(), mode);
       if (!keys_file_path_) {
         return -1;
       }
       values_file_path_ = fopen(valuefile.c_str(), mode);
       if (!values_file_path_) {
         fclose(keys_file_path_);
         keys_file_path_ = nullptr;
         return -1;
       }
       return 0;
     }
   
     int Close() {
       if (keys_file_path_) {
         fclose(keys_file_path_);
         keys_file_path_ = nullptr;
       }
       if (values_file_path_) {
         fclose(values_file_path_);
         values_file_path_ = nullptr;
       }
       return 0;
     }
   
     ssize_t Read(size_t n, K* keys, V* vectors, M* metas) {
       size_t nread_keys = fread(keys, sizeof(K), n, keys_file_path_);
       size_t nread_vevs = fread(vectors, sizeof(V) * D, n, values_file_path_);
       if (nread_keys != nread_vevs) {
         return -1;
       }
       return static_cast<ssize_t>(nread_keys);
     }
   
     ssize_t Write(size_t n, const K* keys, const V* vectors, const M* metas) {
       size_t nwritten_keys = fwrite(keys, sizeof(K), n, keys_file_path_);
       size_t nwritten_vecs = fwrite(vectors, sizeof(V) * D, n, values_file_path_);
       if (nwritten_keys != nwritten_vecs) {
         return -1;
       }
       return static_cast<ssize_t>(nwritten_keys);
     }
   
    private:
     FILE* keys_file_path_;
     FILE* values_file_path_;
   };
   
   }  // namespace merlin
   }  // namespace nv
