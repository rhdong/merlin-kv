#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# lint-as: python3

from abc import ABCMeta
from enum import Enum
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_parsing_ops
from merlin_kv import tensorflow as mkv


class KVCreator(object, metaclass=ABCMeta):
  """
  A generic KV table creator.

    KV table instance will be created by the create function with config.
  And also a config class for specific table instance backend should be
  inited before callling the creator function.
    And then, the KVCreator class instance will be passed to the Variable
  class for creating the real KV table backend(TF resource).

    Example usage:

    ```python
    from merlin_kv import tensorflow as mkv

    mkv_config=mkv.MerlinKVConfig(
        merlin_kv_config_abs_dir="xx/yy.json")
    mkv_creator=mkv.MerlinKVCreator(mkv_config)
    ```
  """

  def __init__(self, config=None):
    self.config = config

  def create(self,
             key_dtype=None,
             value_dtype=None,
             default_value=None,
             name=None,
             checkpoint=None,
             init_size=None,
             config=None):

    raise NotImplementedError('create function must be implemented')


class MerlinKVEvictStrategy(object):
  LRU = int(0)
  CUTOMIZED = int(1)


class MerlinKVConfig(object):

  def __init__(
      self,
      init_capacity=0,
      max_capacity=0,
      max_hbm_for_vectors=0,
      max_bucket_size=128,
      max_load_factor=0.5,
      device_id=0,
      evict_strategy=MerlinKVEvictStrategy.LRU,
  ):
    """
    MerlinKVConfig include nothing for parameter default satisfied.

    Refer to the Merlin-KV TableOptions:
        struct HashTableOptions {
          size_t init_capacity = 0;        ///< The initial capacity of the hash table.
          size_t max_capacity = 0;         ///< The maximum capacity of the hash table.
          size_t max_hbm_for_vectors = 0;  ///< The maximum HBM for vectors, in bytes.
          size_t max_bucket_size = 128;    ///< The length of each bucket.
          float max_load_factor = 0.5f;    ///< The max load factor before rehashing.
          int device_id = 0;               ///< The ID of device.
          EvictStrategy evict_strategy = EvictStrategy::kLru;  ///< The evict strategy.
        };
    """
    self.init_capacity = init_capacity
    self.max_capacity = max_capacity
    self.max_hbm_for_vectors = max_hbm_for_vectors
    self.max_bucket_size = max_bucket_size
    self.max_load_factor = max_load_factor
    self.device_id = device_id
    self.evict_strategy = evict_strategy


class MerlinKVCreator(KVCreator):

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=None,
      config=None,
  ):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.default_value = default_value
    self.name = name
    self.checkpoint = checkpoint
    self.init_size = init_size
    if config is not None:
      self.config = config
    else:
      self.config = self.config if self.config else MerlinKVConfig(
          init_capacity=init_size)

    return mkv.MerlinKV(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=default_value,
        name=name,
        checkpoint=checkpoint,
        init_size=self.config.init_capacity,
        config=self.config,
    )

  def get_config(self):
    if not context.executing_eagerly():
      raise RuntimeError(
          'Unsupported to serialize python object of MerlinKVCreator.')

    config = {
        'key_dtype': self.key_dtype,
        'value_dtype': self.value_dtype,
        'default_value': self.default_value.numpy(),
        'name': self.name,
        'checkpoint': self.checkpoint,
        'init_size': self.init_size,
        'config': self.config,
    }
    return config
