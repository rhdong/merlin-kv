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
"""Export TensorFlow APIs."""

__all__ = [
    'MerlinKV',
    'MerlinKVConfig',
    'MerlinKVCreator',
    'Variable',
    'get_variable',
]

from merlin_kv.tensorflow.python.ops.kv_creator import (
    KVCreator,
    MerlinKVConfig,
    MerlinKVCreator,
)
from merlin_kv.tensorflow.python.ops.merlin_kv_ops import (
    MerlinKV,)
from merlin_kv.tensorflow.python.ops.merlin_kv_variable import (
    Variable,)
from merlin_kv.tensorflow.python.ops.merlin_kv_variable import (
    get_variable,)