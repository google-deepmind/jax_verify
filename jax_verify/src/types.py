# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

"""Common type definitions used by jax_verify."""

from typing import Any, Generic, Mapping, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import typing_extensions


Primitive = jax.core.Primitive
Tensor = jnp.ndarray
T = TypeVar('T')
U = TypeVar('U')
Nest = Union[T, Sequence['Nest[T]'], Mapping[Any, 'Nest[T]']]
Index = Tuple[int, ...]


class TensorFun(typing_extensions.Protocol):

  def __call__(self, *inputs: Tensor) -> Tensor:
    pass


class SpecFn(typing_extensions.Protocol):
  """Specification, expressed as all outputs are <=0."""

  def __call__(self, *inputs: Nest[Tensor]) -> Nest[Tensor]:
    pass


class ArgsKwargsCallable(typing_extensions.Protocol, Generic[T, U]):

  def __call__(self, *args: T, **kwargs) -> U:
    pass
