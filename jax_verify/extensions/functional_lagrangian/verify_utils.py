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

"""Small helper functions."""

import abc
import collections
import dataclasses
import enum
from typing import Callable, List, Optional, Union

import chex
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import lagrangian_form
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import ml_collections


Params = collections.namedtuple('Params', ['inner', 'outer'])
ParamsTypes = collections.namedtuple('ParamsTypes',
                                     ['inner', 'outer', 'lagrangian_form'])

DataSpec = collections.namedtuple(
    'DataSpec',
    ['input', 'true_label', 'target_label', 'epsilon', 'input_bounds'])

Array = chex.Array
ArrayTree = chex.ArrayTree
ConfigDict = ml_collections.ConfigDict
IntervalBound = jax_verify.IntervalBound
Tensor = jnp.array
LayerParams = Union['FCParams', 'ConvParams']
LagrangianForm = lagrangian_form.LagrangianForm
ModelParams = List[LayerParams]
ModelParamsElided = ModelParams


class AbstractParams(abc.ABC):
  """AbstractParams."""

  def __call__(self, inputs: Tensor) -> Tensor:
    """Forward pass on layer."""
    return sdp_utils.fwd(inputs, self.params)

  @property
  @abc.abstractmethod
  def params(self):
    """Representation of params with sdp_utils.fwd convention."""

  @property
  def has_bounds(self):
    return self.w_bound is not None or self.b_bound is not None  # pytype: disable=attribute-error  # bind-properties


@dataclasses.dataclass
class FCParams(AbstractParams):
  """Params of fully connected layer."""

  w: Tensor
  b: Tensor

  w_bound: Optional[IntervalBound] = None
  b_bound: Optional[IntervalBound] = None

  w_std: Optional[Tensor] = None
  b_std: Optional[Tensor] = None

  dropout_rate: float = 0.0

  @property
  def params(self):
    return (self.w, self.b)


@dataclasses.dataclass
class ConvParams(AbstractParams):
  """Params of convolutional layer."""

  w: Tensor
  b: Tensor

  stride: int
  padding: str

  n_cin: Optional[int] = None

  w_bound: Optional[IntervalBound] = None
  b_bound: Optional[IntervalBound] = None

  w_std: Optional[Tensor] = None
  b_std: Optional[Tensor] = None

  dropout_rate: float = 0.0

  @property
  def params(self):
    return {
        'W': self.w,
        'b': self.b,
        'n_cin': self.n_cin,
        'stride': self.stride,
        'padding': self.padding,
    }


class SpecType(enum.Enum):
  # `params` represent a network of repeated relu(Wx+b)
  # The final output also includes a relu activation, and `obj` composes
  # the final layer weights with the original objective
  UNCERTAINTY = 'uncertainty'
  ADVERSARIAL = 'adversarial'
  ADVERSARIAL_SOFTMAX = 'adversarial_softmax'
  PROBABILITY_THRESHOLD = 'probability_threshold'


class Distribution(enum.Enum):
  """Distribution of the weights and biases."""
  GAUSSIAN = 'gaussian'
  BERNOULLI = 'bernoulli'


class NetworkType(enum.Enum):
  """Distribution of the weights and biases."""
  DETERMINISTIC = 'deterministic'
  STOCHASTIC = 'stochastic'


@dataclasses.dataclass(frozen=True)
class InnerVerifInstance:
  """Specification of inner problems."""

  affine_fns: List[Callable[[Array], Array]]
  bounds: List[sdp_utils.IntervalBound]

  lagrangian_form_pre: Optional[LagrangianForm]
  lagrangian_form_post: Optional[LagrangianForm]

  lagrange_params_pre: Optional[ArrayTree]
  lagrange_params_post: Optional[ArrayTree]

  is_first: bool
  is_last: bool

  idx: int
  spec_type: SpecType
  affine_before_relu: bool

  @property
  def same_lagrangian_form_pre_post(self) -> bool:
    if self.is_first:
      return True
    elif self.is_last:
      return True
    else:
      name_pre = self.lagrangian_form_pre.name
      name_post = self.lagrangian_form_post.name
      return name_pre == name_post
