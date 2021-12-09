# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
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

"""Lagrangian penalty functions."""

import abc

from typing import Sequence, Union

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections

PRNGKey = jnp.array
Tensor = jnp.array
Params = Union[Tensor, Sequence[Tensor]]
Shape = Union[int, Sequence[int]]
ConfigDict = ml_collections.ConfigDict


def _flatten_spatial_dims(x: Tensor) -> Tensor:
  """Flatten spatial dimensions (assumed batched)."""
  return jnp.reshape(x, [x.shape[0], -1])


def size_from_shape(shape: Shape) -> int:
  return int(jnp.prod(jnp.array(shape)))


class LagrangianForm(metaclass=abc.ABCMeta):
  """Abstract class for Lagrangian form."""

  def __init__(self, name):
    self._name = name

  @abc.abstractmethod
  def _init_params_per_sample(self, key: PRNGKey, *args) -> Params:
    """Initialize the parameters of the Lagrangian form."""

  def init_params(self, key, *args, **kwargs):
    params = self._init_params_per_sample(key, *args, **kwargs)
    # expansion below currently assumes batch-size of 1
    return jax.tree_map(lambda p: jnp.expand_dims(p, 0), params)

  @abc.abstractmethod
  def _apply(self, x: Tensor, lagrange_params: Params, step: int) -> Tensor:
    """Apply the Lagrangian form the input x given lagrange_params."""

  def apply(self, x: Tensor, lagrange_params: Params, step: int) -> Tensor:
    """Apply the Lagrangian form the input x given lagrange_params.

    Args:
      x: layer inputs, assumed batched (in leading dimension). Note that the
        spatial dimensions of x are flattened.
      lagrange_params: parameters of the lagrangian parameters, assumed to have
        the same batch-size as x. If provided as None, this function returns 0.
      step: outer optimization iteration number (unused).

    Returns:
      value_per_sample: Lagrangian penalty per element of the mini-batch.
    """
    if lagrange_params is None:
      return jnp.array(0.0)
    x = _flatten_spatial_dims(x)
    value_per_sample = self._apply(x, lagrange_params, step)
    return value_per_sample

  def process_params(self, lagrange_params: Params):
    return lagrange_params

  @property
  def name(self):
    """Return name."""
    return self._name


class Linear(LagrangianForm):
  """Linear LagrangianForm (equivalent to DeepVerify formulation)."""

  def __init__(self):
    super().__init__('Linear')

  def _init_params_per_sample(self,
                              key: PRNGKey,
                              l_shape: Shape,
                              init_zeros: bool = True) -> Params:
    size = size_from_shape(l_shape)
    if init_zeros:
      return jnp.zeros([size])
    else:
      return random.normal(key, [size])

  def _apply_per_sample(self, x: Tensor, lagrange_params: Params,
                        step: int) -> Tensor:
    del step
    return jnp.dot(x, lagrange_params)

  def _apply(self, x: Tensor, lagrange_params: Params, step: int) -> Tensor:
    apply_per_sample = lambda a, b: self._apply_per_sample(a, b, step)
    return jax.vmap(apply_per_sample)(x, lagrange_params)


class LinearExp(LagrangianForm):
  """LinearExp LagrangianForm."""

  def __init__(self):
    super().__init__('LinearExp')

  def _init_params_per_sample(self,
                              key: PRNGKey,
                              l_shape: Shape,
                              init_zeros: bool = False) -> Params:
    size = size_from_shape(l_shape)
    if init_zeros:
      return jnp.zeros([size]), jnp.ones(()), jnp.zeros([size])
    else:
      return (1e-4 * random.normal(key, [size]), 1e-2 * random.normal(key, ()),
              1e-2 * random.normal(key, [size]))

  def _apply_per_sample(self, x: Tensor, lagrange_params: Params,
                        step: int) -> Tensor:
    del step
    linear_term = jnp.dot(x, lagrange_params[0])
    lagrange_params = self.process_params(lagrange_params)
    exp_term = lagrange_params[1] * jnp.exp(jnp.dot(x, lagrange_params[2]))

    return linear_term + exp_term

  def _apply(self, x: Tensor, lagrange_params: Params, step: int) -> Tensor:
    apply_per_sample = lambda a, b: self._apply_per_sample(a, b, step)
    return jax.vmap(apply_per_sample)(x, lagrange_params)


def get_lagrangian_form(config_lagrangian_form: ConfigDict) -> LagrangianForm:
  """Create the Lagrangian form."""
  name = config_lagrangian_form['name']
  kwargs = config_lagrangian_form['kwargs']
  if name == 'linear':
    return Linear(**kwargs)
  elif name == 'linear_exp':
    return LinearExp(**kwargs)
  else:
    raise NotImplementedError(f'Unrecognized lagrangian functional: {name}')
