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

"""Utilities."""

from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

InitializeFn = Callable[[chex.Array, chex.Array], chex.Array]
ProjectFn = Callable[[chex.Array, chex.Array], chex.Array]
LossFn = Callable[[chex.Array], chex.Array]


def linf_project_fn(epsilon: float, bounds: Tuple[float, float]) -> ProjectFn:

  def project_fn(x, origin_x):
    dx = jnp.clip(x - origin_x, -epsilon, epsilon)
    return jnp.clip(origin_x + dx, bounds[0], bounds[1])

  return project_fn


def bounded_initialize_fn(
    bounds: Optional[Tuple[chex.Array, chex.Array]] = None,) -> InitializeFn:
  """Returns an initialization function."""
  if bounds is None:
    return noop_initialize_fn()
  else:
    lower_bound, upper_bound = bounds

    def _initialize_fn(rng, x):
      a = jax.random.uniform(rng, x.shape, minval=0., maxval=1.)
      x = a * lower_bound + (1. - a) * upper_bound
      return x

    return _initialize_fn


def noop_initialize_fn() -> InitializeFn:

  def _initialize_fn(rng, x):
    del rng
    return x

  return _initialize_fn
