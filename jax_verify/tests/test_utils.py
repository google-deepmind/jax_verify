# coding=utf-8
# Copyright 2021 The jax_verify Authors.
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

"""Utils functions for writing jax_verify tests."""
from typing import Tuple

import jax
import jax.numpy as jnp


def sample_bounds(key: jnp.ndarray,
                  shape: Tuple[int, ...],
                  minval: float = -2.,
                  maxval: float = 2.) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sample some bounds of the required shape.

  Args:
    key: Random number generator.
    shape: Shape of the bounds to generate.
    minval: Optional, smallest value that the bounds could take.
    maxval: Optional, largest value that the bounds could take.
  Returns:
    lb, ub: Lower and upper bound tensors of the desired shape.
  """
  key_0, key_1 = jax.random.split(key)
  bound_1 = jax.random.uniform(key_0, shape, minval=minval, maxval=maxval)
  bound_2 = jax.random.uniform(key_1, shape, minval=minval, maxval=maxval)
  lb = jnp.minimum(bound_1, bound_2)
  ub = jnp.maximum(bound_1, bound_2)
  return lb, ub


def sample_bounded_points(key: jnp.ndarray,
                          bounds: Tuple[jnp.ndarray, jnp.ndarray],
                          nb_points: int,
                          axis: int = 0) -> jnp.ndarray:
  """Sample uniformly some point respecting the bounds.

  Args:
    key: Random number generator
    bounds: Tuple containing [lower bound, upper bound]
    nb_points: How many points to sample.
    axis: Which dimension to add to correspond to the number of points.
  Returns:
    points: Points contained between the given bounds.
  """
  lb, ub = bounds
  act_shape = lb.shape
  to_sample_shape = act_shape[:axis] + (nb_points,) + act_shape[axis:]
  unif_samples = jax.random.uniform(key, to_sample_shape)

  broad_lb = jnp.expand_dims(lb, axis)
  broad_ub = jnp.expand_dims(ub, axis)

  bound_range = broad_ub - broad_lb
  return broad_lb + unif_samples * bound_range
