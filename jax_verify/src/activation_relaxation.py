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

"""Define convex relaxations for primitives."""
import functools
from typing import Callable, Tuple

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import mccormick
from jax_verify.src import synthetic_primitives


Tensor = bound_propagation.Tensor
TensorFunction = Callable[..., Tensor]

def abs_relaxation(x_lb: Tensor, x_ub: Tensor
                   ) -> Tuple[TensorFunction, TensorFunction]:
  """Perform the relaxation of the absolute value operation.

  Args:
    x_lb: Lower bound on the absolute value input.
    x_ub: Upper bound on the absolute value input.
  Returns:
    lb_fun, ub_fun.
  """
  lb_fun = jnp.abs

  safe_denom = jnp.maximum(x_ub - x_lb, 1e-12)
  ub_slope = (jnp.abs(x_ub) - jnp.abs(x_lb)) / safe_denom
  ub_intercept = jnp.abs(x_lb) - ub_slope * x_lb

  ub_fun = lambda x: ub_slope * x + ub_intercept

  return lb_fun, ub_fun


def leaky_relu_relaxation(x_lb: Tensor, x_ub: Tensor,
                          negative_slope: float
                          ) -> Tuple[TensorFunction, TensorFunction]:
  """Perform the relaxation of the leaky ReLU.

  Depending on how negative_slope compares to 1.0, the leaky_relu function is
  going to either be convex or concave. The other function is going to be given
  by its chord.

  Args:
    x_lb: Lower bound on the leaky ReLU input
    x_ub: Upper bound on the leaky ReLU input
    negative_slope: Slope for negative inputs.
  Returns:
    lb_fun, ub_fun.
  """
  lr = lambda x: jax.nn.leaky_relu(x, negative_slope)
  chord_slope_safe_denom = jnp.maximum(x_ub - x_lb, 1e-12)
  chord_slope = (lr(x_ub) - lr(x_lb)) / chord_slope_safe_denom
  chord_intercept = lr(x_lb) - chord_slope * x_lb

  chord_fun = lambda x: chord_slope * x + chord_intercept
  if negative_slope > 1.:
    # The leaky ReLu is a concave function
    return chord_fun, lr
  else:
    # The leaky Relu is a convex function
    return lr, chord_fun
