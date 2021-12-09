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

"""Solving linear problems."""

from typing import Any

import jax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import lagrangian_form as lag_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils

InnerVerifInstance = verify_utils.InnerVerifInstance


class LpStrategy(dual_build.InnerMaxStrategy):
  """Solves inner maximisations (for linear Lagrangian) in closed form."""

  def supports_stochastic_parameters(self):
    # can use expectations of parameters instead of deterministic parameters
    return True

  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.array,
      step: int,
  ) -> jnp.array:
    """Solve maximization problem of opt_instance in closed form.

    Args:
      inner_dual_vars: Dual variables for the inner maximisation.
      opt_instance: Verification instance that defines optimization problem to
        be solved.
      key: Jax PRNG key.
      step: outer optimization iteration number

    Returns:
      max_value: final value of the objective function found.
    """
    if opt_instance.affine_before_relu:
      raise ValueError('LPStratgey requires affine_before_relu to be False.')

    if not opt_instance.same_lagrangian_form_pre_post:
      raise ValueError('Different lagrangian forms on inputs and outputs not'
                       'supported')

    if (isinstance(opt_instance.lagrangian_form_pre, lag_form.Linear) or
        isinstance(opt_instance.lagrangian_form_post, lag_form.Linear)):
      pass
    else:
      raise ValueError('LpStrategy cannot use Lagrangian form of type '
                       f'{type(opt_instance.lagrangian_form_pre)}.')

    # some renaming to simplify variable names
    affine_fn, = opt_instance.affine_fns
    bounds = opt_instance.bounds
    duals_pre = opt_instance.lagrange_params_pre
    if (opt_instance.is_last and
        opt_instance.spec_type == verify_utils.SpecType.ADVERSARIAL):
      # No duals_post for last layer, and objective folded in.
      batch_size = bounds[0].lb.shape[0]
      duals_post = jnp.ones([batch_size])
    else:
      duals_post = opt_instance.lagrange_params_post

    if opt_instance.is_first:
      # no "pre-activation" for input of first layer
      lb = bounds[0].lb
      ub = bounds[0].ub
    else:
      lb = bounds[0].lb_pre
      ub = bounds[0].ub_pre

    zero_inputs = jnp.zeros_like(lb)
    affine_constant = affine_fn(zero_inputs)
    duals_post = jnp.reshape(duals_post, affine_constant.shape)

    post_slope_x = jax.grad(lambda x: jnp.sum(affine_fn(x) * duals_post))(
        zero_inputs)

    if opt_instance.is_first:
      # find max element-wise (separable problem): either at lower bound or
      # upper bound -- no duals_pre for first layer
      max_per_element = jnp.maximum(
          post_slope_x * lb,
          post_slope_x * ub,
      )
    else:
      # find max element-wise (separable problem): either at lower bound, 0 or
      # upper bound
      duals_pre = jnp.reshape(duals_pre, lb.shape)
      max_per_element_bounds = jnp.maximum(
          post_slope_x * jax.nn.relu(lb) - duals_pre * lb,
          post_slope_x * jax.nn.relu(ub) - duals_pre * ub
      )
      max_per_element = jnp.where(
          jnp.logical_and(lb <= 0, ub >= 0),
          jax.nn.relu(max_per_element_bounds),  # include zero where feasible
          max_per_element_bounds)  # otherwise only at boundaries
    # sum over coordinates and add constant term (does not change max choice)
    max_value = jnp.sum(max_per_element,
                        axis=tuple(range(1, max_per_element.ndim)))
    constant_per_element = affine_constant * duals_post
    constant = jnp.sum(constant_per_element,
                       axis=tuple(range(1, constant_per_element.ndim)))
    return max_value + constant

  def init_layer_inner_params(self, opt_instance):
    """Returns initial inner maximisation duals and their types."""
    # no need for auxiliary variables
    return None, sdp_utils.DualVarTypes.EQUALITY
