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

"""Solve last layer inner max for probability specification."""

import enum
from typing import Any

import jax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils


InnerVerifInstance = verify_utils.InnerVerifInstance
Tensor = jnp.array


class LayerType(enum.Enum):
  # `params` represent a network of repeated relu(Wx+b)
  # The final output also includes a relu activation, and `obj` composes
  # the final layer weights with the original objective
  INPUT = 'input'
  FIRST = 'first'


class InputUncertaintySpecStrategy(dual_build.InnerMaxStrategy):
  """Strategy for solving inner max at final layer with uncertainty spec."""

  def __init__(
      self,
      layer_type: LayerType,
      sig_max: float,
  ):
    """Constructor.

    Args:
      layer_type: Indicates whether optimization is over input layer or first
        linear layer
      sig_max: Maximum standard deviation of input noise
    """
    self._layer_type = layer_type
    self._eps = 1e-10
    self._sig_max = sig_max
    return

  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.array,
      step: int,
  ) -> jnp.array:
    if self._layer_type == LayerType.INPUT:
      return self.solve_max_input(inner_dual_vars, opt_instance, key, step)
    elif self._layer_type == LayerType.FIRST:
      return self.solve_max_first(inner_dual_vars, opt_instance, key, step)
    else:
      raise ValueError('Unrecognized layer type in input uncertainty spec')

  def solve_max_input(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.array,
      step: int,
  ) -> jnp.array:
    assert opt_instance.is_first
    lb = opt_instance.bounds[0].lb
    ub = opt_instance.bounds[0].ub
    x = (ub + lb) / 2.
    pert = (ub - lb) / 2.
    affine_fn, = opt_instance.affine_fns
    lam, kappa, gamma = opt_instance.lagrangian_form_post.process_params(
        opt_instance.lagrange_params_post)
    zeros_pert = jnp.zeros_like(pert)
    wconst = affine_fn(x)
    gamma = jnp.reshape(gamma, wconst.shape)
    gamma_post = jax.grad(lambda x: jnp.sum(gamma * affine_fn(x)))(zeros_pert)
    var_term = (
        jnp.reshape(gamma_post, [-1]) * jnp.reshape(pert, [-1]) * self._sig_max)
    gam_dot_b = jnp.sum(jnp.reshape(gamma, [-1]) * jnp.reshape(wconst, [-1]))
    obj = jnp.exp(.5 * jnp.sum(jnp.square(var_term)) + gam_dot_b + kappa)
    obj += jnp.sum(jnp.reshape(lam, [-1]) * jnp.reshape(wconst, [-1]))
    return obj

  def _optt(self, kappa, theta, gamma, lb, ub):
    eps = self._eps

    def optt(t):
      return kappa * jnp.exp(t) + theta * t

    tmin = jnp.sum(jnp.minimum(gamma * lb, gamma * ub))
    tmax = jnp.sum(jnp.maximum(gamma * lb, gamma * ub))
    optt_c = jnp.where(
        jnp.abs(kappa) > eps, -theta / jnp.maximum(kappa, eps), jnp.exp(tmin))
    optt_c = jnp.where(optt_c > eps, jnp.log(jnp.maximum(optt_c, eps)), tmin)
    optt_c = jax.lax.stop_gradient(optt_c)
    best_obj_t = jnp.maximum(optt(tmin), optt(tmax))
    best_obj_t = jnp.maximum(best_obj_t, optt(optt_c))
    return best_obj_t

  def solve_max_first(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.array,
      step: int,
  ) -> jnp.array:
    theta = jnp.reshape(inner_dual_vars, [-1, 1])
    affine_fn, = opt_instance.affine_fns
    bounds = opt_instance.bounds
    duals_pre, kappa, gamma = opt_instance.lagrangian_form_pre.process_params(
        opt_instance.lagrange_params_pre)
    duals_post = opt_instance.lagrange_params_post
    lb = bounds[0].lb_pre
    ub = bounds[0].ub_pre
    zero_inputs = jnp.zeros_like(lb)
    affine_constant = affine_fn(zero_inputs)
    duals_post = jnp.reshape(duals_post, affine_constant.shape)
    post_slope_x = jax.grad(lambda x: jnp.sum(affine_fn(x) * duals_post))(
        zero_inputs)
    post_slope_x = jnp.reshape(post_slope_x, lb.shape)
    duals_pre = jnp.reshape(duals_pre, lb.shape)
    gamma = jnp.reshape(gamma, lb.shape)

    opt_c = jnp.clip(jnp.zeros_like(lb), lb, ub)

    def funx(x):
      return (post_slope_x * jax.nn.relu(x) -
              (duals_pre + jnp.exp(theta) * gamma) * x)

    best_obj = jnp.maximum(funx(lb), funx(ub))
    best_obj = jnp.sum(jnp.maximum(funx(opt_c), best_obj))
    best_obj += jnp.exp(theta) * (theta - 1 - kappa)
    best_obj += jnp.dot(
        jnp.reshape(affine_constant, [1, -1]), jnp.reshape(duals_post, [-1, 1]))
    return jnp.reshape(best_obj, (1,))

  def init_layer_inner_params(self, opt_instance):
    """Returns initial inner maximisation duals and their types."""
    return jnp.zeros(()), sdp_utils.DualVarTypes.EQUALITY


class ProbabilityThresholdSpecStrategy(dual_build.InnerMaxStrategy):
  """Strategy for solving inner max at final layer with uncertainty spec."""

  def __init__(self):
    """Constructor."""
    self._eps = 1e-10
    return

  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.array,
      step: int,
  ) -> jnp.array:
    assert opt_instance.is_last
    l = opt_instance.bounds[0].lb_pre
    u = opt_instance.bounds[0].ub_pre
    theta = jnp.reshape(inner_dual_vars, [-1, 1])

    def lagr_form(x):
      val = opt_instance.lagrangian_form_pre.apply(
          x, opt_instance.lagrange_params_pre, step)
      return jnp.reshape(val, ())

    lagr_form_const = lagr_form(jnp.zeros_like(l))
    lagr_form_grad = jax.grad(lagr_form)
    lagr_form_param_affine = lagr_form_grad(jnp.zeros_like(l))

    affine_obj = lambda x: jnp.reshape(opt_instance.affine_fns[0](x), ())
    assert len(opt_instance.affine_fns) == 1

    # Extract coefficients of affine_obj
    obj = jax.grad(affine_obj)(l)
    obj = jnp.reshape(obj, [-1, 1])
    l = jnp.reshape(l, [-1, 1])
    u = jnp.reshape(u, [-1, 1])

    # Extract bias term
    obj_bias = -jnp.reshape(lagr_form_const, [-1, 1])
    # Solve max_{l <= x <= u} Indicator[obj^T exp(x) >= 0] - lagr_form(x)
    obj_a = (
        obj_bias + 1 +
        self._optx(theta[0] * obj, -lagr_form_param_affine, l, u))
    obj_b = (
        obj_bias + self._optx(-theta[1] * obj, -lagr_form_param_affine, l, u))
    return jnp.reshape(jnp.maximum(obj_a, obj_b), (1,))

  def _optx(self, a, b, l, u):
    """Optimize a^T exp(x) + b^T x subject to l <= x <= u."""
    a = jnp.reshape(a, [-1, 1])
    b = jnp.reshape(b, [-1, 1])
    l = jnp.reshape(l, [-1, 1])
    u = jnp.reshape(u, [-1, 1])
    eps = self._eps
    opt_candidate = jnp.where(a > eps, -b / jnp.maximum(a, eps),
                              jnp.zeros_like(a))
    opt_candidate += jnp.where(a < -eps, -b / jnp.minimum(a, -eps),
                               jnp.zeros_like(a))
    opt_candidate = jnp.where(opt_candidate > eps,
                              jnp.log(jnp.maximum(opt_candidate, eps)), l)
    opt_candidate = jax.lax.stop_gradient(jnp.clip(opt_candidate, l, u))
    funx = lambda x: a * jnp.exp(x) + b * x
    best_obj = jnp.maximum(funx(l), funx(u))
    best_obj = jnp.maximum(best_obj, funx(opt_candidate))
    return jnp.sum(best_obj)

  def init_layer_inner_params(self, opt_instance):
    """Returns initial inner maximisation duals and their types."""
    return jnp.zeros((2,)), sdp_utils.DualVarTypes.INEQUALITY
