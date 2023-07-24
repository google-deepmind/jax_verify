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

"""Projected gradient ascent."""

import dataclasses
from typing import Any, Text

import jax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import lagrangian_form as lag_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.functional_lagrangian.inner_solvers.pga import optimizer as optimizer_module
from jax_verify.extensions.functional_lagrangian.inner_solvers.pga import square
from jax_verify.extensions.functional_lagrangian.inner_solvers.pga import utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import numpy as np

InnerVerifInstance = verify_utils.InnerVerifInstance
Params = verify_utils.Params
ParamsTypes = verify_utils.ParamsTypes
LagrangianForm = lag_form.LagrangianForm


class PgaStrategy(dual_build.InnerMaxStrategy):
  """Solves inner maximisations with projected gradient ascent."""

  def __init__(self,
               n_iter: int,
               lr: float = 1.,
               n_restarts: int = 1.,
               method: Text = 'pgd',
               finetune_n_iter: int = 0,
               finetune_lr: float = 1.,
               finetune_method: Text = 'pgd',
               normalize: bool = False):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      n_iter: number of iterations of PGA to be performed.
      lr: learning-rate (or multiplier when adative, is kept constant).
      n_restarts: number of restarts.
      method: 'pgd', 'autopgd' or 'square'.
      finetune_n_iter: number of iterations of PGA to be performed after the
        initial optimization.
      finetune_lr: learning-rate when finetuning.
      finetune_method: 'pgd', 'autopgd'.
      normalize: whether to normalise inputs before PGA.
    """
    self._n_iter = n_iter
    self._lr = lr
    self._n_restarts = n_restarts
    self._method = method
    self._finetune_n_iter = finetune_n_iter
    self._finetune_lr = finetune_lr
    self._finetune_method = finetune_method
    self._normalize = normalize

  def _build_optimizer(self, method, n_iter, lr, lower_bound, upper_bound):
    epsilon = jnp.max(upper_bound - lower_bound) / 2

    if method == 'square':
      init_fn = utils.bounded_initialize_fn(bounds=(lower_bound, upper_bound))
      return square.Square(
          num_steps=n_iter,
          epsilon=epsilon,
          bounds=(lower_bound, upper_bound),
          initialize_fn=init_fn)
    elif method == 'pgd':
      init_fn = utils.noop_initialize_fn()
      project_fn = utils.linf_project_fn(
          epsilon=epsilon, bounds=(lower_bound, upper_bound))
      optimizer = optimizer_module.IteratedFGSM(lr)
      return optimizer_module.PGD(optimizer, n_iter, init_fn, project_fn)

    else:
      raise ValueError(f'Unknown method: "{method}"')

  def supports_stochastic_parameters(self):
    # This solver can be used with stochastic parameters (it will use the mean
    # and treat the problem as a deterministic one).
    return True

  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.ndarray,
      step: int,
  ) -> jnp.ndarray:
    """Solve maximization problem of opt_instance with projected gradient ascent.

    Args:
      inner_dual_vars: Dual variables for the inner maximisation.
      opt_instance: Verification instance that defines optimization problem to
        be solved.
      key: Jax PRNG key.
      step: outer optimization iteration number.

    Returns:
      final_value: final value of the objective function found by PGA.
    """
    if not opt_instance.same_lagrangian_form_pre_post:
      raise ValueError('Different lagrangian forms on inputs and outputs not'
                       'supported')
    # only supporting adversarial robustness specification for now
    # when affine_before_relu and logits layer.
    affine_before_relu = opt_instance.affine_before_relu

    assert not (opt_instance.spec_type == verify_utils.SpecType.UNCERTAINTY and
                opt_instance.is_last and affine_before_relu)

    # some renaming to simplify variable names
    if affine_before_relu:
      lower_bound = opt_instance.bounds[0].lb
      upper_bound = opt_instance.bounds[0].ub
    else:
      lower_bound = opt_instance.bounds[0].lb_pre
      upper_bound = opt_instance.bounds[0].ub_pre
    assert lower_bound.shape[0] == 1, 'Batching across samples not supported'

    if self._normalize:
      center = .5 * (upper_bound + lower_bound)
      radius = .5 * (upper_bound - lower_bound)
      normalize_fn = lambda x: x * radius + center
      lower_bound = -jnp.ones_like(lower_bound)
      upper_bound = jnp.ones_like(lower_bound)
    else:
      normalize_fn = lambda x: x

    duals_pre = opt_instance.lagrange_params_pre
    duals_post = opt_instance.lagrange_params_post
    # dual variables never used for grad tracing
    duals_pre_nondiff = jax.lax.stop_gradient(duals_pre)
    duals_post_nondiff = jax.lax.stop_gradient(duals_post)

    # Define the loss function.
    if (opt_instance.spec_type == verify_utils.SpecType.UNCERTAINTY and
        opt_instance.is_last):
      # Last layer here isn't the final spec layer, treat like other layers
      new_opt_instance = dataclasses.replace(opt_instance, is_last=False)
    else:
      new_opt_instance = opt_instance

    softmax = (
        opt_instance.spec_type == verify_utils.SpecType.ADVERSARIAL_SOFTMAX and
        opt_instance.is_last)

    obj = self.build_spec(new_opt_instance, step, softmax)

    def loss_pgd(x):
      # Expects x without batch dimension, as vmap adds batch-dimension.
      x = jnp.reshape(x, lower_bound.shape)
      x = normalize_fn(x)
      v = obj(x, duals_pre_nondiff, duals_post_nondiff)
      return -v

    loss_pgd = jax.vmap(loss_pgd)

    # Compute shape for compatibility with blackbox 'square' attack.
    if jnp.ndim(lower_bound) == 2 and self._method == 'square':
      d = lower_bound.shape[1]
      max_h = int(np.round(np.sqrt(d)))
      for h in range(max_h, 0, -1):
        w, ragged = divmod(d, h)
        if ragged == 0:
          break
      assert d == h * w
      shape = [1, h, w, 1]
    else:
      shape = lower_bound.shape
    # Optimization.
    init_x = (upper_bound + lower_bound) / 2
    init_x = jnp.reshape(init_x, shape)
    optimizer = self._build_optimizer(self._method, self._n_iter, self._lr,
                                      jnp.reshape(lower_bound, shape),
                                      jnp.reshape(upper_bound, shape))
    if self._n_restarts > 1:
      optimizer = optimizer_module.Restarted(
          optimizer, restarts_using_tiling=self._n_restarts)
    key, next_key = jax.random.split(key)
    x = optimizer(loss_pgd, key, init_x)

    if self._finetune_n_iter > 0:
      optimizer = self._build_optimizer(self._finetune_method,
                                        self._finetune_n_iter,
                                        self._finetune_lr,
                                        jnp.reshape(lower_bound, shape),
                                        jnp.reshape(upper_bound, shape))
    x = optimizer(loss_pgd, next_key, x)

    # compute final value and return it
    x = normalize_fn(jnp.reshape(x, lower_bound.shape))
    final_value = obj(
        jax.lax.stop_gradient(x),  # non-differentiable
        duals_pre,
        duals_post  # differentiable
    )
    return final_value

  def init_layer_inner_params(self, opt_instance):
    """Returns initial inner maximisation duals and their types."""
    # pga does not require extra variables
    return None, sdp_utils.DualVarTypes.EQUALITY
