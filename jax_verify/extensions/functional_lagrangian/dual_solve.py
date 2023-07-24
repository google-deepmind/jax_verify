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

"""Solve dual."""

import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import bounding
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import inner_solvers
from jax_verify.extensions.functional_lagrangian import lagrangian_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import ml_collections
import optax


ConfigDict = ml_collections.ConfigDict
DualOp = dual_build.DualOp
InnerMaxStrategy = dual_build.InnerMaxStrategy
ModelParams = verify_utils.ModelParams
Params = verify_utils.Params
ParamsTypes = verify_utils.ParamsTypes
Tensor = jnp.array


def solve_dual(
    config: ConfigDict,
    bounds: Sequence[sdp_utils.IntBound],
    spec_type: verify_utils.SpecType,
    spec_fn: Callable[..., jnp.ndarray],
    params: ModelParams,
    dual_state: ConfigDict,
    mode: str,
    logger: Callable[[int, Mapping[str, Any]], None],
) -> Tuple[float, Tensor]:
  """Run verification algorithm and update dual_state."""
  key_carry, key_init, key_solve = jax.random.split(
      jax.random.PRNGKey(config.seed), 3)

  # define Lagrangian form per layer
  if isinstance(config.dual.lagrangian_form, list):
    lagrangian_form_per_layer = [
        lagrangian_form.get_lagrangian_form(x)
        for x in config.dual.lagrangian_form
    ]
  else:
    lagrangian_form_per_layer = [
        lagrangian_form.get_lagrangian_form(config.dual.lagrangian_form)
        for _ in bounds
    ]

  inner_opt = inner_solvers.get_strategy(config, params, mode)

  input_bounds = jax_verify.IntervalBound(bounds[0].lb, bounds[0].ub)
  boundprop_transform = bounding.BoundsFromCnn(bounds)
  env, dual_params, dual_params_types = inner_opt.init_duals(
      boundprop_transform, spec_type, config.dual.affine_before_relu, spec_fn,
      key_init, lagrangian_form_per_layer, input_bounds)

  device_type = ('gpu' if config.use_gpu else 'cpu')

  if mode == 'train':
    opt, num_steps = dual_build.make_opt_and_num_steps(config.outer_opt)

    dual_state = solve_dual_train(
        env,
        key=key_solve,
        num_steps=num_steps,
        opt=opt,
        dual_params=dual_params,
        dual_params_types=dual_params_types,
        affine_before_relu=config.dual.affine_before_relu,
        spec_type=spec_type,
        inner_opt=inner_opt,
        logger=logger,
        device_type=device_type,
        block_to_time=config.block_to_time,
        dual_state=dual_state,
    )
  elif mode == 'eval':

    dual_state.loss = solve_dual_eval(
        env,
        step=dual_state.step,
        key=key_solve,
        dual_params=dual_state.dual_params,
        dual_params_types=dual_params_types,
        affine_before_relu=config.dual.affine_before_relu,
        logger=logger,
        inner_opt=inner_opt,
        spec_type=spec_type,
    )
  else:
    raise ValueError(f'Invalid mode: {mode}.')

  return key_carry


def solve_dual_train(
    env: Dict[int, DualOp],
    dual_state: ConfigDict,
    opt: optax.GradientTransformation,
    inner_opt: InnerMaxStrategy,
    dual_params: Params,
    spec_type: verify_utils.SpecType,
    dual_params_types: ParamsTypes,
    logger: Callable[[int, Mapping[str, Any]], None],
    key: jnp.ndarray,
    num_steps: int,
    affine_before_relu: bool,
    device_type=None,
    merge_problems: Optional[Dict[int, int]] = None,
    block_to_time: bool = False,
) -> ConfigDict:
  """Compute verified upper bound via functional lagrangian relaxation.

  Args:
    env: Lagrangian computations for each contributing graph node.
    dual_state: state of the dual problem.
    opt: an optimizer for the outer Lagrangian parameters.
    inner_opt: inner optimization strategy for training.
    dual_params: dual parameters to be minimized via gradient-based
      optimization.
    spec_type: Specification type, adversarial or uncertainty specification.
    dual_params_types: types of inequality encoded by the corresponding
      dual_params.
    logger: logging function.
    key: jax.random.PRNGKey.
    num_steps: total number of outer optimization steps.
    affine_before_relu: whether layer ordering uses the affine layer before the
      ReLU.
    device_type: string, used to clamp to a particular hardware device. Default
      None uses JAX default device placement.
    merge_problems: the key of the dictionary corresponds to the index of the
      layer to begin the merge, and the associated value corresponds to the
      number of consecutive layers to be merged with it.
      For example, `{0: 2, 2: 3}` will merge together layer 0 and 1, as well as
        layers 2, 3 and 4.
    block_to_time: whether to block computations at the end of each iteration to
      account for asynchronicity dispatch when timing.

  Returns:
    dual_state: new state of the dual problem.
    info: various information for logging / debugging.
  """
  assert device_type in (None, 'cpu', 'gpu'), 'invalid device_type'

  # create dual functions
  loss_func = dual_build.build_dual_fun(
      env=env,
      lagrangian_form=dual_params_types.lagrangian_form,
      inner_opt=inner_opt,
      merge_problems=merge_problems,
      affine_before_relu=affine_before_relu,
      spec_type=spec_type)

  value_and_grad = jax.value_and_grad(loss_func, has_aux=True)

  def grad_step(params, opt_state, key, step):
    (loss_val, stats), g = value_and_grad(params, key, step)
    updates, new_opt_state = opt.update(g, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val, stats

  # Some solvers (e.g. MIP) cannot be jitted and run on CPU only
  if inner_opt.jittable:
    grad_step = jax.jit(grad_step, backend=device_type)

  dual_state.step = 0
  dual_state.key = key
  dual_state.opt_state = opt.init(dual_params)
  dual_state.dual_params = dual_params
  dual_state.loss = 0.0

  dual_state.best_loss = jnp.inf
  dual_state.best_dual_params = dual_params

  # optimize the dual (Lagrange) parameters with a gradient-based optimizer
  while dual_state.step < num_steps:
    key_step, dual_state.key = jax.random.split(dual_state.key)
    start_time = time.time()
    dual_params, dual_state.opt_state, dual_state.loss, stats = grad_step(
        dual_state.dual_params, dual_state.opt_state, key_step, dual_state.step)
    dual_params = dual_build.project_dual(dual_params, dual_params_types)
    if dual_state.loss <= dual_state.best_loss:
      dual_state.best_loss = dual_state.loss
      # store value from previous iteration as loss corresponds to those params
      dual_state.best_dual_params = dual_state.dual_params
    dual_state.dual_params = dual_params  # projected dual params
    if block_to_time:
      dual_state.loss.block_until_ready()  # asynchronous dispatch
    stats['time_per_iteration'] = time.time() - start_time
    stats['best_loss'] = dual_state.best_loss
    stats['dual_params_norm'] = optax.global_norm(dual_state.dual_params)

    logger(dual_state.step, stats)

    dual_state.step += 1

  return dual_state


def solve_dual_eval(
    env: Dict[int, DualOp],
    inner_opt: InnerMaxStrategy,
    dual_params: Params,
    spec_type: verify_utils.SpecType,
    dual_params_types: ParamsTypes,
    logger: Callable[[int, Mapping[str, Any]], None],
    key: jnp.ndarray,
    affine_before_relu: bool,
    step: int,
    merge_problems: Optional[Dict[int, int]] = None,
) -> float:
  """Compute verified upper bound via functional lagrangian relaxation.

  Args:
    env: Lagrangian computations for each contributing graph node.
    inner_opt: inner optimization strategy for evaluation.
    dual_params: dual parameters to be minimized via gradient-based
      optimization.
    spec_type: Specification type, adversarial or uncertainty specification.
    dual_params_types: types of inequality encoded by the corresponding
      dual_params.
    logger: logging function.
    key: jax.random.PRNGKey.
    affine_before_relu: whether layer ordering uses the affine layer before the
      ReLU.
    step: outer training iteration number, the functional may depend on this.
    merge_problems: the key of the dictionary corresponds to the index of the
      layer to begin the merge, and the associated value corresponds to the
      number of consecutive layers to be merged with it.
      For example, `{0: 2, 2: 3}` will merge together layer 0 and 1, as well as
        layers 2, 3 and 4.

  Returns:
    final_loss: final dual loss, which forms a valid upper bound on the
      objective specified by ``verif_instance``.
  """

  # create dual functions
  loss_func = dual_build.build_dual_fun(
      env=env,
      lagrangian_form=dual_params_types.lagrangian_form,
      inner_opt=inner_opt,
      merge_problems=merge_problems,
      affine_before_relu=affine_before_relu,
      spec_type=spec_type)

  start_time = time.time()
  final_loss, stats = loss_func(dual_params, key, step)
  final_loss.block_until_ready()  # accounting for asynchronous dispatch
  stats['time_per_iteration'] = time.time() - start_time

  logger(0, stats)

  return float(final_loss)
