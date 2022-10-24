# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

# Lint as: python3
"""Library functions for SDP verification of neural networks."""
# pylint: disable=invalid-name
# Capital letters for matrices

import collections
import functools

from absl import logging
import jax
import jax.numpy as jnp
import jax.scipy
from jax_verify.extensions.sdp_verify import eigenvector_utils
from jax_verify.extensions.sdp_verify import utils
import numpy as np
import optax
import tree

IntBound = utils.IntBound
boundprop = utils.boundprop
flatten = lambda x: utils.flatten(x, backend=jnp)


def dual_fun(verif_instance, dual_vars, key=None, n_iter=30, scl=-1,
             exact=False, dynamic_unroll=True, include_info=False):
  # pylint: disable=invalid-name
  """Returns the dual objective value.

  Args:
    verif_instance: a utils.SdpDualVerifInstance, the verification problem
    dual_vars: A list of dual variables at each layer
    key: PRNGKey passed to Lanczos
    n_iter: Number of Lanczos iterations to use
    scl: Inverse temperature in softmax over eigenvalues to smooth optimization
        problem (if negative treat as hardmax)
    exact: Whether to use exact eigendecomposition instead of Lanczos
    dynamic_unroll: bool. Whether to use jax.fori_loop for Lanczos for faster
      JIT compilation. Default is False.
    include_info: if True, also return an `info` dict of various other
      values computed for the objective

  Returns:
    Either a single float, the dual upper bound, or if ``include_info=True``,
    returns a pair, the dual bound and a dict containing debugging info
  """
  key = key if key is not None else jax.random.PRNGKey(0)
  assert isinstance(verif_instance, utils.SdpDualVerifInstance)
  bounds = verif_instance.bounds
  layer_sizes = utils.layer_sizes_from_bounds(bounds)
  layer_sizes_1d = [np.prod(np.array(i), dtype=np.int32) for i in layer_sizes]
  N = sum(layer_sizes_1d) + 1
  info = {}

  # Mean activations at each layer
  activations_center = [(b.lb + b.ub) / 2 for b in bounds]
  # Maximum deviation from mean activations
  radius = [(b.ub - b.lb) / 2 for b in bounds]

  inner_lagrangian = verif_instance.make_inner_lagrangian(dual_vars)
  lagrangian = _make_transformed_lagrangian(
      inner_lagrangian, activations_center, radius)

  # Construct c_lambda and g_lambda terms
  zeros = [jnp.zeros(sz) for sz in layer_sizes]
  c_lambda = lagrangian(zeros)
  g_lambda = jax.grad(lagrangian)(zeros)
  g_lambda = flatten(g_lambda)
  info['c_lambda'] = c_lambda

  def Hv(v):
    """Hessian-vector product for H_lambda - refer to docstring for `Av()`."""
    lag_grad = lambda v2: flatten(jax.grad(lagrangian)(v2))
    hv_v = jax.grad(lambda v2: jnp.vdot(lag_grad(v2), v))(zeros)
    hv_flat = flatten(hv_v)
    return hv_flat

  def Av(v):
    """Matrix-vector product.

    Args:
      v: vector, DeviceArray

    Returns:
      Av: vector, Device array. A is defined as diag(kappa) - M(lambda) where
          M(lambda) = [0, g_lambda';
                       g_lambda, H_lambda], and these terms correspond to
          L~(z) = c_lambda + g_lambda' z + z' H_lambda z
    """
    # Expand Mv=[0 g'; g H] [v0;v1] = [g'v1; v0*g + H(v1)] = [Mv0;Mv1]
    # Compute Mv0 term
    mv_zero = jnp.reshape(jnp.vdot(g_lambda, v[1:]), (1,))
    # Compute Mv1 term
    mv_rest = Hv(v[1:]) + v[0] * g_lambda
    mv = jnp.concatenate([mv_zero, mv_rest], axis=0)
    diag_kappa_v = jnp.reshape(dual_vars[-1], mv.shape) * v
    av = diag_kappa_v - mv
    return jnp.reshape(av, v.shape)

  # Construct dual function (dual_vars[-1]=kappa)
  if exact:
    eig_vec, eig_info = eigenvector_utils.min_eigenvector_exact(
        Av, N, scl=scl, report_all=True)
    info['eig_info'] = eig_info
  else:
    eig_vec = eigenvector_utils.min_eigenvector_lanczos(
        Av, N, min(N, n_iter), key, scl, dynamic_unroll=dynamic_unroll)
  info['eig_vec'] = eig_vec
  info['kappa'] = dual_vars[-1]
  hess_val = jnp.vdot(eig_vec, Av(eig_vec))/(jnp.vdot(eig_vec, eig_vec))
  hess_val = jnp.reshape(hess_val, ())

  # Form dual objective
  lambda_minus = jnp.minimum(hess_val, 0.)
  kappa_hat = jnp.maximum(0, dual_vars[-1] - lambda_minus)
  dual_val = c_lambda + 0.5 * jnp.sum(kappa_hat)
  if include_info:
    return dual_val, info
  return dual_val


def _make_transformed_lagrangian(lagrangian, activations_center, radius):
  """Returns a function that computes transformed Lagrangian L~(z).

  Args:
    lagrangian: function L(x), the lagrangian with fixed dual variables.
    activations_center: list of Device arrays corresponding to mean activations
      by layer.
    radius: list of Device arrays corresponding to the interval bound radii by
      layer.

  Returns:
    transformed_lagrangian: a function L~(z), defined as
      L~(z) = L(activations_center + z * radius).

  For compatibility with the paper, the inner optimization should have [-1, 1]
  element-wise constraints, so here we re-express:
    max_{x: a-rad<=x<=a+rad} L(x), as
    max_{z: -1<=z<=1} L~(z), with L~ defined as above.
  """
  def transformed_lagrangian(zs):
    zs = [a + z * r for (a, z, r) in zip(activations_center, zs, radius)]
    return lagrangian(zs)

  return transformed_lagrangian


def project_duals(dual_vars, dual_types):
  """Projects dual variables to satisfy dual constraints."""
  make_pos = lambda v: None if v is None else jnp.maximum(v, 0)
  _project = lambda v, t: make_pos(v) if t == DualVarTypes.INEQUALITY else v
  return jax.tree_map(_project, dual_vars, dual_types)


def solve_sdp_dual_simple(verif_instance, key=None, opt=None, num_steps=10000,
                          eval_every=1000, verbose=False,
                          use_exact_eig_eval=True, use_exact_eig_train=False,
                          n_iter_lanczos=100,
                          kappa_reg_weight=None, kappa_zero_after=None,
                          device_type=None):
  """Compute verified lower bound via dual of SDP relaxation.

  Args:
    verif_instance: a utils.SdpDualVerifInstance
    key: jax.random.PRNGKey, used for Lanczos
    opt: an optax.GradientTransformation instance, the optimizer.
      If None, defaults to Adam with learning rate 1e-3.
    num_steps: int, the number of outer loop optimization steps
    eval_every: int, frequency of running evaluation step
    verbose: bool, enables verbose logging
    use_exact_eig_eval: bool, whether to use exact eigendecomposition instead of
      Lanczos when computing evaluation loss
    use_exact_eig_train: bool, whether to use exact eigendecomposition instead
      of Lanczos during training
    n_iter_lanczos: int, number of Lanczos iterations
    kappa_reg_weight: float, adds a penalty of sum(abs(kappa_{1:N})) to loss,
      which regularizes kappa_{1:N} towards zero. Default None is disabled.
    kappa_zero_after: int, clamps kappa_{1:N} to zero after ``kappa_zero_after``
      steps. Default None is disabled.
    device_type: string, used to clamp to a particular hardware device. Default
      None uses JAX default device placement

  Returns:
    A pair. The first element is a float, the final dual loss, which forms a
    valid upper bound on the objective specified by ``verif_instance``. The
    second element is a dict containing various debug info.
  """
  assert device_type in (None, 'cpu', 'gpu'), 'invalid device_type'
  assert isinstance(verif_instance, utils.SdpDualVerifInstance), 'invalid type'

  key = key if key is not None else jax.random.PRNGKey(0)
  opt = opt if opt is not None else optax.adam(1e3)
  dual_vars = jax.tree_map(
      lambda s: None if s is None else jnp.zeros(s), verif_instance.dual_shapes)
  dual_vars = init_duals_ibp(verif_instance, dual_vars)

  # Define loss function
  def loss(dual_vars, exact=use_exact_eig_train):
    return _loss(dual_vars, exact)

  @functools.partial(jax.jit, static_argnums=(1,), backend=device_type)
  def _loss(dual_var, exact):
    loss_val, step_info = dual_fun(
        verif_instance, dual_var, key, n_iter=n_iter_lanczos, exact=exact,
        include_info=True)
    step_info['loss_val'] = loss_val
    return loss_val, step_info

  # Define a compiled update step
  grad = jax.jit(jax.grad(loss, has_aux=True), backend=device_type)

  @functools.partial(jax.jit, backend=device_type)
  def grad_step(params, opt_state):
    g, info = grad(params)
    updates, new_opt_state = opt.update(g, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, info

  # Optimize parameters in a loop
  opt_state = opt.init(dual_vars)
  info = collections.defaultdict(list)
  loss_log = []
  best_loss = 1e9

  # Main loop
  for i in range(num_steps):
    dual_vars, opt_state, step_info = grad_step(dual_vars, opt_state)
    loss_val = step_info['loss_val']
    print(f'Iter {i}: Loss {loss_val}')
    best_loss = min(best_loss, loss_val)
    loss_log.append(loss_val)

    # Regularization of kappa
    if kappa_reg_weight is not None and kappa_reg_weight >= 0:
      onehot = jax.nn.one_hot([0], dual_vars[-1].shape[1])
      mask = jnp.ones_like(onehot) - onehot
      dual_vars[-1] -= mask * kappa_reg_weight
    if (kappa_zero_after is not None and kappa_zero_after >= 0 and
        i > kappa_zero_after):
      onehot = jax.nn.one_hot([0], dual_vars[-1].shape[1])
      dual_vars[-1] *= onehot

    dual_vars = project_duals(dual_vars, verif_instance.dual_types)

    if i % eval_every == 0:
      dual_val, _ = loss(dual_vars, exact=use_exact_eig_eval)
      info['steps'].append(i)
      info['loss_vals'].append(float(dual_val))
      if verbose:
        print(f'Dual iter {i}: Train loss: {loss_val} Loss {dual_val}')

  final_loss = float(loss(dual_vars, exact=use_exact_eig_eval)[0])
  info['final_dual_vars'] = dual_vars
  info['final_loss'] = final_loss
  info['loss_log'] = loss_log
  info['best_train_loss'] = best_loss
  return final_loss, info


def solve_sdp_dual(verif_instance, key=None, opt=None, num_steps=10000,
                   verbose=False, eval_every=1000, use_exact_eig_eval=True,
                   use_exact_eig_train=False, n_iter_lanczos=30, scl=-1.0,
                   lr_init=1e-3, steps_per_anneal=100, anneal_factor=1.0,
                   num_anneals=3, opt_name='adam', gd_momentum=0.9,
                   add_diagnostic_stats=False,
                   opt_multiplier_fn=None, init_dual_vars=None,
                   init_opt_state=None, opt_dual_vars=None,
                   kappa_reg_weight=None, kappa_zero_after=None,
                   device_type=None, save_best_k=1, include_opt_state=False):
  # pylint: disable=g-doc-return-or-yield, g-doc-args
  """Compute verified lower bound via dual of SDP relaxation.

  NOTE: This method exposes many hyperparameter options, and the method
  signature is subject to change. We instead suggest using
  ``solve_sdp_dual_simple`` instead if you need a stable interface.
  """
  # NB: Whereas the rest of the code in this library is fairly top-down
  # readable, avoids excessive `if` statements, tries to make the code look
  # like the formalism, etc, this is not the case for this method.
  # This is essentially the outer loop, and includes all the debugging/logging/
  # optimization tricks we need to get/debug good results.
  #
  # NB: Time profiling: On toy VerifInstances, JIT compilation dominates time
  # cost: JIT compilation takes ~12s, then we do ~3000 steps/sec.
  assert device_type in (None, 'cpu', 'gpu'), 'invalid device_type'
  assert isinstance(verif_instance, utils.SdpDualVerifInstance), 'invalid type'
  key = key if key is not None else jax.random.PRNGKey(0)

  dual_vars = jax.tree_map(
      lambda s: None if s is None else jnp.zeros(s), verif_instance.dual_shapes)
  dual_vars = init_duals_ibp(verif_instance, dual_vars)

  if init_dual_vars is not None:
    # Casting, here for Colab. Essentially same as `dual_vars = init_dual_vars`
    dual_vars = utils.structure_like(init_dual_vars, dual_vars)
  if opt_dual_vars is not None:
    opt_dual_vars = utils.structure_like(opt_dual_vars, dual_vars)

  # Create optimizer
  if opt is None:
    if (isinstance(steps_per_anneal, float) or
        isinstance(steps_per_anneal, int)):
      anneal_steps = [steps_per_anneal*(i+1) for i in range(num_anneals)]
    else:
      anneal_steps = np.cumsum(steps_per_anneal)
    anneal_steps = jnp.array(anneal_steps)
    def lr_schedule(t):
      cur_epoch = jnp.minimum(num_anneals, jnp.sum(t > anneal_steps))
      return lr_init * jnp.float_power(anneal_factor, cur_epoch)
    opt_class = getattr(optax, opt_name)
    base_opt = (opt_class(1., momentum=gd_momentum) if opt_name == 'sgd' else
                opt_class(1.))
    opt = optax.chain(base_opt, optax.scale_by_schedule(lr_schedule))
    if opt_multiplier_fn:
      # NB: Interface very specific to tree.map_structure_with_path
      # Example: opt_multiplier_fn=lambda path: 0.1 if 'lam' in path else 1.0
      opt_multipliers = tree.map_structure_with_path(
          lambda path, v: opt_multiplier_fn(path), dual_vars)
      opt = optax.chain(base_opt, optax.scale_by_schedule(lr_schedule),
                        utils.scale_by_variable_opt(opt_multipliers))
    else:
      opt = optax.chain(base_opt, optax.scale_by_schedule(lr_schedule))

  # Define loss function
  def loss(dual_vars, loss_scl=scl, exact=use_exact_eig_train):
    return _loss(dual_vars, loss_scl, exact)

  @functools.partial(jax.jit, static_argnums=(1, 2), backend=device_type)
  def _loss(dual_var, loss_scl, exact):
    loss_val, step_info = dual_fun(
        verif_instance, dual_var, key, n_iter=n_iter_lanczos, exact=exact,
        scl=loss_scl, include_info=True)
    step_info['loss_val'] = loss_val
    return loss_val, step_info

  # Define a compiled update step
  grad = jax.jit(jax.grad(loss, has_aux=True), backend=device_type)

  @functools.partial(jax.jit, backend=device_type)
  def grad_step(params, opt_state):
    g, info = grad(params)
    updates, new_opt_state = opt.update(g, opt_state)
    new_params = optax.apply_updates(params, updates)
    info['g'] = g
    info['updates'] = updates
    return new_params, new_opt_state, info

  # Optimize parameters in a loop
  opt_state = opt.init(dual_vars)
  if init_opt_state:
    opt_state = utils.structure_like(init_opt_state, opt_state)
  info = collections.defaultdict(list)
  loss_log = []
  store_best = []
  recent_eig_vecs = collections.deque(maxlen=10)
  best_loss = 1e9
  last_H = None
  start_i = 0

  # Main loop
  for i in range(start_i, num_steps):
    dual_vars_prev = dual_vars
    dual_vars, opt_state, step_info = grad_step(dual_vars, opt_state)
    loss_val = step_info['loss_val']
    print(f'Iter {i}: Loss {loss_val}')
    best_loss = min(best_loss, loss_val)
    if add_diagnostic_stats:
      info['dual_vars'].append(dual_vars_prev)
      eig_vec = step_info['eig_vec']
      cosine_sims = []
      for prev_eig_vec in recent_eig_vecs:
        denom = jnp.sqrt(jnp.linalg.norm(eig_vec)*jnp.linalg.norm(prev_eig_vec))
        eig_sim = jnp.sum(prev_eig_vec * eig_vec) / denom
        cosine_sims.append(abs(float(eig_sim)))
      info['c_lambda'].append(float(step_info['c_lambda']))
      info['past_10_cosine_sims'].append(np.array(cosine_sims))
      info['g'].append(step_info['g'])
      info['updates'].append(step_info['updates'])
      if use_exact_eig_train:
        # The info is for -H, so to get smallest for H, take negative of max
        eig_vals = -step_info['eig_info'][0][-1:-20:-1]
        cur_H = step_info['eig_info'][2]
        diff_H = 0 if last_H is None else np.linalg.norm(cur_H - last_H)
        last_H = cur_H
        info['diff_H'].append(float(diff_H))
        info['smallest_20_eig_vals'].append(eig_vals)
      recent_eig_vecs.appendleft(eig_vec)

    loss_log.append(loss_val)
    if len(store_best) < save_best_k:
      store_best.append((loss_val, dual_vars_prev))
      store_best.sort(key=lambda x: x[0])
    elif loss_val < store_best[-1][0]:
      store_best[-1] = (loss_val, dual_vars_prev)
      store_best.sort(key=lambda x: x[0])

    # Regularization of kappa
    if kappa_reg_weight is not None and kappa_reg_weight >= 0:
      onehot = jax.nn.one_hot([0], dual_vars[-1].shape[1])
      mask = jnp.ones_like(onehot) - onehot
      dual_vars[-1] -= mask * kappa_reg_weight
    if (kappa_zero_after is not None and kappa_zero_after >= 0 and
        i > kappa_zero_after):
      onehot = jax.nn.one_hot([0], dual_vars[-1].shape[1])
      dual_vars[-1] *= onehot

    dual_vars = project_duals(dual_vars, verif_instance.dual_types)

    if opt_dual_vars:
      distance_to_opt = jax.tree_map(lambda x, y: jnp.linalg.norm(x - y),
                                          dual_vars, opt_dual_vars)
      info['distance_to_opt'].append(distance_to_opt)

    if i % eval_every == 0:
      dual_val, _ = loss(dual_vars, loss_scl=-1, exact=use_exact_eig_eval)
      info['steps'].append(i)
      info['loss_vals'].append(float(dual_val))
      if verbose:
        print(f'Dual iter {i}: Train loss: {loss_val} Loss {dual_val}')

  final_loss = float(loss(dual_vars, loss_scl=-1, exact=use_exact_eig_eval)[0])
  info['final_dual_vars'] = dual_vars
  info['final_opt_state'] = opt_state
  info['final_loss'] = final_loss
  info['loss_log'] = loss_log
  info['store_best'] = store_best
  if include_opt_state:
    return final_loss, info, opt_state
  else:
    return final_loss, info

solve_dual_sdp_elided = solve_sdp_dual  # Alias

############     Dual initialization     ############

DualVarTypes = utils.DualVarTypes


def init_duals(verif_instance, key):
  """Initialize dual variables to zeros."""
  del key  # unused
  assert isinstance(verif_instance, utils.SdpDualVerifInstance)
  zeros_or_none = lambda s: None if s is None else jnp.zeros(s)
  return jax.tree_map(zeros_or_none, verif_instance.dual_shapes)


def _get_g_lambda(verif_instance, dual_vars):
  """Helper method for IBP initialization."""
  # NB: This code is (intentionally) copy-pasted from `dual_fun`, in order to
  # keep that method more top-to-bottom readable
  bounds = verif_instance.bounds
  # Mean activations at each layer
  activations_center = [(b.lb + b.ub) / 2 for b in bounds]
  # Maximum deviation from mean activations
  radius = [(b.ub - b.lb) / 2 for b in bounds]

  inner_lagrangian = verif_instance.make_inner_lagrangian(dual_vars)
  lagrangian = _make_transformed_lagrangian(
      inner_lagrangian, activations_center, radius)

  g_lambda = jax.grad(lagrangian)(activations_center)
  g_lambda = flatten(g_lambda)
  return jnp.reshape(g_lambda, (1, -1))


def init_duals_ibp(verif_instance, dual_vars):
  """Closed-form solution for dual variables which recovers IBP bound."""
  zero_duals = jax.tree_map(lambda x: x * 0., dual_vars)
  g_lambda = _get_g_lambda(verif_instance, zero_duals)
  kappa_opt_zero = jnp.reshape(jnp.sum(jnp.abs(g_lambda)), (1, 1))
  kappa_opt = jnp.concatenate([kappa_opt_zero, jnp.abs(g_lambda)], axis=1)
  ibp_duals = zero_duals[:-1] + [kappa_opt]
  return ibp_duals
