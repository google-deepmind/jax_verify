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

"""Solve last layer inner max for probability specification."""

import enum
import itertools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import lagrangian_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.functional_lagrangian.inner_solvers import exact_opt_softmax
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import numpy as np
import optax
import scipy


InnerVerifInstance = verify_utils.InnerVerifInstance
Tensor = jnp.array


class MaxType(enum.Enum):
  EXP = 'exp'
  EXP_BOUND = 'exp_bound'


class UncertaintySpecStrategy(dual_build.InnerMaxStrategy):
  """Strategy for solving inner max at final layer with uncertainty spec."""

  def __init__(
      self,
      n_iter: int,
      n_pieces: int,
      solve_max: MaxType,
      learning_rate: float = 1.0,
  ):
    """Constructor.

    Args:
      n_iter: number of iterations of binary search to use for inner max.
      n_pieces: number of discrete points to use for scalar inner max.
      solve_max: Which maximization routine to use.
      learning_rate: learning-rate to use for PGD attacks.
    """
    self._n_iter = n_iter
    self._n_pieces = n_pieces
    self._solve_max = solve_max
    self._learning_rate = learning_rate

  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.ndarray,
      step: int,
  ) -> jnp.ndarray:
    if self._solve_max == MaxType.EXP:
      return self.solve_max_exp(inner_dual_vars, opt_instance, key, step)
    elif self._solve_max == MaxType.EXP_BOUND:
      return self.upper_bound_softmax_plus_affine(inner_dual_vars, opt_instance,
                                                  key, step)
    else:
      raise ValueError(
          f'Unrecognized solve_max in uncertainty spec: {self._solve_max}.')

  def supports_stochastic_parameters(self):
    # does not rely on parameters
    return True

  def solve_max_exp(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.ndarray,
      step: int,
  ) -> jnp.ndarray:
    """Solve inner max problem for final layer with uncertainty specification.

    Maximize obj'*softmax(x) - lagrangian_form(x) subject to l<=x<=u

    Args:
      inner_dual_vars: () jax scalar.
      opt_instance: Inner optimization instance.
      key: RNG key.
      step: outer optimization iteration number.

    Returns:
      opt: Optimal value.
    """
    assert opt_instance.is_last
    l = opt_instance.bounds[0].lb_pre
    u = opt_instance.bounds[0].ub_pre

    def lagr_form(x):
      val = opt_instance.lagrangian_form_pre.apply(
          x, opt_instance.lagrange_params_pre, step)
      return jnp.reshape(val, ())

    affine_obj = lambda x: jnp.reshape(opt_instance.affine_fns[0](x), ())
    assert len(opt_instance.affine_fns) == 1

    def max_objective_fn(anyx):
      return affine_obj(jax.nn.softmax(anyx)) - lagr_form(anyx)

    min_objective_fn = lambda x: -max_objective_fn(x)

    opt = optax.adam(self._learning_rate)
    grad_fn = jax.grad(min_objective_fn)

    def cond_fn(inputs):
      it, x, grad_x, _ = inputs
      not_converged = jnp.logical_not(has_converged(x, grad_x, l, u))
      return jnp.logical_and(it < self._n_iter, not_converged)

    def body_fn(inputs):
      it, x, _, opt_state = inputs
      grad_x = grad_fn(x)
      updates, opt_state = opt.update(grad_x, opt_state, x)
      x = optax.apply_updates(x, updates)
      x = jnp.clip(x, l, u)
      it = it + 1
      return it, x, grad_x, opt_state

    def find_max_from_init(x):
      opt_state = opt.init(x)

      # iteration, x, grad_x, opt_state
      init_val = (jnp.zeros(()), x, jnp.ones_like(x), opt_state)
      _, adv_x, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)

      adv_x = jnp.clip(adv_x, l, u)

      return jnp.reshape(max_objective_fn(jax.lax.stop_gradient(adv_x)), (1,))

    # initialization heuristic 1: max when ignoring softmax
    mask_ignore_softmax = jax.grad(lagr_form)(jnp.ones_like(u)) < 0
    x = mask_ignore_softmax * u + (1 - mask_ignore_softmax) * l
    objective_1 = find_max_from_init(x)

    # initialization heuristic 2: max when ignoring affine
    mask_ignore_affine = jax.grad(affine_obj)(jnp.ones_like(u)) > 0
    x = mask_ignore_affine * u + (1 - mask_ignore_affine) * l
    objective_2 = find_max_from_init(x)

    # also try at boundaries
    objective_3 = find_max_from_init(l)
    objective_4 = find_max_from_init(u)

    # select best of runs
    objective = jnp.maximum(
        jnp.maximum(objective_1, objective_2),
        jnp.maximum(objective_3, objective_4))

    return objective

  def upper_bound_softmax_plus_affine(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.ndarray,
      step: int,
  ) -> jnp.ndarray:
    """Upper bound (softmax + affine)-type problem with cvxpy.

    Upper bound obj'*softmax(x) - lagrangian_form(x) subject to l<=x<=u

    Note that this function cannot be differentiated through; using it at
    training time will lead to an error.

    Args:
      inner_dual_vars: jax () scalar.
      opt_instance: Inner optimization instance.
      key: RNG key.
      step: outer optimization iteration number.

    Returns:
      Optimal value.
    Raises:
      ValueError if Lagrangian form is not supported or if the problem is not
        solved to optimality.
    """
    if not isinstance(opt_instance.lagrangian_form_pre, lagrangian_form.Linear):
      raise ValueError('Unsupported Lagrangian form.')

    lower = opt_instance.bounds[0].lb_pre
    upper = opt_instance.bounds[0].ub_pre

    def lagr_form(x):
      val = opt_instance.lagrangian_form_pre.apply(
          x, opt_instance.lagrange_params_pre, step)
      return jnp.reshape(val, ())

    # extract coeff_linear via autodiff (including negative sign here)
    coeff_linear = -jax.grad(lagr_form)(jnp.zeros_like(lower))

    assert len(opt_instance.affine_fns) == 1
    # extract coeff_softmax via autodiff
    coeff_softmax_fn = lambda x: jnp.reshape(opt_instance.affine_fns[0](x), ())
    coeff_softmax = jax.grad(coeff_softmax_fn)(jnp.zeros_like(lower))

    if opt_instance.spec_type == verify_utils.SpecType.ADVERSARIAL_SOFTMAX:
      upper_bounding_method = exact_opt_softmax.exact_opt_softmax_plus_affine
    else:
      upper_bounding_method = upper_bound_softmax_plus_affine_exact
    upper_bound, _ = upper_bounding_method(
        c_linear=np.array(coeff_linear).squeeze(0).astype(np.float64),
        c_softmax=np.array(coeff_softmax).squeeze(0).astype(np.float64),
        lb=np.array(lower).squeeze(0).astype(np.float64),
        ub=np.array(upper).squeeze(0).astype(np.float64),
    )

    constant = (
        coeff_softmax_fn(jnp.zeros_like(lower)) -
        lagr_form(jnp.zeros_like(lower)))

    result = jnp.array(upper_bound) + constant

    return jnp.reshape(result, [lower.shape[0]])

  def init_layer_inner_params(self, opt_instance):
    """Returns initial inner maximisation duals and their types."""
    if self._solve_max == MaxType.EXP:
      return None, sdp_utils.DualVarTypes.EQUALITY
    else:
      return (jnp.zeros_like(opt_instance.bounds[0].lb_pre),
              sdp_utils.DualVarTypes.EQUALITY)


def has_converged(x: Tensor, grad: Tensor, l: Tensor, u: Tensor):
  stuck_at_lower = jnp.logical_and(x == l, grad >= 0)
  stuck_at_upper = jnp.logical_and(x == u, grad <= 0)
  zero_grad = grad == 0

  stuck_at_border = jnp.logical_or(stuck_at_lower, stuck_at_upper)
  converged = jnp.logical_or(stuck_at_border, zero_grad)
  return jnp.all(converged)


def find_stationary_softmax_affine(
    c_linear: np.ndarray,
    c_softmax: np.ndarray,
    const_normalization: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
  """Find stationary point of softmax plus linear function.

  More specifically, solve for stationary point of
  exp(c_softmax ' * x)/(sum(exp(x))+const_normalization) +
  c_linear'* x with highest objective value.

  Args:
    c_linear: (n,) numpy array of linear coefficients.
    c_softmax: (n,) numpy array of softmax coefficients.
    const_normalization: (1,) numpy array representing constant term
    lb: Lower bounds
    ub: Upper bounds

  Returns:
    xopt: Optimal stationary point (None if none exists)
  """
  assert _is_zero(c_softmax) or _is_one_hot(c_softmax)

  idx = -1
  if np.sum(c_softmax) > 0:
    idx = np.argmax(c_softmax)

  def funx(x):
    x = np.reshape(x, [-1])
    if idx < 0:
      return 1 / (np.sum(np.exp(x)) + const_normalization) + np.sum(
          c_linear * x)
    else:
      return (np.exp(x[idx]) / (np.sum(np.exp(x)) + const_normalization) +
              np.sum(c_linear * x))

  if idx > 0:
    popt = np.zeros_like(c_linear)
    if c_linear[idx] < 0. and c_linear[idx] > -.25:
      popt_idx_a = .5 * (1 + np.sqrt(1 + 4 * c_linear[idx]))
      popt_idx_b = .5 * (1 - np.sqrt(1 + 4 * c_linear[idx]))
    else:
      return lb
    xopts = [lb, ub]
    for (i, popt_idx) in enumerate([popt_idx_a, popt_idx_b]):
      popt = c_linear / popt_idx
      popt[idx] = popt_idx
      if np.any(popt < 0.) or np.sum(popt) > 1.:
        xopts[i] = lb
      else:
        const = np.log(const_normalization / (1 - np.sum(popt)))
        xopt = const + np.log(popt)
        xopt = np.reshape(xopt, lb.shape)
        xopts[i] = np.clip(xopt, lb, ub)
    if funx(xopts[0]) > funx(xopts[1]):
      return xopts[0]
    else:
      return xopts[1]
  else:
    radical = 1 - 4. * np.sum(c_linear) * const_normalization
    if radical > 0. and np.all(c_linear > 0):
      const_a = .5 * (1 + np.sqrt(radical)) / np.sum(c_linear)
      const_b = .5 * (1 - np.sqrt(radical)) / np.sum(c_linear)
      x_a = np.clip(2 * np.log(const_a) + np.log(c_linear), lb, ub)
      x_b = np.clip(2 * np.log(const_b) + np.log(c_linear), lb, ub)
      if funx(x_a) > funx(x_b):
        xopt = x_a
      else:
        xopt = x_b
      xopt = np.reshape(xopt, lb.shape)
      return xopt
  return lb


def _truncated_exp(
    x: np.ndarray,
    x_min: np.ndarray = -20,
    x_max: np.ndarray = 20,
) -> np.ndarray:  # pytype: disable=annotation-type-mismatch
  """Truncate before exponentiation for numerical stability."""
  return np.exp(np.clip(x, x_min, x_max))


def _is_one_hot(x: np.ndarray) -> bool:
  return np.all(x * x == x) and np.sum(x) == 1


def _is_zero(x: np.ndarray) -> bool:
  return np.all(x == 0)


def upper_bound_softmax_plus_affine_exact(
    c_linear: np.ndarray,
    c_softmax: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    epsilon: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
  """Upper bound exp(c_softmax'*x-logsumexp(x)) + c_linear'* x subject to l <= x <= u.

  Args:
    c_linear: (n,) numpy array of linear coefficients.
    c_softmax: (n,) numpy array of softmax coefficients.
    lb: (n,) numpy array of lower bounds.
    ub: (n,) numpy array of upper bounds.
    epsilon: small constant for numerical stability.

  Returns:
    objval: optimal value.
    xopt: solution found for the upper bound.
  """
  assert _is_one_hot(c_softmax)

  # flatten all arrays
  c_softmax = np.ravel(c_softmax)
  c_linear = np.ravel(c_linear)
  lb = np.ravel(lb)
  ub = np.ravel(ub)

  # find index encoded by one-hot softmax
  idx = np.argmax(c_softmax)

  def objective_fn(x: np.ndarray) -> np.ndarray:
    return np.sum(c_softmax * scipy.special.softmax(x)) + np.sum(c_linear * x)

  obj_best = -np.inf
  num_coordinates = lb.size

  # each coordinate of the solution can be either at the lower bound, the upper
  # bound, or in the interior feasible domain. Iterate over all possible
  # combinations below
  for state_coordinates in itertools.product(['interior', 'lower', 'upper'],
                                             repeat=num_coordinates):
    in_interior = np.ones([num_coordinates], dtype=bool)

    const_normalization = epsilon  # zero init + small constant for stability
    const_numerator = 1.0

    # initialize candidate solution at lower bound
    xsol = np.copy(lb)

    for j in range(num_coordinates):
      if state_coordinates[j] in ('lower', 'upper'):
        # coordinate j is at lower bound or upper bound
        in_interior[j] = 0
        xsol[j] = lb[j] if state_coordinates[j] == 'lower' else ub[j]
        # compute exp while taking care of numerical stability
        exp_xsol_j = _truncated_exp(xsol[j])
        const_normalization += exp_xsol_j
        if j == idx:
          const_numerator *= exp_xsol_j

    if not _is_zero(in_interior):
      # solve following problem for coordinates that are `in_interior`:
      # max_x [exp(c_softmax^T x) / (sum(exp(x))+ const_normalization)
      #        + 1/const_numerator * c_linear^T x]
      # s.t. lb <= x <= ub.
      #
      # estimate relative importance of softmax to detect numerical issues
      softmax_relative_importance = (
          const_numerator / (const_normalization + epsilon))

      if softmax_relative_importance > epsilon:
        # solve softmax + linear problem if well-conditioned
        interior_values = find_stationary_softmax_affine(
            c_linear[in_interior] / const_numerator, c_softmax[in_interior],
            const_normalization, lb[in_interior], ub[in_interior])
      else:
        # otherwise softmax can be ignored, solve linear part in closed-form
        mask = c_linear[in_interior] > 0
        interior_values = mask * ub[in_interior] + (1 - mask) * lb[in_interior]

      # update candidate solution with values found at interior
      if interior_values is not None:
        xsol[in_interior] = interior_values

    # project candidate to feasible space and evaluate
    xsol = np.clip(xsol, lb, ub)
    objective_xsol = objective_fn(xsol)

    # keep if best so far
    if objective_xsol > obj_best:
      obj_best = objective_xsol
      xbest = xsol

  return obj_best, xbest
