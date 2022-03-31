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

"""Solve linear(softmax(x)) + linear(x) subject to l <= x <= b."""

import itertools
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy


def exact_opt_softmax_plus_affine(c_linear: np.ndarray, c_softmax: np.ndarray,
                                  lb: np.ndarray, ub: np.ndarray):
  """Maximize c_softmax'* softmax(x) + c_linear'* x subject to l <= x <= u.

  Args:
    c_linear: (n,) numpy array of linear coefficients.
    c_softmax: (n,) numpy array of softmax coefficients.
    lb: (n,) numpy array of lower bounds.
    ub: (n,) numpy array of upper bounds.

  Returns:
    objval: optimal value.
    xopt: solution found for the upper bound.
  """
  lb = np.reshape(lb, [-1])
  ub = np.reshape(ub, [-1])

  # Offset for numerical stability
  offset = np.max(ub)
  lb = lb - offset
  ub = ub - offset

  def funx(x):
    x = np.reshape(x, [-1])
    return np.sum(c_softmax * scipy.special.softmax(x)) + np.sum(c_linear * x)

  xbest = None
  obj_best = -np.inf
  for nums in itertools.product(['interior', 'lower', 'upper'], repeat=lb.size):
    nonbinding = np.ones((len(nums),))
    const_normalization = 0.
    const_numerator = 0.
    # Make a copy of lb
    xsol = lb + np.zeros_like(lb)
    for j in range(len(nums)):
      if nums[j] in ('lower', 'upper'):
        cj = lb[j] if nums[j] == 'lower' else ub[j]
        const_normalization += np.exp(cj)
        const_numerator += c_softmax[j] * np.exp(cj)
        nonbinding[j] = 0
        xsol[j] = cj
    xsols = []
    if np.sum(nonbinding) > 0:
      nonbinding = nonbinding > 0
      nonbinding_vals = solve_equality_subproblem(
          const_numerator, const_normalization, c_softmax[nonbinding],
          c_linear[nonbinding], lb[nonbinding], ub[nonbinding])
      for i in range(nonbinding_vals.shape[1]):
        xsol_i = np.copy(np.reshape(xsol, [-1]))
        xsol_i[nonbinding] = nonbinding_vals[:, i]
        xsols.append(xsol_i)
    else:
      xsols = [xsol]
    for xsol in xsols:
      xsol = np.clip(xsol, lb, ub)
      obj_cur = funx(xsol)
      if obj_cur > obj_best:
        obj_best = obj_cur
        xbest = xsol
  # Add constant correction for offsetting lb and ub
  obj_best = obj_best + offset * jnp.sum(c_linear)
  return obj_best, xbest


def solve_equality_subproblem(
    scalar_a: float,
    scalar_b: float,
    coeff_vec: np.ndarray,
    lagrangian_vec: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
):
  """Maximize equality constrained subproblem.

  (scalar_a + coeff_vec'* exp(x))/(scalar_b + sum(exp(x))) +
    lagrangian_vec '* x
  subject to lb <= x <= ub.

  Args:
    scalar_a: Scalar
    scalar_b: Scalar
    coeff_vec: (n,) numpy array of rescaled softmax coefficients.
    lagrangian_vec: (n,) numpy array of linear coefficients.
    lb: (n,) numpy array of lower bounds.
    ub: (n,) numpy array of upper bounds.

  Returns:
    objval: optimal value.
    xopt: solution found for the upper bound.
  """
  coeff_vec = np.reshape(coeff_vec, [-1])
  lagrangian_vec = np.reshape(lagrangian_vec, [-1])
  xopt = stationary_points(scalar_a, scalar_b, coeff_vec, lagrangian_vec)
  if xopt is not None:
    return np.clip(xopt, np.reshape(lb, [-1, 1]), np.reshape(ub, [-1, 1]))
  else:
    return np.reshape(lb, [-1, 1])


def _coeffs(f, degree):
  return jnp.linalg.solve(
      np.vander((np.arange(degree) + 1)).astype(float),
      f((jnp.arange(degree) + 1).astype(float)))


@jax.jit
def eval_polynomial(
    x: jnp.ndarray,
    coeff_a: float,
    coeff_b: float,
    mul_coeffs: jnp.ndarray,
    sub_coeffs: jnp.ndarray,
) -> jnp.ndarray:
  """Evaluate polynomial.

  Evaluate the polynomial corresponding to the rational equation
  (coeff_b * x - coeff_a) + sum_i mul_coeffs[i]/(x-sub_coeffs[i])
  at x.

  Args:
    x: (n,)
    coeff_a: Scalar
    coeff_b: Scalar
    mul_coeffs: (n,) numpy array of multiplicative coefficients
    sub_coeffs: (n,) numpy array of subtractive coefficients

  Returns:
    Values of polynomial at x (same shape as x).
  """
  result = 0.
  x = jnp.reshape(x, [-1, 1])
  for i in range(mul_coeffs.size):
    coeffs_not_i = (np.arange(mul_coeffs.size) != i)
    result += (
        mul_coeffs[i] *
        jnp.prod(x - jnp.reshape(sub_coeffs[coeffs_not_i], [1, -1]), axis=-1))
  result = jnp.reshape(result, [-1])
  result -= (
      jnp.reshape(coeff_b * x - coeff_a, [-1]) * jnp.reshape(
          jnp.prod(x - jnp.reshape(sub_coeffs, [1, -1]), axis=-1), [-1]))
  return jnp.reshape(result, [-1])


def stationary_points(
    scalar_a: float,
    scalar_b: float,
    c_vec: np.ndarray,
    lam_vec: np.ndarray,
) -> Union[None, np.ndarray]:
  """Get stationary points for equality constrained problem.

  Find stationary points of
  (scalar_a + c_vec'* exp(x))/(scalar_b + sum(exp(x))) + lam_vec '* x.

  Args:
    scalar_a: Scalar
    scalar_b: Scalar
    c_vec: (n,) numpy array of multiplicative coefficients
    lam_vec: (n,) numpy array of subtractive coefficients

  Returns:
    (n, k) array of stationary points (where k is the number of stationary
      points)
  """
  assert scalar_b >= 0.

  # Toleranace for numerical issues
  eps = 1e-5
  vec_x = lambda x: np.reshape(x, [-1])
  lam_vec = vec_x(lam_vec)
  c_vec = vec_x(c_vec)

  # Solve the scalar equation
  # (coeff_b * z - coeff_a) +
  #   sum_i (c_vec[i] * scalar_b - scalar_a) * slam_vec[i]/(z-c_vec[i]) = 0
  # for z.
  # Roots of this equaltion represent possible values of
  # (scalar_a + c_vec'* exp(x))/(scalar_b + sum(exp(x))) at a stationary point

  # Solve by turning this into a polynomial equation by multiplying out the
  # denominators of the rational terms.

  # We first collect all equal terms on the denomintor, to minimize the degree
  # of the resulting polynomial
  c_vec_uniq = np.unique(c_vec)
  lam_vec_uniq = np.zeros_like(c_vec_uniq)
  for i in range(lam_vec_uniq.size):
    lam_vec_uniq[i] = np.sum(lam_vec[c_vec == c_vec_uniq[i]])
  lamc_uniq = (c_vec_uniq * scalar_b - scalar_a) * lam_vec_uniq

  # This represents the polynomial version of the rational function
  poly = lambda z: eval_polynomial(z, scalar_a, scalar_b, lamc_uniq, c_vec_uniq)

  # Extract coefficients of polynomial and compute roots.
  cs = _coeffs(poly, c_vec_uniq.size + 2)
  roots = np.roots(cs)

  # Compute x corresponding to each root
  sols = []
  for root in roots:

    # We only consider real roots
    root = np.real(root)

    # p_sol represents exp(x)/(scalar_b + sum(exp(x)))
    p_sol = lam_vec / (root - c_vec)

    # Check that p_sol has negative entries and root does not coincide with
    # c_vec, which would have rendered the rational equation undefined
    if np.all(p_sol > 0.) and np.all(np.abs(root - c_vec) > eps):

      # If scalar_b is positive, p_sol must add up to something smaller than 1.
      # If scalar_b is zero, p_sol must add up to 1.
      if scalar_b > 0. and (np.sum(p_sol) < 1.):

        # Recover exp(x) from p_sol
        sol = p_sol * scalar_b / (1 - np.sum(p_sol))
      else:
        sol = p_sol
      sols.append(np.reshape(np.log(sol), [-1, 1]))
  if sols:
    return np.concatenate(sols, axis=-1)
  else:
    return None
