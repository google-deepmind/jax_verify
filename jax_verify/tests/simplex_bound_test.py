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

"""Tests for IntervalSimplex bounds."""

from absl.testing import absltest

import chex
import jax
import jax.numpy as jnp


from jax_verify.src import simplex_bound
from jax_verify.src.linear import linear_relaxations
from jax_verify.src.types import Tensor
from jax_verify.tests import test_utils


def _check_bounds(inp_bound: simplex_bound.SimplexIntervalBound,
                  lin_coeffs: linear_relaxations.LinearExpression,
                  ref_bounds: Tensor):
  nb_tests = lin_coeffs.shape[0]
  # Let's compare the bound and make sure that they are all computed
  # correctly.
  lin_expr = linear_relaxations.LinearExpression(
      lin_coeffs, jnp.zeros((nb_tests,)))
  computed_bounds = (
      simplex_bound.concretize_linear_function_simplexinterval_constraints(
          lin_expr, inp_bound))
  chex.assert_trees_all_close(ref_bounds, computed_bounds)

  # Let's ensure that the linear bound offsets are correctly incorporated
  # too.
  offsets = jnp.arange(nb_tests).astype(jnp.float32)
  lin_expr_with_offsets = linear_relaxations.LinearExpression(
      lin_coeffs, offsets)
  ref_bounds_with_offsets = ref_bounds + offsets

  computed_bounds = (
      simplex_bound.concretize_linear_function_simplexinterval_constraints(
          lin_expr_with_offsets, inp_bound))
  chex.assert_trees_all_close(ref_bounds_with_offsets, computed_bounds)


class SimplexIntervalBoundTest(absltest.TestCase):

  def test_01inp_linear_bounds(self):
    inp_bound = simplex_bound.SimplexIntervalBound(
        jnp.array([0., 0., 0.]), jnp.array([1., 1., 1.]),
        1.5)

    lin_coeffs = jnp.array([
        # Computing the ground truth is always going to correspond to
        # picking completely the smallest value, and half of the second one.
        [3., 2., 0.],    # -> Pick 2, half of 1 -> 1.
        [-3., -1., 2.],    # -> Pick 0, half of 1 -> -3.5.
        [1., -2., -5.],    # -> Pick 2, half of 1 -> -6.
        [-2., -3., -1.],   # -> Pick 1, half of 0 -> -4.
        # Testing with some ties.
        [1., 1., 1.],    # -> Pick whatever -> 1.5
        [-2., 1., 1.],    # -> Pick 0, half of either 1 or 2 -> -1.5
        [2., 1., 1.],    # -> Pick 1.5 out of (1, 2) -> 1.5.
    ])
    ref_bounds = jnp.array([1., -3.5, -6., -4., 1.5, -1.5, 1.5])
    _check_bounds(inp_bound, lin_coeffs, ref_bounds)

  def test_fixedbysimplexinp_linear_bounds(self):
    fixed_at_ub_inp_bound = simplex_bound.SimplexIntervalBound(
        jnp.array([0., 0., 0.]), jnp.array([1., 2., 3.]),
        6.)

    lin_coeffs = jnp.array([
        [1., 1., 1.],
        [-1., 0., 1.],
        [-3., -1., -2.],
        [1., 3., -1.],
        [0., 0., 0.]
    ])
    # In any case, we need to pick up everything.
    ref_bounds = lin_coeffs @ fixed_at_ub_inp_bound.upper
    _check_bounds(fixed_at_ub_inp_bound, lin_coeffs, ref_bounds)

    fixed_at_lb_inp_bound = simplex_bound.SimplexIntervalBound(
        jnp.array([1., 2., 3.]), jnp.array([4., 6., 5.]),
        6.)
    ref_bounds = lin_coeffs @ fixed_at_lb_inp_bound.lower
    _check_bounds(fixed_at_lb_inp_bound, lin_coeffs, ref_bounds)

  def test_negativesimplex_linearbounds(self):
    inp_bound = simplex_bound.SimplexIntervalBound(
        -jnp.ones((3,)), jnp.ones((3,)), -2.)

    lin_coeffs = jnp.array([
        [1., 1., 1.],
        [0., 1., 2.],
        [-1., -2., -3.],
        [-1., 0., 2.],
        [1., -1., -1.]
    ])
    # This is equivalent to solving the problem where we have a simplex sum of
    # 1., except that we add the constant term where all the elements are set at
    # the minimum of -1.
    lin_exp = linear_relaxations.LinearExpression(
        lin_coeffs, jnp.zeros(lin_coeffs.shape[0]))
    eq_bound = simplex_bound.SimplexIntervalBound(jnp.zeros((3,)),
                                                  2*jnp.ones((3,)), 1.)
    shifted_prob_sol = (
        simplex_bound.concretize_linear_function_simplexinterval_constraints(
            lin_exp, eq_bound))
    ref_bounds = lin_coeffs @ inp_bound.lower + shifted_prob_sol
    _check_bounds(inp_bound, lin_coeffs, ref_bounds)

  def test_project_onto_bound(self):
    shape = (1, 10)
    lower, upper = test_utils.sample_bounds(jax.random.PRNGKey(0),
                                            shape)
    simplex_sum_interp = jax.random.uniform(jax.random.PRNGKey(1), ())
    simplex_sum = lower.sum() + (upper - lower).sum() * simplex_sum_interp

    inp_bound = simplex_bound.SimplexIntervalBound(lower, upper, simplex_sum)  # pytype: disable=wrong-arg-types  # jax-ndarray

    test_tensor = jax.random.uniform(jax.random.PRNGKey(2), shape)

    proj_test_tensor = inp_bound.project_onto_bound(test_tensor)

    self.assertGreaterEqual((proj_test_tensor-lower).min(), 0.)
    self.assertGreaterEqual((upper - proj_test_tensor).min(), 0.)
    self.assertAlmostEqual(proj_test_tensor.sum(), simplex_sum)

  def test_project_onto_bound_identity(self):
    shape = (1, 10)
    lower, upper = test_utils.sample_bounds(jax.random.PRNGKey(0),
                                            shape)
    simplex_sum_interp = jax.random.uniform(jax.random.PRNGKey(1), ())
    simplex_sum = lower.sum() + (upper - lower).sum() * simplex_sum_interp

    inp_bound = simplex_bound.SimplexIntervalBound(lower, upper, simplex_sum)  # pytype: disable=wrong-arg-types  # jax-ndarray
    test_tensor = lower + (upper - lower) * simplex_sum_interp

    proj_test_tensor = inp_bound.project_onto_bound(test_tensor)
    chex.assert_trees_all_close(test_tensor, proj_test_tensor,
                                atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
