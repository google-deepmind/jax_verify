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

"""Tests for solving the convex relaxation using CVXPY."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src.mip_solver import cvxpy_relaxation_solver
from jax_verify.src.mip_solver import relaxation


class CVXPYRelaxationTest(parameterized.TestCase):

  def assertArrayAlmostEqual(self, lhs, rhs):
    diff = jnp.abs(lhs - rhs).max()
    self.assertAlmostEqual(diff, 0.)

  def get_bounds(self, fun, input_bounds):
    output = fun(input_bounds.lower)

    boundprop_transform = jax_verify.ibp_transform
    relaxation_transform = relaxation.RelaxationTransform(boundprop_transform)
    var, env = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(relaxation_transform),
        fun, input_bounds)

    objective_bias = 0.
    index = 0

    lower_bounds = []
    upper_bounds = []
    for output_idx in range(output.size):
      objective = (jnp.arange(output.size) == output_idx).astype(jnp.float32)

      lower_bound, _, _ = relaxation.solve_relaxation(
          cvxpy_relaxation_solver.CvxpySolver, objective, objective_bias,
          var, env, index)

      neg_upper_bound, _, _ = relaxation.solve_relaxation(
          cvxpy_relaxation_solver.CvxpySolver, -objective, objective_bias,
          var, env, index)
      lower_bounds.append(lower_bound)
      upper_bounds.append(-neg_upper_bound)

    return jnp.array(lower_bounds), jnp.array(upper_bounds)

  def test_linear_cvxpy_relaxation(self):

    def linear_model(inp):
      return hk.Linear(1)(inp)

    z = jnp.array([[1., 2., 3.]])
    params = {'linear':
              {'w': jnp.ones((3, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(
        hk.without_apply_rng(hk.transform(linear_model)).apply,
        params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    lower_bounds, upper_bounds = self.get_bounds(fun, input_bounds)
    self.assertAlmostEqual(5., lower_bounds)
    self.assertAlmostEqual(11., upper_bounds)

  def test_conv1d_cvxpy_relaxation(self):

    def conv1d_model(inp):
      return hk.Conv1D(output_channels=1, kernel_shape=2,
                       padding='VALID', stride=1, with_bias=True)(inp)
    z = jnp.array([3., 4.])
    z = jnp.reshape(z, [1, 2, 1])

    params = {'conv1_d':
              {'w': jnp.ones((2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(
        hk.without_apply_rng(hk.transform(conv1d_model)).apply,
        params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    lower_bounds, upper_bounds = self.get_bounds(fun, input_bounds)

    self.assertAlmostEqual(7., lower_bounds, delta=1e-5)
    self.assertAlmostEqual(11., upper_bounds, delta=1e-5)

  def test_conv2d_cvxpy_relaxation(self):
    def conv2d_model(inp):
      return hk.Conv2D(output_channels=1, kernel_shape=(2, 2),
                       padding='VALID', stride=1, with_bias=True)(inp)
    z = jnp.array([1., 2., 3., 4.])
    z = jnp.reshape(z, [1, 2, 2, 1])

    params = {'conv2_d':
              {'w': jnp.ones((2, 2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(
        hk.without_apply_rng(hk.transform(conv2d_model)).apply,
        params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    lower_bounds, upper_bounds = self.get_bounds(fun, input_bounds)
    self.assertAlmostEqual(8., lower_bounds)
    self.assertAlmostEqual(16., upper_bounds)

  def test_relu_cvxpy_relaxation(self):
    def relu_model(inp):
      return jax.nn.relu(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    lower_bounds, upper_bounds = self.get_bounds(relu_model, input_bounds)

    self.assertArrayAlmostEqual(jnp.array([[0., 2.]]), lower_bounds)
    self.assertArrayAlmostEqual(jnp.array([[0., 4.]]), upper_bounds)


if __name__ == '__main__':
  absltest.main()
