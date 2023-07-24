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

"""Tests for Interval Bound Propagation."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
import numpy as np


class IBPTest(parameterized.TestCase):

  def assertArrayAlmostEqual(self, lhs, rhs):
    diff = jnp.abs(lhs - rhs).max()
    self.assertAlmostEqual(diff, 0.)

  def test_linear_ibp(self):

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
    output_bounds = jax_verify.interval_bound_propagation(fun, input_bounds)

    self.assertAlmostEqual(5., output_bounds.lower)
    self.assertAlmostEqual(11., output_bounds.upper)

    fused_output_bounds = jax_verify.interval_bound_propagation(
        fun, input_bounds, fused_linear=True)
    self.assertAlmostEqual(5., fused_output_bounds.lower)
    self.assertAlmostEqual(11., fused_output_bounds.upper)

  def test_conv1d_ibp(self):

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
    output_bounds = jax_verify.interval_bound_propagation(fun, input_bounds)

    self.assertAlmostEqual(7., output_bounds.lower, delta=1e-5)
    self.assertAlmostEqual(11., output_bounds.upper, delta=1e-5)

    fused_output_bounds = jax_verify.interval_bound_propagation(
        fun, input_bounds, fused_linear=True)

    self.assertAlmostEqual(7., fused_output_bounds.lower, delta=1e-5)
    self.assertAlmostEqual(11., fused_output_bounds.upper, delta=1e-5)

  def test_conv2d_ibp(self):
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
    output_bounds = jax_verify.interval_bound_propagation(fun, input_bounds)

    self.assertAlmostEqual(8., output_bounds.lower)
    self.assertAlmostEqual(16., output_bounds.upper)

    fused_output_bounds = jax_verify.interval_bound_propagation(
        fun, input_bounds, fused_linear=True)

    self.assertAlmostEqual(8., fused_output_bounds.lower)
    self.assertAlmostEqual(16., fused_output_bounds.upper)

  @parameterized.named_parameters(
      ('exp', jnp.exp, [[-2.0, 3.0]]),
      ('log', jnp.log, [[3.0, 5.0]]),
      ('relu', jax.nn.relu, [[-2.0, 3.0]]),
      ('softplus', jax.nn.softplus, [[-2.0, 3.0]]),
      ('sign', jnp.sign, [[-2.0, 3.0]]),
      ('sigmoid', jax.nn.sigmoid, [[-2.0, 3.0]]),
      (
          'dynamic_slice',
          lambda x: jax.lax.dynamic_slice(x, (1, 2, 3), (2, 1, 1)),
          np.arange(24).reshape((2, 3, 4)),
      ),
  )
  def test_passthrough_primitive(self, fn, inputs):
    z = jnp.array(inputs)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.interval_bound_propagation(fn, input_bounds)

    self.assertArrayAlmostEqual(fn(input_bounds.lower), output_bounds.lower)
    self.assertArrayAlmostEqual(fn(input_bounds.upper), output_bounds.upper)

  def test_ibp_neg(self):
    fn = lambda x: -x

    input_bounds = jax_verify.IntervalBound(jnp.zeros((2,)), jnp.ones((2,)))
    output_bounds = jax_verify.interval_bound_propagation(fn, input_bounds)

    self.assertArrayAlmostEqual(output_bounds.lower, -jnp.ones((2,)))
    self.assertArrayAlmostEqual(output_bounds.upper, jnp.zeros((2,)))

  @parameterized.named_parameters(
      ('positive', (1.0, 4.0), (1.0, 2.0)),
      ('negative', (-4.0, -1.0), (float('nan'), float('nan'))),
      ('zero_edge', (0.0, 1.0), (0.0, 1.0)),
      ('zero_cross', (-1.0, 1.0), (float('nan'), 1.0)))
  def test_sqrt(self, input_bounds, expected):
    input_bounds = jax_verify.IntervalBound(  # pytype: disable=wrong-arg-types  # jax-ndarray
        np.array([input_bounds[0], 0.0]), np.array([input_bounds[1], 0.0]))
    output_bounds = jax_verify.interval_bound_propagation(
        jnp.sqrt, input_bounds)
    np.testing.assert_array_equal(
        np.array([expected[0], 0.0]), output_bounds.lower)
    np.testing.assert_array_equal(
        np.array([expected[1], 0.0]), output_bounds.upper)

  @parameterized.named_parameters(
      ('square_positive', 2, (1.0, 2.0), (1.0, 4.0)),
      ('square_negative', 2, (-2.0, -1.0), (1.0, 4.0)),
      ('square_zero', 2, (-1.0, 2.0), (0.0, 4.0)),
      ('cube_positive', 3, (1.0, 2.0), (1.0, 8.0)),
      ('cube_negative', 3, (-2.0, -1.0), (-8.0, -1.0)),
      ('cube_zero', 3, (-1.0, 2.0), (-1.0, 8.0)))
  def test_integer_pow(self, exponent, input_bounds, expected):

    @jax.jit
    def _compute_bounds(lower, upper):
      input_bounds = jax_verify.IntervalBound(lower, upper)
      output_bounds = jax_verify.interval_bound_propagation(
          lambda x: x**exponent, input_bounds)
      return output_bounds.lower, output_bounds.upper

    output_bounds = _compute_bounds(
        np.array([input_bounds[0], 0.0]), np.array([input_bounds[1], 0.0]))
    np.testing.assert_array_equal(
        np.array([expected[0], 0.0]), output_bounds[0])
    np.testing.assert_array_equal(
        np.array([expected[1], 0.0]), output_bounds[1])


if __name__ == '__main__':
  absltest.main()
