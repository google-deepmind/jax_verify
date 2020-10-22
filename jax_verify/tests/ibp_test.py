# coding=utf-8
# Copyright 2020 The jax_verify Authors.
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
        hk.without_apply_rng(hk.transform(linear_model, apply_rng=True)).apply,
        params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.interval_bound_propagation(fun, input_bounds)

    self.assertAlmostEqual(5., output_bounds.lower)
    self.assertAlmostEqual(11., output_bounds.upper)

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
        hk.without_apply_rng(hk.transform(conv1d_model, apply_rng=True)).apply,
        params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.interval_bound_propagation(fun, input_bounds)

    self.assertAlmostEqual(7., output_bounds.lower)
    self.assertAlmostEqual(11., output_bounds.upper)

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
        hk.without_apply_rng(hk.transform(conv2d_model, apply_rng=True)).apply,
        params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.interval_bound_propagation(fun, input_bounds)

    self.assertAlmostEqual(8., output_bounds.lower)
    self.assertAlmostEqual(16., output_bounds.upper)

  def test_relu_ibp(self):
    def relu_model(inp):
      return jax.nn.relu(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.interval_bound_propagation(relu_model,
                                                          input_bounds)

    self.assertArrayAlmostEqual(jnp.array([[0., 2.]]), output_bounds.lower)
    self.assertArrayAlmostEqual(jnp.array([[0., 4.]]), output_bounds.upper)

  def test_softplus_ibp(self):
    def softplus_model(inp):
      return jax.nn.softplus(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.interval_bound_propagation(softplus_model,
                                                          input_bounds)

    self.assertArrayAlmostEqual(jnp.logaddexp(z - 1., 0),
                                output_bounds.lower)
    self.assertArrayAlmostEqual(jnp.logaddexp(z + 1., 0),
                                output_bounds.upper)


if __name__ == '__main__':
  absltest.main()
