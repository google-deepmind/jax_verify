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

"""Tests for Forward Linear Bounds."""

import functools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src.linear import forward_linear_bounds
from jax_verify.src.linear import linear_bound_utils
from jax_verify.tests import test_utils


def get_boundprop(name: str, elision: bool
                  ) -> Callable[..., forward_linear_bounds.LinearBound]:
  if name == 'fastlin':
    relaxer = linear_bound_utils.fastlin_rvt_relaxer
  elif name == 'crown':
    relaxer = linear_bound_utils.crown_rvt_relaxer

  transform = forward_linear_bounds.ForwardLinearBoundTransform(
      relaxer, elision)
  algorithm = bound_propagation.ForwardPropagationAlgorithm(transform)
  def bound_prop(function, *bounds) -> forward_linear_bounds.LinearBound:
    output_bound, _ = bound_propagation.bound_propagation(
        algorithm, function, *bounds)
    return output_bound
  return bound_prop


class ForwardLinBoundTest(parameterized.TestCase):

  def assertArrayAlmostEqual(self, lhs, rhs):
    diff = jnp.abs(lhs - rhs).max()
    self.assertAlmostEqual(diff, 0.)

  def assertArrayGreaterEqual(self, lhs, rhs):
    diff = (lhs-rhs).min()
    self.assertGreaterEqual(diff, 0.)

  @parameterized.named_parameters(
      ('fastlin_noelision', 'fastlin', False),
      ('fastlin_elision', 'fastlin', True),
      ('crown_noelison', 'crown', False),
      ('crown_elision', 'crown', True))
  def test_fc_fastlin(self, name, elision):

    @hk.without_apply_rng
    @hk.transform
    def linear_model(inp):
      return hk.Linear(1)(inp)

    z = jnp.array([[1., 2., 3.]])
    params = {'linear':
              {'w': jnp.ones((3, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}
    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    fun = functools.partial(linear_model.apply, params)
    bound_prop = get_boundprop(name, elision)
    output_bounds = bound_prop(fun, input_bounds)

    all_linear_functions = list(output_bounds.linear_functions())
    self.assertLen(all_linear_functions, 1)
    linear_fun = all_linear_functions[0]
    self.assertTrue(jnp.all(linear_fun.lower_lin.lin_coeffs == 1.))
    self.assertTrue(jnp.all(linear_fun.lower_lin.offset == 2.))
    self.assertTrue(jnp.all(linear_fun.upper_lin.lin_coeffs == 1.))
    self.assertTrue(jnp.all(linear_fun.upper_lin.offset == 2.))
    self.assertArrayAlmostEqual(jnp.array([[0., 1., 2.]]),
                                linear_fun.reference_bound.bound.lower)
    self.assertArrayAlmostEqual(jnp.array([[2., 3., 4.]]),
                                linear_fun.reference_bound.bound.upper)

    self.assertAlmostEqual(5., output_bounds.lower)
    self.assertAlmostEqual(11., output_bounds.upper)

  @parameterized.named_parameters(
      ('fastlin_noelision', 'fastlin', False),
      ('fastlin_elision', 'fastlin', True),
      ('crown_noelison', 'crown', False),
      ('crown_elision', 'crown', True))
  def test_conv2d_fastlin(self, name, elision):

    @hk.without_apply_rng
    @hk.transform
    def conv2d_model(inp):
      return hk.Conv2D(output_channels=1, kernel_shape=(2, 2),
                       padding='VALID', stride=1, with_bias=True)(inp)

    z = jnp.array([1., 2., 3., 4.])
    z = jnp.reshape(z, [1, 2, 2, 1])

    params = {'conv2_d':
              {'w': jnp.ones((2, 2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(conv2d_model.apply, params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    bound_prop = get_boundprop(name, elision)
    output_bounds = bound_prop(fun, input_bounds)

    self.assertAlmostEqual(8., output_bounds.lower)
    self.assertAlmostEqual(16., output_bounds.upper)

  @parameterized.named_parameters(
      ('fastlin_noelision', 'fastlin', False),
      ('fastlin_elision', 'fastlin', True),
      ('crown_noelison', 'crown', False),
      ('crown_elision', 'crown', True))
  def test_conv1d_fastlin(self, name, elision):

    @hk.without_apply_rng
    @hk.transform
    def conv1d_model(inp):
      return hk.Conv1D(output_channels=1, kernel_shape=2,
                       padding='VALID', stride=1, with_bias=True)(inp)

    z = jnp.array([3., 4.])
    z = jnp.reshape(z, [1, 2, 1])

    params = {'conv1_d':
              {'w': jnp.ones((2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(conv1d_model.apply, params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    bound_prop = get_boundprop(name, elision)
    output_bounds = bound_prop(fun, input_bounds)

    self.assertAlmostEqual(7., output_bounds.lower, delta=1e-5)
    self.assertAlmostEqual(11., output_bounds.upper, delta=1e-5)

  @parameterized.named_parameters(
      ('fastlin_noelision', 'fastlin', False),
      ('fastlin_elision', 'fastlin', True),
      ('crown_noelison', 'crown', False),
      ('crown_elision', 'crown', True))
  def test_multiinput_add_fastlin(self, name, elision):

    def add_model(inp_1, inp_2):
      interm = inp_1 + inp_2
      return interm.sum(axis=1)

    z_1 = jnp.array([[-1., 1.]])
    z_2 = jnp.array([[-1., 1.]])

    bound_1 = jax_verify.IntervalBound(z_1 - 1., z_1 + 1.)
    bound_2 = jax_verify.IntervalBound(z_2 - 1., z_2 + 1.)

    out_lower = (z_1 + z_2 - 2.).sum()
    out_upper = (z_1 + z_2 + 2.).sum()

    bound_prop = get_boundprop(name, elision)
    output_bounds = bound_prop(add_model, bound_1, bound_2)

    self.assertArrayAlmostEqual(out_lower, output_bounds.lower)
    self.assertArrayAlmostEqual(out_upper, output_bounds.upper)

  @parameterized.named_parameters(
      ('fastlin_noelision', 'fastlin', False),
      ('fastlin_elision', 'fastlin', True),
      ('crown_noelison', 'crown', False),
      ('crown_elision', 'crown', True))
  def test_multiinput_sub_fastlin(self, name, elision):

    def sub_model(inp_1, inp_2):
      interm = inp_1 - inp_2
      return interm.sum(axis=1)

    z_1 = jnp.array([[-1., 1.]])
    z_2 = jnp.array([[-1., 1.]])

    bound_1 = jax_verify.IntervalBound(z_1 - 1., z_1 + 1.)
    bound_2 = jax_verify.IntervalBound(z_2 - 1., z_2 + 1.)

    out_lower = (z_1 - z_2 - 2.).sum()
    out_upper = (z_1 - z_2 + 2.).sum()

    bound_prop = get_boundprop(name, elision)
    output_bounds = bound_prop(sub_model, bound_1, bound_2)

    self.assertArrayAlmostEqual(out_lower, output_bounds.lower)
    self.assertArrayAlmostEqual(out_upper, output_bounds.upper)

  @parameterized.named_parameters(
      ('fastlin_noelision', 'fastlin', False),
      ('fastlin_elision', 'fastlin', True),
      ('crown_noelison', 'crown', False),
      ('crown_elision', 'crown', True))
  def test_multiinput_concatenate_fastlin(self, name, elision):

    def concatenate_and_sum_model(inp_1, inp_2):
      interm = jnp.concatenate((inp_1, inp_2), axis=1)
      return interm.sum(axis=1)

    z_1 = jnp.array([[-1., 1.]])
    z_2 = jnp.array([[-1., 1.]])

    bound_1 = jax_verify.IntervalBound(z_1 - 1., z_1 + 1.)
    bound_2 = jax_verify.IntervalBound(z_2 - 1., z_2 + 1.)

    out_lower = (z_1 + z_2 - 2.).sum()
    out_upper = (z_1 - z_2 + 2.).sum()

    bound_prop = get_boundprop(name, elision)
    output_bounds = bound_prop(concatenate_and_sum_model, bound_1, bound_2)

    self.assertArrayAlmostEqual(out_lower, output_bounds.lower)
    self.assertArrayAlmostEqual(out_upper, output_bounds.upper)

  def test_relu_fixed_fastlin(self):
    def relu_model(inp):
      return jax.nn.relu(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.forward_fastlin_bound_propagation(relu_model,
                                                                 input_bounds)

    self.assertArrayAlmostEqual(jnp.array([[0., 2.]]), output_bounds.lower)
    self.assertArrayAlmostEqual(jnp.array([[0., 4.]]), output_bounds.upper)

  def test_relu_random_fastlin(self):
    def relu_model(inp):
      return jax.nn.relu(inp)
    relu_inp_shape = (4, 7)
    lb, ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), relu_inp_shape, minval=-10., maxval=10.)

    input_bounds = jax_verify.IntervalBound(lb, ub)
    output_bounds = jax_verify.forward_fastlin_bound_propagation(
        relu_model, input_bounds)

    uniform_inps = test_utils.sample_bounded_points(jax.random.PRNGKey(1),
                                                    (lb, ub), 100)
    uniform_outs = jax.vmap(relu_model)(uniform_inps)
    empirical_min = uniform_outs.min(axis=0)
    empirical_max = uniform_outs.max(axis=0)
    self.assertGreaterEqual((output_bounds.upper - empirical_max).min(), 0.,
                            'Invalid upper bound for ReLU. The gap '
                            'between upper bound and empirical max is < 0')
    self.assertGreaterEqual((empirical_min - output_bounds.lower).min(), 0.,
                            'Invalid lower bound for ReLU. The gap'
                            'between emp. min and lower bound is negative.')

  def test_exp_fastlin(self):
    def exp_model(inp):
      return jnp.exp(inp)
    exp_inp_shape = (4, 7)
    lb, ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), exp_inp_shape, minval=-10., maxval=10.)

    input_bounds = jax_verify.IntervalBound(lb, ub)
    output_bounds = jax_verify.forward_fastlin_bound_propagation(
        exp_model, input_bounds)

    uniform_inps = test_utils.sample_bounded_points(jax.random.PRNGKey(1),
                                                    (lb, ub), 100)
    uniform_outs = jax.vmap(exp_model)(uniform_inps)
    empirical_min = uniform_outs.min(axis=0)
    empirical_max = uniform_outs.max(axis=0)
    self.assertGreaterEqual((output_bounds.upper - empirical_max).min(), 0.,
                            'Invalid upper bound for Exponential. The gap '
                            'between upper bound and empirical max is < 0')
    self.assertGreaterEqual((empirical_min - output_bounds.lower).min(), 0.,
                            'Invalid lower bound for Exponential. The gap'
                            'between emp. min and lower bound is negative.')

  def test_multiply_fastlin(self):
    def multiply_model(lhs, rhs):
      return lhs * rhs
    mul_inp_shape = (4, 7)
    lhs_lb, lhs_ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), mul_inp_shape, minval=-10., maxval=10.)
    rhs_lb, rhs_ub = test_utils.sample_bounds(
        jax.random.PRNGKey(1), mul_inp_shape, minval=-10., maxval=10.)

    lhs_bounds = jax_verify.IntervalBound(lhs_lb, lhs_ub)
    rhs_bounds = jax_verify.IntervalBound(rhs_lb, rhs_ub)
    output_bounds = jax_verify.forward_fastlin_bound_propagation(
        multiply_model, lhs_bounds, rhs_bounds)

    uniform_lhs_inps = test_utils.sample_bounded_points(jax.random.PRNGKey(2),
                                                        (lhs_lb, lhs_ub), 100)
    uniform_rhs_inps = test_utils.sample_bounded_points(jax.random.PRNGKey(3),
                                                        (rhs_lb, rhs_ub), 100)

    uniform_outs = jax.vmap(multiply_model)(uniform_lhs_inps, uniform_rhs_inps)
    empirical_min = uniform_outs.min(axis=0)
    empirical_max = uniform_outs.max(axis=0)

    self.assertGreaterEqual((output_bounds.upper - empirical_max).min(), 0.,
                            'Invalid upper bound for Multiply. The gap '
                            'between upper bound and empirical max is negative')
    self.assertGreaterEqual((empirical_min - output_bounds.lower).min(), 0.,
                            'Invalid lower bound for Multiply. The gap'
                            'between emp. min and lower bound is negative.')

  def test_nobatch_batch_inputs(self):
    batch_shape = (3, 2)
    unbatch_shape = (2, 4)

    def bilinear_model(inp_1, inp_2):
      return jnp.einsum('bh,hH->bH', inp_1, inp_2)

    lb_1, ub_1 = test_utils.sample_bounds(jax.random.PRNGKey(0), batch_shape,
                                          minval=-10, maxval=10.)
    lb_2, ub_2 = test_utils.sample_bounds(jax.random.PRNGKey(1), unbatch_shape,
                                          minval=-10, maxval=10.)
    bound_1 = jax_verify.IntervalBound(lb_1, ub_1)
    bound_2 = jax_verify.IntervalBound(lb_2, ub_2)

    output_bounds = jax_verify.forward_fastlin_bound_propagation(
        bilinear_model, bound_1, bound_2)

    uniform_1 = test_utils.sample_bounded_points(jax.random.PRNGKey(2),
                                                 (lb_1, ub_1), 100)
    uniform_2 = test_utils.sample_bounded_points(jax.random.PRNGKey(3),
                                                 (lb_2, ub_2), 100)

    uniform_outs = jax.vmap(bilinear_model)(uniform_1, uniform_2)
    empirical_min = uniform_outs.min(axis=0)
    empirical_max = uniform_outs.max(axis=0)

    self.assertGreaterEqual((output_bounds.upper - empirical_max).min(), 0.,
                            'Invalid upper bound for mix of batched/unbatched'
                            'input bounds.')
    self.assertGreaterEqual((empirical_min - output_bounds.lower).min(), 0.,
                            'Invalid lower bound for mix of batched/unbatched'
                            'input bounds.')


class IBPFastLinBoundTest(parameterized.TestCase):

  def assertArrayAlmostEqual(self, lhs, rhs):
    diff = jnp.abs(lhs - rhs).max()
    self.assertAlmostEqual(diff, 0.)

  def test_fc_fastlin(self):

    @hk.without_apply_rng
    @hk.transform
    def linear_model(inp):
      return hk.Linear(1)(inp)

    z = jnp.array([[1., 2., 3.]])
    params = {'linear':
              {'w': jnp.ones((3, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}
    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    fun = functools.partial(linear_model.apply, params)
    output_bounds = jax_verify.ibpforwardfastlin_bound_propagation(fun,
                                                                   input_bounds)

    self.assertAlmostEqual(5., output_bounds.lower)
    self.assertAlmostEqual(11., output_bounds.upper)

  def test_conv2d_fastlin(self):

    @hk.without_apply_rng
    @hk.transform
    def conv2d_model(inp):
      return hk.Conv2D(output_channels=1, kernel_shape=(2, 2),
                       padding='VALID', stride=1, with_bias=True)(inp)

    z = jnp.array([1., 2., 3., 4.])
    z = jnp.reshape(z, [1, 2, 2, 1])

    params = {'conv2_d':
              {'w': jnp.ones((2, 2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(conv2d_model.apply, params)
    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    output_bounds = jax_verify.ibpforwardfastlin_bound_propagation(fun,
                                                                   input_bounds)

    self.assertAlmostEqual(8., output_bounds.lower)
    self.assertAlmostEqual(16., output_bounds.upper)

  def test_conv1d_fastlin(self):

    @hk.without_apply_rng
    @hk.transform
    def conv1d_model(inp):
      return hk.Conv1D(output_channels=1, kernel_shape=2,
                       padding='VALID', stride=1, with_bias=True)(inp)

    z = jnp.array([3., 4.])
    z = jnp.reshape(z, [1, 2, 1])

    params = {'conv1_d':
              {'w': jnp.ones((2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(conv1d_model.apply, params)
    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    output_bounds = jax_verify.ibpforwardfastlin_bound_propagation(fun,
                                                                   input_bounds)

    self.assertAlmostEqual(7., output_bounds.lower, delta=1e-5)
    self.assertAlmostEqual(11., output_bounds.upper, delta=1e-5)

  def test_relu_fastlin(self):
    def relu_model(inp):
      return jax.nn.relu(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    output_bounds = jax_verify.ibpforwardfastlin_bound_propagation(relu_model,
                                                                   input_bounds)

    self.assertArrayAlmostEqual(jnp.array([[0., 2.]]), output_bounds.lower)
    self.assertArrayAlmostEqual(jnp.array([[0., 4.]]), output_bounds.upper)


if __name__ == '__main__':
  absltest.main()
