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

"""Tests for Backward linear bounds (Crown / RVT)."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify

from jax_verify.src import bound_propagation
from jax_verify.src import concretization
from jax_verify.src import ibp
from jax_verify.src import optimizers
from jax_verify.src.linear import backward_crown
from jax_verify.src.linear import linear_relaxations
from jax_verify.tests import test_utils

import numpy as np
import optax


class BackwardCrownBoundTest(parameterized.TestCase):

  def assertArrayAlmostEqual(self, lhs, rhs):
    diff = jnp.abs(lhs - rhs).max()
    self.assertAlmostEqual(diff, 0., delta=1e-5)

  def test_fc_crown(self):

    @hk.without_apply_rng
    @hk.transform
    def linear_model(inp):
      return hk.Linear(1)(inp)

    z = jnp.array([[1., 2., 3.]])
    params = {'linear':
              {'w': jnp.ones((3, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}
    fun = functools.partial(linear_model.apply, params)

    # Test with standard interval bounds.
    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    output_bounds = jax_verify.backward_crown_bound_propagation(
        fun, input_bounds)

    self.assertArrayAlmostEqual(5., output_bounds.lower)
    self.assertArrayAlmostEqual(11., output_bounds.upper)

  def test_conv2d_crown(self):

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

    # Test with standard interval bounds
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.backward_crown_bound_propagation(
        fun, input_bounds)

    self.assertArrayAlmostEqual(8., output_bounds.lower)
    self.assertArrayAlmostEqual(16., output_bounds.upper)

  def test_dynamic_slice(self):
    z = jnp.arange(24).reshape((2, 3, 4))
    fun = lambda x: jax.lax.dynamic_slice(x, (1, 2, 3), (2, 1, 1))
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    output_bounds = jax_verify.backward_crown_bound_propagation(
        fun, input_bounds)
    self.assertArrayAlmostEqual(output_bounds.lower,
                                fun(input_bounds.lower))
    self.assertArrayAlmostEqual(output_bounds.upper,
                                fun(input_bounds.upper))

  def test_relu_crown(self):
    def relu_model(inp):
      return jax.nn.relu(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)
    output_bounds = jax_verify.backward_crown_bound_propagation(
        relu_model, input_bounds)

    self.assertArrayAlmostEqual(jnp.array([[0., 2.]]), output_bounds.lower)
    self.assertArrayAlmostEqual(jnp.array([[0., 4.]]), output_bounds.upper)

  def test_abs_crown(self):
    def abs_model(inp):
      return jnp.abs(inp)
    abs_inp_shape = (4, 7)
    lb, ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), abs_inp_shape, minval=-10., maxval=10.)

    input_bounds = jax_verify.IntervalBound(lb, ub)
    output_bounds = jax_verify.backward_crown_bound_propagation(
        abs_model, input_bounds)

    uniform_inps = test_utils.sample_bounded_points(jax.random.PRNGKey(1),
                                                    (lb, ub), 100)
    uniform_outs = jax.vmap(abs_model)(uniform_inps)
    empirical_min = uniform_outs.min(axis=0)
    empirical_max = uniform_outs.max(axis=0)

    self.assertGreaterEqual((output_bounds.upper - empirical_max).min(), 0.,
                            'Invalid upper bound for AbsValue. The gap '
                            'between upper bound and empirical max is < 0')
    self.assertGreaterEqual((empirical_min - output_bounds.lower).min(), 0.,
                            'Invalid lower bound for AbsValue. The gap'
                            'between emp. min and lower bound is negative.')

  def test_leaky_relu_crown(self):
    def leaky_relu_model(inp):
      return jax.nn.leaky_relu(inp)
    leaky_relu_inp_shape = (4, 7)
    lb, ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), leaky_relu_inp_shape, minval=-10., maxval=10.)

    input_bounds = jax_verify.IntervalBound(lb, ub)
    output_bounds = jax_verify.backward_crown_bound_propagation(
        leaky_relu_model, input_bounds)

    uniform_inps = test_utils.sample_bounded_points(jax.random.PRNGKey(1),
                                                    (lb, ub), 100)
    uniform_outs = jax.vmap(leaky_relu_model)(uniform_inps)
    empirical_min = uniform_outs.min(axis=0)
    empirical_max = uniform_outs.max(axis=0)

    self.assertGreaterEqual((output_bounds.upper - empirical_max).min(), 0.,
                            'Invalid upper bound for LeakyReLU. The gap '
                            'between upper bound and empirical max is < 0')
    self.assertGreaterEqual((empirical_min - output_bounds.lower).min(), 0.,
                            'Invalid lower bound for LeakyRelu. The gap'
                            'between emp. min and lower bound is negative.')

  def test_exp_crown(self):
    def exp_model(inp):
      return jnp.exp(inp)
    exp_inp_shape = (4, 7)
    lb, ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), exp_inp_shape, minval=-10., maxval=10.)

    input_bounds = jax_verify.IntervalBound(lb, ub)
    output_bounds = jax_verify.backward_crown_bound_propagation(
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

  def test_multiply_crown(self):
    def multiply_model(lhs, rhs):
      return lhs * rhs
    mul_inp_shape = (4, 7)
    lhs_lb, lhs_ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), mul_inp_shape, minval=-10., maxval=10.)
    rhs_lb, rhs_ub = test_utils.sample_bounds(
        jax.random.PRNGKey(1), mul_inp_shape, minval=-10., maxval=10.)

    lhs_bounds = jax_verify.IntervalBound(lhs_lb, lhs_ub)
    rhs_bounds = jax_verify.IntervalBound(rhs_lb, rhs_ub)
    output_bounds = jax_verify.backward_crown_bound_propagation(
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

    output_bounds = backward_crown.backward_crown_bound_propagation(
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

  def test_equal_bounds_parameterized(self):
    model = jax.nn.relu
    sample_value = jnp.array([-1., 1.])
    inp_bound = jax_verify.IntervalBound(sample_value, sample_value)

    optimizer = optimizers.OptaxOptimizer(optax.adam(1e-3), num_steps=10)

    concretizer = concretization.ChunkedBackwardConcretizer(
        backward_crown.OptimizingLinearBoundBackwardTransform(
            linear_relaxations.parameterized_relaxer,
            backward_crown.CONCRETIZE_ARGS_PRIMITIVE,
            optimizer))

    algorithm = concretization.BackwardConcretizingAlgorithm(concretizer)
    bound, _ = bound_propagation.bound_propagation(
        algorithm, model, inp_bound)

    np.testing.assert_array_almost_equal(bound.lower, bound.upper)  # pytype: disable=attribute-error  # jax-ndarray

  def test_forwardconcretization_withbackwardalg_reference_out(self):
    architecture = [2, 4, 4, 2]
    problem_key = jax.random.PRNGKey(42)
    fun, (lb, ub) = test_utils.set_up_toy_problem(problem_key, 2,
                                                  architecture)
    def model_fun(inp):
      out = fun(inp)
      select = jnp.array([[True, False]])
      # We are causing the last operation to be a select, to force it to be a
      # reference.
      return jnp.where(select, out, -1.* out)

    inp_bound = jax_verify.IntervalBound(lb, ub)

    optimizer = optimizers.OptaxOptimizer(optax.adam(1e-3), num_steps=10)
    concretizer = concretization.ChunkedBackwardConcretizer(
        backward_crown.OptimizingLinearBoundBackwardTransform(
            linear_relaxations.parameterized_relaxer,
            backward_crown.CONCRETIZE_ARGS_PRIMITIVE,
            optimizer))

    algorithm = concretization.BackwardAlgorithmForwardConcretization(
        ibp.bound_transform, concretizer)

    # This used to cause an exception as the backward discovery of nodes
    # needing relaxation was not working. The Scanner was not being
    # propagated backwards.
    bound_propagation.bound_propagation(algorithm, model_fun, inp_bound)

  def test_keyword_and_flat_params(self):
    def model_fun(p_dict, a):
      elt_1 = p_dict['elt_1']
      elt_2 = p_dict['elt_2']

      interm = jax.nn.relu(elt_1 @ elt_2)

      return jnp.sum(interm + a)

    p_dict = {'elt_1': jnp.ones((2, 10)),
              'elt_2': jax_verify.IntervalBound(jnp.zeros((10, 2)),
                                                jnp.ones((10, 2)))}
    a = jax_verify.IntervalBound(jnp.zeros((2, 2)),
                                 jnp.ones((2, 2)))

    # This used to cause an exception due to a bug in the implementation
    # of concretization.BackwardConcretizationAlgorithm, when the inputs bound
    # get filled in the backward_env.
    backward_crown.backward_crown_bound_propagation(model_fun, p_dict, a)

  @parameterized.named_parameters(
      ('crown', linear_relaxations.crown_rvt_relaxer),
      ('fastlin', linear_relaxations.fastlin_rvt_relaxer),
      ('parameterized', linear_relaxations.parameterized_relaxer),
  )
  def test_chunking(self, relaxer):
    batch_size = 3
    input_size = 2
    hidden_size = 5
    final_size = 4

    input_shape = (batch_size, input_size)
    hidden_lay_weight_shape = (input_size, hidden_size)
    final_lay_weight_shape = (hidden_size, final_size)

    inp_lb, inp_ub = test_utils.sample_bounds(
        jax.random.PRNGKey(0), input_shape,
        minval=-1., maxval=1.)
    inp_bound = jax_verify.IntervalBound(inp_lb, inp_ub)

    hidden_lay_weight = jax.random.uniform(jax.random.PRNGKey(1),
                                           hidden_lay_weight_shape)
    final_lay_weight = jax.random.uniform(jax.random.PRNGKey(2),
                                          final_lay_weight_shape)

    def model_fun(inp):
      hidden = inp @ hidden_lay_weight
      act = jax.nn.relu(hidden)
      final = act @ final_lay_weight
      return final

    if isinstance(relaxer, linear_relaxations.ParameterizedLinearBoundsRelaxer):
      optimizer = optimizers.OptaxOptimizer(optax.adam(1e-3), num_steps=10)
      concretizing_transform = (
          backward_crown.OptimizingLinearBoundBackwardTransform(
              relaxer, backward_crown.CONCRETIZE_ARGS_PRIMITIVE,
              optimizer))
    else:
      concretizing_transform = backward_crown.LinearBoundBackwardTransform(
          relaxer, backward_crown.CONCRETIZE_ARGS_PRIMITIVE)

    chunked_concretizer = concretization.ChunkedBackwardConcretizer(
        concretizing_transform, max_chunk_size=16)
    unchunked_concretizer = concretization.ChunkedBackwardConcretizer(
        concretizing_transform, max_chunk_size=0)

    chunked_algorithm = concretization.BackwardConcretizingAlgorithm(
        chunked_concretizer)
    unchunked_algorithm = concretization.BackwardConcretizingAlgorithm(
        unchunked_concretizer)

    chunked_bound, _ = bound_propagation.bound_propagation(
        chunked_algorithm, model_fun, inp_bound)
    unchunked_bound, _ = bound_propagation.bound_propagation(
        unchunked_algorithm, model_fun, inp_bound)

    np.testing.assert_array_almost_equal(chunked_bound.lower,  # pytype: disable=attribute-error  # jax-ndarray
                                         unchunked_bound.lower)  # pytype: disable=attribute-error  # jax-ndarray
    np.testing.assert_array_almost_equal(chunked_bound.upper,  # pytype: disable=attribute-error  # jax-ndarray
                                         unchunked_bound.upper)  # pytype: disable=attribute-error  # jax-ndarray


if __name__ == '__main__':
  absltest.main()
