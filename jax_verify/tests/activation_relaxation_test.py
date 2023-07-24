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

"""Test the convex relaxation of different primitives."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

import cvxpy as cp
import jax
from jax import lax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import simplex_bound
from jax_verify.src import synthetic_primitives
from jax_verify.tests import test_utils
import numpy as np


IntervalBound = bound_propagation.IntervalBound


TOL = 1e-5


class ConvexRelaxationTest(parameterized.TestCase):

  def _sample_from_bound(self, rng_key, bound, nb_points):
    if isinstance(bound, simplex_bound.SimplexIntervalBound):
      return test_utils.sample_bounded_simplex_points(
          rng_key, (bound.lower, bound.upper), bound.simplex_sum,
          nb_points)
    else:
      return test_utils.sample_bounded_points(
          rng_key, (bound.lower, bound.upper), nb_points)

  def _check_bounds(self, key, fun, lb_fun, ub_fun, bounds, nb_samples=1000):
    """Check that lb_fun and ub_fun actually bound the function fun.

    This is evaluated at a number of random samples.

    Args:
      key: PRNG key for random number generation
      fun: Function to be bounded.
      lb_fun: Lower bound function.
      ub_fun: Upper bound function.
      bounds: List of bounds on the inputs.
      nb_samples: How many random samples to draw for testing.
    """
    keys = jax.random.split(key, len(bounds))

    # Build the uniform samples.
    inps = []
    for inp_idx, bound in enumerate(bounds):
      inps.append(self._sample_from_bound(keys[inp_idx], bound, nb_samples))
    vmap_fun = jax.vmap(fun)
    vmap_lbfun = jax.vmap(lb_fun)
    vmap_ubfun = jax.vmap(ub_fun)

    samples_eval = vmap_fun(*inps)
    lb_eval = vmap_lbfun(*inps)
    ub_eval = vmap_ubfun(*inps)

    self.assertGreaterEqual(
        (samples_eval - lb_eval).min(), -TOL,
        msg='Lower Bound is invalid')
    self.assertGreaterEqual(
        (ub_eval - samples_eval).min(), -TOL,
        msg='Upper Bound is invalid')

  def _check_convexity(self, key, fun, bounds, is_convex, nb_samples=100):
    """Check that the function is convex or concave.

    We do this by sanity-checking that the function is below its chord.

    Args:
      key: PRNG key for random number generation
      fun: Function to be checked.
      bounds: List of bounds on the inputs.
      is_convex: Boolean, if True:  check that the function is convex.
                          if False: check that the function is concave.
      nb_samples: How many random samples to draw for testing.
    """
    keys = jax.random.split(key, 2*len(bounds) + 1)

    a_inps = []
    b_inps = []
    interp_inps = []
    interp_coeffs = jax.random.uniform(keys[-1], (nb_samples,))
    for inp_idx, bound in enumerate(bounds):
      interp_coeffs_shape = (-1,) + (1,)*bound.lower.ndim
      broad_interp_coeffs = jnp.reshape(interp_coeffs, interp_coeffs_shape)

      a_inp = self._sample_from_bound(keys[2*inp_idx], bound, nb_samples)
      b_inp = self._sample_from_bound(keys[2*inp_idx+1], bound, nb_samples)
      interp_inp = (a_inp * broad_interp_coeffs +
                    b_inp * (1. - broad_interp_coeffs))
      a_inps.append(a_inp)
      b_inps.append(b_inp)
      interp_inps.append(interp_inp)

    vmap_fun = jax.vmap(fun)

    a_eval = vmap_fun(*a_inps)
    b_eval = vmap_fun(*b_inps)
    interp_eval = vmap_fun(*interp_inps)
    interp_coeffs_shape = (-1,) + (1,)*(interp_eval.ndim - 1)
    broad_interp_coeffs = jnp.reshape(interp_coeffs, interp_coeffs_shape)
    chord_eval = (a_eval * broad_interp_coeffs +
                  b_eval * (1. - broad_interp_coeffs))

    if is_convex:
      self.assertGreaterEqual(
          (chord_eval - interp_eval).min(), -TOL,
          msg='Function is not convex')
    else:
      self.assertGreaterEqual(
          (interp_eval - chord_eval).min(), -TOL,
          msg='Function is not concave')


class DefaultConvexRelaxationTest(ConvexRelaxationTest):

  def test_abs(self):

    batch_size = 5
    axis_dim = 8

    abs_inp_shape = (batch_size, axis_dim)

    def abs_model(inp):
      return jnp.abs(inp)

    bound_key = jax.random.PRNGKey(0)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, abs_inp_shape,
                                              minval=-10., maxval=10.)
    inp_bound = IntervalBound(inp_lb, inp_ub)
    lb_fun, ub_fun = activation_relaxation.convex_fn_relaxation(
        lax.abs_p, inp_bound)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, abs_model, lb_fun, ub_fun,
                       [inp_bound])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_bound], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_bound], False)

  @parameterized.named_parameters(
      ('pos_smaller_than_1', 0.5),
      ('pos_higher_than_1', 1.5),
      ('neg', -1.))
  def test_leaky_relu(self, negative_slope):
    batch_size = 5
    axis_dim = 8

    leaky_relu_inp_shape = (batch_size, axis_dim)

    def leaky_relu_model(inp):
      return jax.nn.leaky_relu(inp, negative_slope)

    bound_key = jax.random.PRNGKey(0)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, leaky_relu_inp_shape,
                                              minval=-10., maxval=10.)
    inp_bound = IntervalBound(inp_lb, inp_ub)

    lb_fun, ub_fun = activation_relaxation.intersection_relaxation(
        activation_relaxation.leaky_relu_piecewise_linear_relaxation,
        inp_bound, negative_slope=negative_slope)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, leaky_relu_model, lb_fun, ub_fun,
                       [inp_bound])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_bound], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_bound], False)

  @parameterized.named_parameters(
      ('small_scale', 0.01),
      ('normal_scale', 1),
      ('large_scale', 1e4),
      ('very_large_scale', 1e8))
  def test_sigmoid(self, scale):
    batch_size = 5
    axis_dim = 8

    sigmoid_inp_shape = (batch_size, axis_dim)
    sigmoid = jax.nn.sigmoid

    bound_key = jax.random.PRNGKey(0)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, sigmoid_inp_shape,
                                              minval=-scale, maxval=scale)
    inp_bound = IntervalBound(inp_lb, inp_ub)

    lb_fun, ub_fun = activation_relaxation.sigmoid_relaxation(inp_bound)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, sigmoid, lb_fun, ub_fun,
                       [inp_bound])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_bound], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_bound], False)

  def test_sigmoid_parallel_different_setting(self):
    # This is a reproduction of a bug that was identified where we were
    # incorrectly computing the tangent point, when we were, in the same batch
    # of point, having points where the upper bound was fully linear, fully
    # sigmoid, or a mix.
    lower = jnp.array([-100., 1., -3.])
    upper = jnp.array([-1., 100., 3.])
    inp_bound = IntervalBound(lower, upper)

    lb_fun, ub_fun = activation_relaxation.sigmoid_relaxation(inp_bound)

    # At the edge of the feasible domain, the convex relaxation should be tight.
    at_lower_lb_gap = jnp.abs(lb_fun(lower) - jax.nn.sigmoid(lower))
    at_lower_ub_gap = jnp.abs(ub_fun(lower) - jax.nn.sigmoid(lower))

    at_upper_lb_gap = jnp.abs(lb_fun(upper) - jax.nn.sigmoid(upper))
    at_upper_ub_gap = jnp.abs(ub_fun(upper) - jax.nn.sigmoid(upper))

    self.assertAlmostEqual(at_lower_lb_gap.max(), 0.)
    self.assertAlmostEqual(at_lower_ub_gap.max(), 0.)
    self.assertAlmostEqual(at_upper_lb_gap.max(), 0.)
    self.assertAlmostEqual(at_upper_ub_gap.max(), 0.)

  @parameterized.named_parameters(
      ('small_scale', 0.01),
      ('normal_scale', 1),
      ('large_scale', 1e4),
      ('very_large_scale', 1e8))
  def test_tanh(self, scale):
    batch_size = 5
    axis_dim = 8

    tanh_inp_shape = (batch_size, axis_dim)

    bound_key = jax.random.PRNGKey(0)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, tanh_inp_shape,
                                              minval=-scale, maxval=scale)
    inp_bound = IntervalBound(inp_lb, inp_ub)

    lb_fun, ub_fun = activation_relaxation.tanh_relaxation(inp_bound)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, jnp.tanh, lb_fun, ub_fun,
                       [inp_bound])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_bound], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_bound], False)

  @parameterized.named_parameters(
      ('all_included', 0.1),
      ('mixed', 2.),
      ('large_scale', 1e4),
      ('very_large_scale', 1e8))
  def test_clip(self, scale):
    batch_size = 5
    axis_dim = 8

    clip_inp_shape = (batch_size, axis_dim)
    clip_fun = functools.partial(jnp.clip, a_min=-1., a_max=1.)

    bound_key = jax.random.PRNGKey(0)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, clip_inp_shape,
                                              minval=-scale, maxval=scale)
    inp_bound = IntervalBound(inp_lb, inp_ub)

    clip_relaxation = activation_relaxation.relaxation_fns[
        synthetic_primitives.clip_p]
    lb_fun, ub_fun = clip_relaxation.relaxation_fn(
        inp_bound, a_min=-1., a_max=1.)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, clip_fun, lb_fun, ub_fun, [inp_bound])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_bound], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_bound], False)

  def test_fusedrelu(self):
    inp_dim = 5
    out_dim = 7

    param_key = jax.random.PRNGKey(0)
    weight_key, bias_key = jax.random.split(param_key, 2)
    lin_layer_weight = jax.random.normal(weight_key, (inp_dim, out_dim))
    lin_layer_bias = jax.random.normal(bias_key, (out_dim,))

    bound_key = jax.random.PRNGKey(1)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, (inp_dim,),
                                              minval=-1., maxval=1.)

    def linear_layer(inp, lin_weight, lin_bias):
      return inp @ lin_weight + lin_bias

    def fused_relu_model(inp, lin_weight, lin_bias, *_):
      return jax.nn.relu(linear_layer(inp, lin_weight, lin_bias))

    # Let's get the jaxpr corresponding to the function, similarly to what would
    # be extracted by the synthetic primitives simplifier.
    parsed = synthetic_primitives.make_jaxpr_nojit(
        fused_relu_model, inp_lb, lin_layer_weight, lin_layer_bias)
    inp_is_bound = {var: is_bound for var, is_bound
                    in zip(parsed.jaxpr.invars, [True, False, False])}
    simplified_graph = synthetic_primitives.simplify_graph(
        synthetic_primitives.fused_relu_simplifier, parsed.jaxpr, inp_is_bound)

    linear_eqn = simplified_graph.eqns[0]
    assert linear_eqn.primitive == synthetic_primitives.linear_p
    relu_eqn = simplified_graph.eqns[1]
    assert relu_eqn.primitive == synthetic_primitives.fused_relu_p

    net_inp = IntervalBound(inp_lb, inp_ub)

    linear_bound = jax_verify.interval_bound_propagation(
        linear_layer, net_inp, lin_layer_weight, lin_layer_bias)

    lb_fun, ub_fun = activation_relaxation.fused_relu_relaxation(
        linear_bound, net_inp, lin_layer_weight, lin_layer_bias,
        **relu_eqn.params)

    # Check that the bounds are valid
    def tied_inp_lb_fun(lin_inp, lin_weight, lin_bias):
      lin_out = linear_layer(lin_inp, lin_weight, lin_bias)
      return lb_fun(lin_out, lin_inp, lin_weight, lin_bias)

    def tied_inp_ub_fun(lin_inp, lin_weight, lin_bias):
      lin_out = linear_layer(lin_inp, lin_weight, lin_bias)
      return ub_fun(lin_out, lin_inp, lin_weight, lin_bias)

    all_inp_bounds = [net_inp,
                      IntervalBound(lin_layer_weight, lin_layer_weight),
                      IntervalBound(lin_layer_bias, lin_layer_bias)]
    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(2)
    self._check_bounds(
        uniform_check_key, fused_relu_model, tied_inp_lb_fun, tied_inp_ub_fun,
        all_inp_bounds)

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(3)
    self._check_convexity(cvx_check_key, tied_inp_lb_fun, all_inp_bounds, True)
    ccv_check_key = jax.random.PRNGKey(4)
    self._check_convexity(ccv_check_key, tied_inp_ub_fun, all_inp_bounds, False)

  def test_fusedrelu_conv(self):
    height = 5
    width = 5
    inp_channels = 3
    out_channels = 4
    ker_size = 2

    img_shape = (1, inp_channels, height, width)
    ker_shape = (out_channels, inp_channels, ker_size, ker_size)
    bias_shape = (1, out_channels, 1, 1)

    param_key = jax.random.PRNGKey(0)
    weight_key, bias_key = jax.random.split(param_key, 2)
    lin_kernel_weight = jax.random.normal(weight_key, ker_shape)
    lin_kernel_bias = jax.random.normal(bias_key, bias_shape)

    bound_key = jax.random.PRNGKey(1)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, img_shape,
                                              minval=-1., maxval=1.)

    def linear_layer(inp, lin_kernel, lin_bias):
      return jax.lax.conv(inp, lin_kernel, (1, 1), 'SAME') + lin_bias

    def fused_relu_model(inp, lin_kernel, lin_bias):
      return jax.nn.relu(linear_layer(inp, lin_kernel, lin_bias))

    # Let's get the jaxpr corresponding to the function, similarly to what would
    # be extracted by the synthetic primitives simplifier.
    parsed = synthetic_primitives.make_jaxpr_nojit(
        fused_relu_model, inp_lb, lin_kernel_weight, lin_kernel_bias)
    inp_is_bound = {var: is_bound for var, is_bound
                    in zip(parsed.jaxpr.invars, [True, False, False])}
    simplified_graph = synthetic_primitives.simplify_graph(
        synthetic_primitives.fused_relu_simplifier, parsed.jaxpr, inp_is_bound)

    linear_eqn = simplified_graph.eqns[0]
    assert linear_eqn.primitive == synthetic_primitives.linear_p
    relu_eqn = simplified_graph.eqns[1]
    assert relu_eqn.primitive == synthetic_primitives.fused_relu_p

    net_inp = IntervalBound(inp_lb, inp_ub)

    linear_bound = jax_verify.interval_bound_propagation(
        linear_layer, net_inp, lin_kernel_weight, lin_kernel_bias)

    lb_fun, ub_fun = activation_relaxation.fused_relu_relaxation(
        linear_bound, net_inp, lin_kernel_weight, lin_kernel_bias,
        **relu_eqn.params)

    # Check that the bounds are valid
    def tied_inp_lb_fun(lin_inp, lin_kernel, lin_bias):
      lin_out = linear_layer(lin_inp, lin_kernel, lin_bias)
      return lb_fun(lin_out, lin_inp, lin_kernel, lin_bias)

    def tied_inp_ub_fun(lin_inp, lin_kernel, lin_bias):
      lin_out = linear_layer(lin_inp, lin_kernel, lin_bias)
      return ub_fun(lin_out, lin_inp, lin_kernel, lin_bias)

    all_inp_bounds = [net_inp,
                      IntervalBound(lin_kernel_weight, lin_kernel_weight),
                      IntervalBound(lin_kernel_bias, lin_kernel_bias)]
    uniform_check_key = jax.random.PRNGKey(2)
    self._check_bounds(
        uniform_check_key, fused_relu_model, tied_inp_lb_fun, tied_inp_ub_fun,
        all_inp_bounds)

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(3)
    self._check_convexity(cvx_check_key, tied_inp_lb_fun, all_inp_bounds, True)
    ccv_check_key = jax.random.PRNGKey(4)
    self._check_convexity(ccv_check_key, tied_inp_ub_fun, all_inp_bounds, False)

  def test_equivalent_hypercube_fusedrelu_relaxation(self):
    inp_dim = 10
    out_dim = 50

    param_key = jax.random.PRNGKey(0)
    weight_key, bias_key = jax.random.split(param_key, 2)
    lin_layer_weight = jax.random.normal(weight_key, (inp_dim, out_dim))
    lin_layer_bias = jax.random.normal(bias_key, (out_dim,))

    bound_key = jax.random.PRNGKey(1)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, (inp_dim,),
                                              minval=-1., maxval=1.)

    ub_fun = activation_relaxation.alt_fused_relu_hypercube_upper_bound(
        inp_lb, inp_ub)

    alt_ub_fun, _ = activation_relaxation.fused_relu_hypercube_upper_bound(
        inp_lb, inp_ub)

    all_neuron_ub_fun = functools.partial(jax.vmap(ub_fun,
                                                   in_axes=(1, 0, None)),
                                          lin_layer_weight, lin_layer_bias)
    all_neuron_alt_ub_fun = functools.partial(jax.vmap(alt_ub_fun,
                                                       in_axes=(1, 0, None)),
                                              lin_layer_weight, lin_layer_bias)

    batch_ub_fun = jax.vmap(all_neuron_ub_fun)
    batch_alt_ub_fun = jax.vmap(all_neuron_alt_ub_fun)

    samples_key = jax.random.PRNGKey(2)
    samples = test_utils.sample_bounded_points(samples_key, (inp_lb, inp_ub),
                                               nb_points=256, axis=0)

    ub_out = batch_ub_fun(samples)
    alt_ub_out = batch_alt_ub_fun(samples)

    max_diff = jnp.abs(ub_out - alt_ub_out).max()
    self.assertAlmostEqual(max_diff, 0., delta=1e-5)

if __name__ == '__main__':
  absltest.main()
