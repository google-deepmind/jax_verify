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

"""Test the convex relaxation of different primitives."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
from jax_verify.tests import test_utils


IntervalBound = bound_propagation.IntervalBound


TOL = 1e-5


class ConvexRelaxationTest(parameterized.TestCase):

  def _check_bounds(self, key, fun, lb_fun, ub_fun, lbs, ubs, nb_samples=1000):
    """Check that lb_fun and ub_fun actually bound the function fun.

    This is evaluated at a number of random samples.

    Args:
      key: PRNG key for random number generation
      fun: Function to be bounded.
      lb_fun: Lower bound function
      ub_fun: Upper bound function
      lbs: List of lower bounds on the inputs
      ubs: List of upper bounds on the inputs
      nb_samples: How many random samples to draw for testing.
    """
    assert len(lbs) == len(ubs)
    keys = jax.random.split(key, len(lbs))

    # Build the uniform samples.
    inps = []
    for inp_idx, bounds in enumerate(zip(lbs, ubs)):
      inps.append(test_utils.sample_bounded_points(
          keys[inp_idx], bounds, nb_samples))
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

  def _check_convexity(self, key, fun, lbs, ubs, is_convex, nb_samples=100):
    """Check that the function is convex or concave.

    We do this by sanity-checking that the function is below its chord.

    Args:
      key: PRNG key for random number generation
      fun: Function to be checked.
      lbs: List of lower bounds on the inputs
      ubs: List of upper bounds on the inputs
      is_convex: Boolean, if True:  check that the function is convex.
                          if False: check that the function is concave.
      nb_samples: How many random samples to draw for testing.
    """
    assert len(lbs) == len(ubs)
    keys = jax.random.split(key, 2*len(lbs) + 1)

    a_inps = []
    b_inps = []
    interp_inps = []
    interp_coeffs = jax.random.uniform(keys[-1], (nb_samples,))
    for inp_idx, bounds in enumerate(zip(lbs, ubs)):
      interp_coeffs_shape = (-1,) + (1,)*bounds[0].ndim
      broad_interp_coeffs = jnp.reshape(interp_coeffs, interp_coeffs_shape)

      a_inp = test_utils.sample_bounded_points(keys[2*inp_idx], bounds,
                                               nb_samples)
      b_inp = test_utils.sample_bounded_points(keys[2*inp_idx + 1], bounds,
                                               nb_samples)
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

    lb_fun, ub_fun = activation_relaxation.convex_fn_relaxation(
        lax.abs_p, IntervalBound(inp_lb, inp_ub))

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, abs_model, lb_fun, ub_fun,
                       [inp_lb], [inp_ub])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_lb], [inp_ub], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_lb], [inp_ub], False)

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

    lb_fun, ub_fun = activation_relaxation.intersection_relaxation(
        activation_relaxation.leaky_relu_piecewise_linear_relaxation,
        IntervalBound(inp_lb, inp_ub), negative_slope=negative_slope)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, leaky_relu_model, lb_fun, ub_fun,
                       [inp_lb], [inp_ub])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_lb], [inp_ub], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_lb], [inp_ub], False)

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

    lb_fun, ub_fun = activation_relaxation.sigmoid_relaxation(
        IntervalBound(inp_lb, inp_ub))

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, sigmoid, lb_fun, ub_fun,
                       [inp_lb], [inp_ub])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_lb], [inp_ub], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_lb], [inp_ub], False)

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

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(2)
    self._check_bounds(
        uniform_check_key, fused_relu_model, tied_inp_lb_fun, tied_inp_ub_fun,
        [inp_lb, lin_layer_weight, lin_layer_bias],
        [inp_ub, lin_layer_weight, lin_layer_bias])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(3)
    self._check_convexity(
        cvx_check_key, tied_inp_lb_fun,
        [inp_lb, lin_layer_weight, lin_layer_bias],
        [inp_ub, lin_layer_weight, lin_layer_bias], True)
    ccv_check_key = jax.random.PRNGKey(4)
    self._check_convexity(
        ccv_check_key, tied_inp_ub_fun,
        [inp_lb, lin_layer_weight, lin_layer_bias],
        [inp_ub, lin_layer_weight, lin_layer_bias], False)

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

    uniform_check_key = jax.random.PRNGKey(2)
    self._check_bounds(
        uniform_check_key, fused_relu_model, tied_inp_lb_fun, tied_inp_ub_fun,
        [inp_lb, lin_kernel_weight, lin_kernel_bias],
        [inp_ub, lin_kernel_weight, lin_kernel_bias])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(3)
    self._check_convexity(
        cvx_check_key, tied_inp_lb_fun,
        [inp_lb, lin_kernel_weight, lin_kernel_bias],
        [inp_ub, lin_kernel_weight, lin_kernel_bias], True)
    ccv_check_key = jax.random.PRNGKey(4)
    self._check_convexity(
        ccv_check_key, tied_inp_ub_fun,
        [inp_lb, lin_kernel_weight, lin_kernel_bias],
        [inp_ub, lin_kernel_weight, lin_kernel_bias], False)


if __name__ == '__main__':
  absltest.main()
