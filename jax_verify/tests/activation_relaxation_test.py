# coding=utf-8
# Copyright 2021 The jax_verify Authors.
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
import jax.numpy as jnp
from jax_verify.src import activation_relaxation
from jax_verify.src import synthetic_primitives
from jax_verify.tests import test_utils


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

  def test_abs(self):

    batch_size = 5
    axis_dim = 8

    abs_inp_shape = (batch_size, axis_dim)

    def abs_model(inp):
      return jnp.abs(inp)

    bound_key = jax.random.PRNGKey(0)
    inp_lb, inp_ub = test_utils.sample_bounds(bound_key, abs_inp_shape,
                                              minval=-10., maxval=10.)

    lb_fun, ub_fun = activation_relaxation.abs_relaxation(
        inp_lb, inp_ub)

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

    lb_fun, ub_fun = activation_relaxation.leaky_relu_relaxation(
        inp_lb, inp_ub, negative_slope)

    # Check that the bounds are valid
    uniform_check_key = jax.random.PRNGKey(1)
    self._check_bounds(uniform_check_key, leaky_relu_model, lb_fun, ub_fun,
                       [inp_lb], [inp_ub])

    # Sanity check the convexity of the relaxation
    cvx_check_key = jax.random.PRNGKey(2)
    self._check_convexity(cvx_check_key, lb_fun, [inp_lb], [inp_ub], True)
    ccv_check_key = jax.random.PRNGKey(3)
    self._check_convexity(ccv_check_key, ub_fun, [inp_lb], [inp_ub], False)


if __name__ == '__main__':
  absltest.main()
