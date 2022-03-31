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

"""Tests for linear relaxations, both fixed and parameterised."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
from jax_verify.src.linear import linear_relaxations


class LinearRelaxationsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('exp', lax.exp_p, jnp.exp, 0.),
      ('softplus', synthetic_primitives.softplus_p, jax.nn.softplus, 0.),
      ('posreciprocal',
       synthetic_primitives.posreciprocal_p, lambda x: 1./x, 2.),
  )
  def test_convex_fn_relaxes_with_upper_chord(self, primitive, fn, shift):
    parameterized_relaxer = linear_relaxations.parameterized_relaxer
    linearizer = parameterized_relaxer.parameterized_linearizer(
        (), primitive, [4])
    chex.assert_equal(linearizer.arity, 1)

    lb = jnp.array([0., -1., -1., -1.]) + shift
    ub = jnp.array([1., 1.1, 0., 1.]) + shift
    input_bounds = bound_propagation.IntervalBound(lb, ub)
    relax_params = [jnp.array([.3, .4, 1., .5])], ()

    chex.assert_trees_all_equal_shapes(
        linearizer.initial_params(input_bounds), relax_params)
    relax_params = linearizer.project_params(relax_params)
    lower, upper = linearizer.linearize(relax_params, input_bounds)

    # Both bounds should be linear.
    lower_grad = _elementwise_grad(lower)
    upper_grad = _elementwise_grad(upper)
    mid = (lb + ub) / 2.
    chex.assert_trees_all_close(lower_grad(lb), lower_grad(mid), lower_grad(ub))
    chex.assert_trees_all_close(upper_grad(lb), upper_grad(mid), upper_grad(ub))

    # Lower bound should be a supporting hyperplane.
    x = jnp.array([.3, -.16, 0., 0.]) + shift
    fn_grad = _elementwise_grad(fn)
    chex.assert_trees_all_close(lower(x), fn(x))
    chex.assert_trees_all_close(lower_grad(x), fn_grad(x))

    # Upper bound should be a chord.
    chex.assert_trees_all_close(upper(lb), fn(lb))
    chex.assert_trees_all_close(upper(ub), fn(ub))

  @parameterized.named_parameters(
      ('sigmoid', synthetic_primitives.sigmoid_p,
       activation_relaxation.sigmoid_relaxation),
  )
  def test_smooth_fn_relaxes_with_convex_relaxation(
      self, primitive, convex_relaxation):
    parameterized_relaxer = linear_relaxations.parameterized_relaxer
    linearizer = parameterized_relaxer.parameterized_linearizer(
        (), primitive, [4])
    chex.assert_equal(linearizer.arity, 1)

    lb = jnp.array([0., -1., -1., -1.])
    ub = jnp.array([1., 1.1, 0., 1.])
    input_bounds = bound_propagation.IntervalBound(lb, ub)
    relax_params = (
        [jnp.array([.3, .4, 1., .5])],
        [jnp.array([.4, .5, 0., .6])])

    chex.assert_trees_all_equal_shapes(
        linearizer.initial_params(input_bounds), relax_params)
    relax_params = linearizer.project_params(relax_params)
    lower, upper = linearizer.linearize(relax_params, input_bounds)

    # Obtain the convex bounds. We can trust that these are valid, as they
    # have been tested in `activation_relaxation_test.py`.
    mu, eta = convex_relaxation(input_bounds)

    # Both bounds should be linear.
    lower_grad = _elementwise_grad(lower)
    upper_grad = _elementwise_grad(upper)
    mid = (lb + ub) / 2.
    chex.assert_trees_all_close(lower_grad(lb), lower_grad(mid), lower_grad(ub))
    chex.assert_trees_all_close(upper_grad(lb), upper_grad(mid), upper_grad(ub))

    # Lower bound should be a supporting hyperplane of the convex lower bound.
    x = jnp.array([.3, -.16, 0., 0.])
    mu_grad = _elementwise_grad(mu)
    chex.assert_trees_all_close(lower(x), mu(x))
    chex.assert_trees_all_close(lower_grad(x), mu_grad(x))

    # Upper bound should be a supporting hyperplane of the concave upper bound.
    x = jnp.array([.4, .05, -1., .2])
    eta_grad = _elementwise_grad(eta)
    chex.assert_trees_all_close(upper(x), eta(x))
    chex.assert_trees_all_close(upper_grad(x), eta_grad(x))

  @parameterized.named_parameters(
      ('relu', synthetic_primitives.relu_p, jax.nn.relu, {}),
      ('leaky_relu', synthetic_primitives.leaky_relu_p, jax.nn.leaky_relu,
       {'negative_slope': .2}),
      ('abs', lax.abs_p, jnp.abs, {}),
  )
  def test_relu_relaxes_with_upper_chord_and_subgradient(
      self, primitive, fn, params):
    parameterized_relaxer = linear_relaxations.parameterized_relaxer
    linearizer = parameterized_relaxer.parameterized_linearizer(
        (), primitive, [6], **params)
    chex.assert_equal(linearizer.arity, 1)

    lb = jnp.array([0., -1., -1., -1., -3.5, 2.1])
    ub = jnp.array([1., 1.1, 0., 1., -2.5, 3.6])
    input_bounds = bound_propagation.IntervalBound(lb, ub)
    relax_params = jnp.array([.3, .4, 1., .5, .6, .7]), ()

    chex.assert_trees_all_equal_shapes(
        linearizer.initial_params(input_bounds), relax_params)
    relax_params = linearizer.project_params(relax_params)
    lower, upper = linearizer.linearize(relax_params, input_bounds)

    # Both bounds should be linear.
    lower_grad = _elementwise_grad(lower)
    upper_grad = _elementwise_grad(upper)
    mid = (lb + ub) / 2.
    chex.assert_trees_all_close(lower_grad(lb), lower_grad(mid), lower_grad(ub))
    chex.assert_trees_all_close(upper_grad(lb), upper_grad(mid), upper_grad(ub))

    # Lower bound should be a subgradient at the origin.
    z = jnp.zeros_like(lb)
    chex.assert_trees_all_close(lower(z), fn(z, **params))
    for x in (lb, ub):
      chex.assert_trees_all_close(
          lower(x) < fn(x, **params) + 1.e-6,
          jnp.ones_like(lb, dtype=jnp.bool_))

    # Upper bound should be a chord.
    chex.assert_trees_all_close(upper(lb), fn(lb, **params))
    chex.assert_trees_all_close(upper(ub), fn(ub, **params))


def _elementwise_grad(f):
  return jax.grad(lambda x: jnp.sum(f(x)))


if __name__ == '__main__':
  absltest.main()
