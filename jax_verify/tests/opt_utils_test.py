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

"""Tests for jax_verify opt_utils."""
import functools
from absl.testing import absltest

import chex
import jax
from jax import numpy as jnp
from jax_verify.src import opt_utils
import numpy as np


class OptUtilsTest(absltest.TestCase):

  def test_greedy_assign(self):

    # Build a list of upper bounds, sum, and the expected greedy assginment.
    problems = [
        (0.5 * jnp.ones(5,), 2.5, 0.5 * jnp.ones(5,)),
        (0.5 * jnp.ones(5,), 1.0, jnp.array([0.5, 0.5, 0., 0., 0.])),
        (0.5 * jnp.ones(5,), 0.75, jnp.array([0.5, 0.25, 0., 0., 0.])),
        (0.5 * jnp.ones(5,), 0.3, jnp.array([0.3, 0., 0., 0., 0.])),
        (jnp.array([0., 1., 0., 0.5]), 1.2, jnp.array([0., 1., 0., 0.2])),
        (jnp.array([1., 2., 3.]), 2.5, jnp.array([1., 1.5, 0.]))
    ]

    for upper, total_sum, ref_answer in problems:
      # Try the forward assignment.
      pred = opt_utils.greedy_assign(upper, total_sum)
      chex.assert_trees_all_close(pred, ref_answer)

  def test_1d_binary_search(self):
    for seed in range(10):
      argmax = jax.random.uniform(jax.random.PRNGKey(seed), ())

      # Try out two possible types of concave function for which we know the
      # maximum.
      ccv_fun = lambda x, argmax=argmax: -(x - argmax)**2
      pred_argmax, max_val = opt_utils.concave_1d_max(
          ccv_fun, jnp.zeros(()), jnp.ones(()), num_steps=64)
      self.assertAlmostEqual(max_val, 0., delta=1e-6)  # pytype: disable=wrong-arg-types  # jax-ndarray
      self.assertAlmostEqual(pred_argmax, argmax, delta=1e-6)

      alt_ccv_fun = lambda x, argmax=argmax: -jnp.abs(x - argmax)
      pred_argmax, max_val = opt_utils.concave_1d_max(
          alt_ccv_fun, jnp.zeros(()), jnp.ones(()), num_steps=64)
      self.assertAlmostEqual(max_val, 0., delta=1e-6)  # pytype: disable=wrong-arg-types  # jax-ndarray
      self.assertAlmostEqual(pred_argmax, argmax, delta=1e-6)

    x, y = opt_utils.concave_1d_max(
        lambda x: -x**2 + 4.*x - 3.,  # max at x=2, y=1
        jnp.array([0., -11., 10.]),
        jnp.array([3., -10., 11.]),
    )
    np.testing.assert_array_almost_equal(x, np.array([2., -10., 10.]),
                                         decimal=3)
    np.testing.assert_array_almost_equal(y, np.array([1., -143., -63.]),
                                         decimal=4)

  def test_simplex_projection_fully_constrained(self):
    # Test the edge case of an simplex sum with one element.
    # This should always give the simplex_sum if it's in the valid bounds.
    all_initial_values = jnp.expand_dims(jnp.linspace(-10., 10., 100), 1)

    project_onto_01 = functools.partial(opt_utils.project_onto_interval_simplex,
                                        jnp.zeros((1,)), jnp.ones((1,)),
                                        1.0)
    batch_project_onto_01 = jax.vmap(project_onto_01)
    all_res = batch_project_onto_01(all_initial_values)

    self.assertAlmostEqual(all_res.min(), 1.0, delta=1e-6)
    self.assertAlmostEqual(all_res.max(), 1.0, delta=1e-6)

    project_onto_03 = functools.partial(opt_utils.project_onto_interval_simplex,
                                        jnp.zeros((1,)), 3*jnp.ones((1,)),
                                        1.0)
    batch_project_onto_03 = jax.vmap(project_onto_03)
    all_res = batch_project_onto_03(all_initial_values)
    self.assertAlmostEqual(all_res.min(), 1.0, delta=1e-6)
    self.assertAlmostEqual(all_res.max(), 1.0, delta=1e-6)

    key = jax.random.PRNGKey(0)
    initial_values = jax.random.uniform(key, (100, 5), minval=-10, maxval=10)
    # There is only one valid solution to this problem: everything is 1.
    project = functools.partial(opt_utils.project_onto_interval_simplex,
                                jnp.zeros((5,)), jnp.ones((5,)), 5.0)
    batch_project = jax.vmap(project)
    all_res = batch_project(initial_values)
    self.assertAlmostEqual(all_res.min(), 1.0, delta=1e-6)
    self.assertAlmostEqual(all_res.max(), 1.0, delta=1e-6)


if __name__ == '__main__':
  absltest.main()
