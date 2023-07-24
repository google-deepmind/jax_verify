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

"""Tests for backpropagation of sensitivity values."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src.branching import backpropagation


class BackpropagationTest(chex.TestCase):

  def test_identity_network_leaves_sensitivities_unchanged(self):
    # Set up an identity network.
    def logits_fn(x):
      return x
    input_bounds = jax_verify.IntervalBound(
        lower_bound=jnp.array([-1., 0., 1.]),
        upper_bound=jnp.array([2., 3., 4.]))

    # Backpropagation
    output_sensitivities = jnp.array([.1, .2, -.3])
    sensitivity_computation = backpropagation.SensitivityAlgorithm(
        jax_verify.ibp_transform, [(0,)], output_sensitivities)
    bound_propagation.bound_propagation(sensitivity_computation,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                        logits_fn, input_bounds)
    input_sensitivities, = sensitivity_computation.target_sensitivities

    chex.assert_trees_all_close(input_sensitivities, jnp.array([.1, .2, -.3]))

  def test_relu_network_applies_chord_slopes_to_sensitivities(self):
    # Set up some ReLUs, with a variety of input bounds:
    # 1 blocking, 1 passing, and 3 'ambiguous' (straddling zero).
    def logits_fn(x):
      return jax.nn.relu(x)
    input_bounds = jax_verify.IntervalBound(
        lower_bound=jnp.array([-2., 1., -1., -4., -2.]),
        upper_bound=jnp.array([-1., 2., 1., 1., 3.]))

    # Backpropagation.
    output_sensitivities = jnp.array([10., 10., 10., 10., 10.])
    sensitivity_computation = backpropagation.SensitivityAlgorithm(
        jax_verify.ibp_transform, [(0,)], output_sensitivities)
    bound_propagation.bound_propagation(sensitivity_computation,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                        logits_fn, input_bounds)
    input_sensitivities, = sensitivity_computation.target_sensitivities

    # Expect blocking neurons to have no sensitivity, passing neurons to have
    # full sensitivity, and ambiguous neurons to interpolate between the two.
    chex.assert_trees_all_close(
        input_sensitivities, jnp.array([0., 10., 5., 2., 6.]))

  def test_affine_network_applies_transpose_to_sensitivites(self):
    # Set up a matmul with bias.
    w = jnp.array([[1., 4., -5.], [2., -3., 6.]])
    b = jnp.array([20., 30., 40.])
    def logits_fn(x):
      return x @ w + b

    input_bounds = jax_verify.IntervalBound(
        lower_bound=jnp.zeros(shape=(1, 2)),
        upper_bound=jnp.ones(shape=(1, 2)))

    # Backpropagation.
    output_sensitivities = jnp.array([[1., 0., -1.]])
    sensitivity_computation = backpropagation.SensitivityAlgorithm(
        jax_verify.ibp_transform, [(0,)], output_sensitivities)
    bound_propagation.bound_propagation(sensitivity_computation,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                        logits_fn, input_bounds)
    input_sensitivities, = sensitivity_computation.target_sensitivities
    # Expect the transpose of w to have been applied to the sensitivities.
    # The bias will be ignored.
    chex.assert_trees_all_close(
        input_sensitivities, jnp.array([[6., -4.]]))


if __name__ == '__main__':
  absltest.main()
