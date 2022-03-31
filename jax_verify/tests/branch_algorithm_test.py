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

"""Tests for branching."""

import functools

from absl.testing import absltest
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import ibp
from jax_verify.src.branching import branch_algorithm
from jax_verify.src.branching import branch_selection


class BranchAlgorithmTest(chex.TestCase):

  def test_identity_network_leaves_sensitivities_unchanged(self):
    # Set up a small network.
    @hk.transform
    def forward_fn(x):
      x = hk.Linear(7)(x)
      x = jax.nn.relu(x)
      x = hk.Linear(5)(x)
      return x

    input_bounds = jax_verify.IntervalBound(
        lower_bound=jnp.array([-1., 0., 1.]),
        upper_bound=jnp.array([2., 3., 4.]))

    params = forward_fn.init(jax.random.PRNGKey(0), input_bounds.lower)
    spec_fn = functools.partial(forward_fn.apply, params, None)

    upper_bound = branch_algorithm.upper_bound_with_branching(
        ibp.bound_transform,
        branch_selection.ReluSelector(),
        spec_fn,
        input_bounds,
        num_branches=5)
    chex.assert_equal((5,), upper_bound.shape)


if __name__ == '__main__':
  absltest.main()
