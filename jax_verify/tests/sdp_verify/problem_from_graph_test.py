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

"""Tests for problem_from_graph.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import jax_verify
from jax_verify.extensions.sdp_verify import problem
from jax_verify.extensions.sdp_verify import problem_from_graph
from jax_verify.extensions.sdp_verify import sdp_verify
from jax_verify.extensions.sdp_verify import utils
from jax_verify.src import ibp
from jax_verify.tests.sdp_verify import test_utils


class SdpProblemTest(parameterized.TestCase):

  def assertArrayAlmostEqual(self, lhs, rhs):
    self.assertEqual(lhs is None, rhs is None)
    if lhs is not None:
      diff = jnp.abs(lhs - rhs).max()
      self.assertAlmostEqual(diff, 0., places=5)

  def test_sdp_problem_equivalent_to_sdp_verify(self):
    # Set up a verification problem for test purposes.
    verif_instance = test_utils.make_toy_verif_instance(label=2, target_label=1)

    # Set up a spec function that replicates the test problem.
    inputs = jnp.zeros((1, 5))
    input_bounds = jax_verify.IntervalBound(
        jnp.zeros_like(inputs), jnp.ones_like(inputs))
    boundprop_transform = ibp.bound_transform
    def spec_fn(x):
      x = utils.predict_mlp(verif_instance.params, x)
      x = jax.nn.relu(x)
      return jnp.sum(
          jnp.reshape(x, (-1,)) * verif_instance.obj) + verif_instance.const

    # Build an SDP verification instance using the code under test.
    sdp_relu_problem = problem_from_graph.SdpReluProblem(
        boundprop_transform, spec_fn, input_bounds)
    sdp_problem_vi = sdp_relu_problem.build_sdp_verification_instance()

    # Build an SDP verification instance using existing `sdp_verify` code.
    sdp_verify_vi = problem.make_sdp_verif_instance(verif_instance)

    self._assert_verif_instances_equal(sdp_problem_vi, sdp_verify_vi)

  def _assert_verif_instances_equal(self, sdp_problem_vi, sdp_verify_vi):
    # Assert that bounds are the same.
    self.assertEqual(len(sdp_problem_vi.bounds), len(sdp_verify_vi.bounds))
    for sdp_problem_bound, sdp_verify_bound in zip(
        sdp_problem_vi.bounds, sdp_verify_vi.bounds):
      self.assertArrayAlmostEqual(sdp_problem_bound.lb, sdp_verify_bound.lb)
      self.assertArrayAlmostEqual(sdp_problem_bound.ub, sdp_verify_bound.ub)

    # Don't compare dual shapes/types in detail, because the different
    # implementations can and do represent them in different
    # (but equivalent) ways.
    # They should have the same length, though.
    self.assertEqual(len(sdp_problem_vi.dual_shapes),
                     len(sdp_verify_vi.dual_shapes))
    self.assertEqual(len(sdp_problem_vi.dual_types),
                     len(sdp_verify_vi.dual_types))

    # Evaluate each problem's dual objective on the same random dual variables.
    def random_dual_fun(verif_instance):
      key = jax.random.PRNGKey(103)
      random_like = lambda x: jax.random.uniform(key, x.shape, x.dtype)
      duals = sdp_verify.init_duals(verif_instance, None)
      duals = jax.tree_map(random_like, duals)
      return sdp_verify.dual_fun(verif_instance, duals)

    self.assertAlmostEqual(
        random_dual_fun(sdp_problem_vi), random_dual_fun(sdp_verify_vi),
        places=5)


if __name__ == '__main__':
  absltest.main()
