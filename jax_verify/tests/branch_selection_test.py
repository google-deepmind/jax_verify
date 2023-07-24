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

"""Tests for branching decisions."""


from absl.testing import absltest
import chex
import jax.numpy as jnp
from jax_verify.src import ibp
from jax_verify.src.branching import branch_selection


class BranchSelectionTest(chex.TestCase):

  def test_jittable_branching_decisions_enforced(self):

    free_bounds = ibp.IntervalBound(-jnp.ones(4,),
                                    jnp.ones(4,))

    layer_index = (1, 2)
    branching_decision_list = [
        # Neuron 0 is greater than 0.
        branch_selection.BranchDecision(layer_index, 0, 0., 1),
        # Neuron 1 is smaller than 0.5
        branch_selection.BranchDecision(layer_index, 1, 0.5, -1),
        # Neuron 2 is between -0.3 and 0.3
        branch_selection.BranchDecision(layer_index, 2, -0.3, 1),
        branch_selection.BranchDecision(layer_index, 2, 0.3, -1),
        # Neuron 3 is below 2., which is a spurious constraint
        branch_selection.BranchDecision(layer_index, 3, 2., -1)
    ]
    branching_decisions_tensors = branch_selection.branching_decisions_tensors(
        branching_decision_list, 3, 8)

    enforced_bounds = branch_selection.enforce_jittable_branching_decisions(
        branching_decisions_tensors, layer_index, free_bounds)

    chex.assert_trees_all_close((enforced_bounds.lower, enforced_bounds.upper),
                                (jnp.array([0., -1., -0.3, -1.]),
                                 jnp.array([1., 0.5, 0.3, 1.])))

    # check that the bounds are not modified when enforced on another layer.
    other_lay_bound = branch_selection.enforce_jittable_branching_decisions(
        branching_decisions_tensors, (1, 3), free_bounds)

    chex.assert_trees_all_close((free_bounds.lower, free_bounds.upper),
                                (other_lay_bound.lower, other_lay_bound.upper))

  def test_infeasible_bounds_detection(self):

    non_crossing_bounds = ibp.IntervalBound(jnp.zeros(3,), jnp.ones(3,))
    crossing_bounds = ibp.IntervalBound(jnp.array([0., 0., 1.]),
                                        jnp.array([1., 1., 0.5]))

    non_crossing_infeasible = branch_selection.infeasible_bounds(
        non_crossing_bounds.to_jittable())
    self.assertFalse(non_crossing_infeasible)

    crossing_infeasible = branch_selection.infeasible_bounds(
        crossing_bounds.to_jittable())
    self.assertTrue(crossing_infeasible)

if __name__ == '__main__':
  absltest.main()
