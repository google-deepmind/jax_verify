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

"""Test for the support of branching constraints."""
from absl.testing import absltest

import jax
from jax import numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import concretization
from jax_verify.src import optimizers
from jax_verify.src import synthetic_primitives
from jax_verify.src.branching import branch_selection
from jax_verify.src.linear import backward_crown
from jax_verify.src.linear import backward_linearbounds_with_branching as blwb
from jax_verify.src.linear import linear_relaxations
from jax_verify.tests import test_utils
import numpy as np

import optax


def _find_linear_layers_indexes(fun, input_bound):
  """Find the index of linear layers.

  We don't require the branching to be on those, but it is easy to reason about
  them so that's what the tests are doing.

  Args:
    fun: Function to be bounded.
    input_bound: Input bound to the function
  Returns:
    linear_layers_indexes: Indexes of all the linear layers.
  """
  graph_inspector = bound_utils.GraphInspector()
  inspector_algorithm = bound_propagation.ForwardPropagationAlgorithm(
      graph_inspector)
  bound_propagation.bound_propagation(inspector_algorithm, fun, input_bound)
  return [node.index for node in graph_inspector.nodes.values()
          if node.primitive == synthetic_primitives.linear_p]


class LinearBoundBranchingTest(absltest.TestCase):

  def test_linear_network(self):
    """Test the imposition of constraints.

    We are going to take a network that is performing the sum of the inputs.
    All inputs are contained between [-1, 1].

    We will impose the constraints that the output is greater than zero, and
    that it is smaller than 2. as branching constraints, and check that this
    will be tighten the bounds.
    """

    def lin_network(x):
      return x.sum()

    input_bound = jax_verify.IntervalBound(-jnp.ones((3,)), jnp.ones((3,)))

    # Let's first check we get the expected bounds.
    unbranched_bounds = jax_verify.backward_crown_bound_propagation(
        lin_network, input_bound)
    self.assertAlmostEqual(unbranched_bounds.lower, -3.)
    self.assertAlmostEqual(unbranched_bounds.upper, 3.)

    lin_layer_indexes = _find_linear_layers_indexes(lin_network, input_bound)
    self.assertLen(lin_layer_indexes, 1)
    layer_index = lin_layer_indexes[0]

    ## First block of test: imposing restrictive constraints.
    tighter_branching_decisions_list = [
        # In layer `layer_index`, neuron 0 is greater than 0.
        branch_selection.BranchDecision(layer_index, 0, 0., 1),
        # In layer `layer_index`, neuron 0 is smaller than 2.
        branch_selection.BranchDecision(layer_index, 0, 2., -1),
    ]
    nb_steps = 100
    slope_ss_schedule = optax.exponential_decay(1e-2, 1, 0.95)
    lag_ss_schedule = optax.exponential_decay(1, 1, 0.95)
    slope_opt = optax.adam(slope_ss_schedule)
    lag_opt = optax.adam(lag_ss_schedule)
    tighter_branching_decisions_tensors = branch_selection.branching_decisions_tensors(
        tighter_branching_decisions_list, 5, 4)
    tighter_branched_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        slope_opt, lag_opt, nb_steps, lin_network,
        tighter_branching_decisions_tensors, input_bound)

    # Let's check that the bounds have improved.
    self.assertGreater(tighter_branched_bounds.lower, unbranched_bounds.lower)
    self.assertLess(tighter_branched_bounds.upper, unbranched_bounds.upper)
    # Let's check that the bounds have not become better than they should be.
    self.assertLessEqual(tighter_branched_bounds.lower, 0.)
    self.assertGreaterEqual(tighter_branched_bounds.upper, 2.)

    ## Second block of test: imposing useless constraints.
    useless_branching_decisions_list = [
        # In layer `layer_index`, neuron 0 is greater than -5.
        branch_selection.BranchDecision(layer_index, 0, -5., 1),
        # In layer `layer_index`, neuron 0 is smaller than 5.
        branch_selection.BranchDecision(layer_index, 0, 5., -1),
    ]
    useless_branching_decisions_tensors = branch_selection.branching_decisions_tensors(
        useless_branching_decisions_list, 5, 4)
    useless_branched_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        slope_opt, lag_opt, nb_steps, lin_network,
        useless_branching_decisions_tensors, input_bound)
    # Let's check that the bounds are still valid.
    self.assertLessEqual(useless_branched_bounds.lower, -3.)
    self.assertGreaterEqual(useless_branched_bounds.upper, 3.)

    ## Third block of test: Verifying that even with a bad optimizer, we keep
    ## valid bounds.
    bad_opt = optax.adam(-10)
    bad_opt_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        bad_opt, bad_opt, nb_steps, lin_network,
        useless_branching_decisions_tensors, input_bound)
    # Let's check that the bounds are still valid.
    self.assertLessEqual(bad_opt_bounds.lower, -3.)
    self.assertGreaterEqual(bad_opt_bounds.upper, 3.)

  def test_relu_network(self):
    """Test the imposition of constraints in an intermediate ReLU layer.

    We're going to design the network in a way to be able to reason easily
    about bounds and branching decisions.

    2 inputs: x, y
    4 hidden units: (x+y, x+y, -x-y, -x-y)
    The four hidden units go through a ReLU
    Final layer weights of (1, -1, 1, -1)

    The output of the network is always 0, and the hidden units are all
    correlated. However, if you do a convex or linear relaxation of the network,
    you are not going to get tight bounds (the ReLU will have to be relaxed, and
    given that they have different signs on their output, some will be set at
    their lower bound while some will be set at their upper bound, making the
    bound non zero).
    """

    def relu_net(inp):
      lin_1_weight = jnp.array([[1., -1.],
                                [1., -1.],
                                [-1., 1.],
                                [-1., 1.]])
      lin_2_weight = jnp.array([[1., -1., 1., -1.]])

      return lin_2_weight @ (jax.nn.relu(lin_1_weight @ inp))

    input_bound = jax_verify.IntervalBound(-jnp.ones((2,)),
                                           jnp.ones((2,)))
    # Let's first check we get the expected bounds.
    unbranched_bounds = jax_verify.backward_crown_bound_propagation(
        relu_net, input_bound)
    self.assertGreater(unbranched_bounds.upper, 1.)
    self.assertLess(unbranched_bounds.lower, -1.)

    lin_layer_indexes = _find_linear_layers_indexes(relu_net, input_bound)
    self.assertLen(lin_layer_indexes, 2)
    ini_layer_index = lin_layer_indexes[0]

    upper_branching_decisions_list = [
        # In layer `ini_layer_index`, neuron 0 is greater than 0.
        branch_selection.BranchDecision(ini_layer_index, 0, 0., 1),
    ]
    # As we impose this constraint, this should ideally impose also the
    # constaint that the second neuron is greater than 0. too (they have the
    # same coefficients.) Similarly, it should force the others neurons (3 and
    # 4) to be smaller than 0.
    # As a result, all the ReLUs would be fixed, which means that there would be
    # no looseness, and the bound verification method should obtain a tight
    # bound.
    nb_steps = 100
    slope_ss_schedule = optax.exponential_decay(1e-2, 1, 0.95)
    lag_ss_schedule = optax.exponential_decay(1, 1, 0.95)
    slope_opt = optax.adam(slope_ss_schedule)
    lag_opt = optax.adam(lag_ss_schedule)
    upper_branching_decisions_tensors = branch_selection.branching_decisions_tensors(
        upper_branching_decisions_list, 3, 4)
    upper_branched_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        slope_opt, lag_opt, nb_steps, relu_net,
        upper_branching_decisions_tensors, input_bound)

    self.assertAlmostEqual(upper_branched_bounds.lower, 0., delta=1e-4)
    self.assertAlmostEqual(upper_branched_bounds.upper, 0., delta=1e-4)

    # Let's also impose the lower equivalent constraint (as if we were doing a
    # branch-and bound process.)
    lower_branching_decisions_list = [
        # In layer `ini_layer_index`, neuron 0 is smaller than 0.
        branch_selection.BranchDecision(ini_layer_index, 0, 0., -1),
    ]
    lower_branching_decisions_tensors = branch_selection.branching_decisions_tensors(
        lower_branching_decisions_list, 3, 4)
    lower_branched_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        slope_opt, lag_opt, nb_steps, relu_net,
        lower_branching_decisions_tensors, input_bound)
    self.assertAlmostEqual(lower_branched_bounds.lower, 0., delta=1e-4)
    self.assertAlmostEqual(lower_branched_bounds.upper, 0., delta=1e-4)

    # Let's also test out what happens when we have unsatisfiable constraints
    # that we add. In practice, this should result in the lower bounds going to
    # +infinity and the upper bounds going to -infinity.
    # This can be observed by them crossing.
    impossible_branching_decisions_list = [
        # In layer `layer_index`, neuron 0 (x-y) is greater than 1.
        branch_selection.BranchDecision(ini_layer_index, 0, 1., 1),
        # In layer `layer_index`, neuron 1 (x-y) is smaller than -1.
        branch_selection.BranchDecision(ini_layer_index, 1, -1., -1),
    ]
    impossible_branching_decisions_tensors = branch_selection.branching_decisions_tensors(
        impossible_branching_decisions_list, 2, 4)
    nb_steps_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        slope_opt, lag_opt, nb_steps, relu_net,
        impossible_branching_decisions_tensors, input_bound)

    nb_steps_times_2_bounds = blwb.lagrangian_backward_linear_compute_bounds(
        slope_opt, lag_opt, 2 * nb_steps, relu_net,
        impossible_branching_decisions_tensors, input_bound)

    # Observe that the bounds are crossing.
    self.assertGreater(nb_steps_bounds.lower, nb_steps_bounds.upper)
    self.assertGreater(nb_steps_times_2_bounds.lower,
                       nb_steps_times_2_bounds.upper)
    # Make sure that the bounds are diverging.
    self.assertGreater(nb_steps_times_2_bounds.lower, nb_steps_bounds.lower)
    self.assertLess(nb_steps_times_2_bounds.upper, nb_steps_bounds.upper)

  def test_nobranch_noimpact(self):
    """When we impose no branching decisions, we should match alpha-crown."""
    architecture = [16, 8, 8, 2]
    problem_key = jax.random.PRNGKey(42)
    fun, (lb, ub) = test_utils.set_up_toy_problem(problem_key, 2, architecture)

    nb_steps = 100
    slope_ss_schedule = optax.exponential_decay(1e-2, 1, 0.95)
    lag_ss_schedule = optax.exponential_decay(1, 1, 0.95)
    slope_opt = optax.adam(slope_ss_schedule)
    lag_opt = optax.adam(lag_ss_schedule)
    slope_optimizer = optimizers.OptaxOptimizer(slope_opt, num_steps=nb_steps)
    branch_optimizer = optimizers.OptaxOptimizer(
        blwb.slope_and_lagrangian_optimizer(slope_opt, lag_opt),
        num_steps=nb_steps)

    reference_transform = backward_crown.OptimizingLinearBoundBackwardTransform(
        linear_relaxations.parameterized_relaxer,
        backward_crown.CONCRETIZE_ARGS_PRIMITIVE, slope_optimizer)
    reference_concretizer = concretization.ChunkedBackwardConcretizer(
        reference_transform, max_chunk_size=0)
    reference_algorithm = concretization.BackwardConcretizingAlgorithm(
        reference_concretizer)

    branching_decisions_list = []
    branching_decisions_tensor = branch_selection.branching_decisions_tensors(
        branching_decisions_list, 2, 4)
    nobranch_transform = blwb.BranchedOptimizingLinearBoundBackwardTransform(
        branching_decisions_tensor, linear_relaxations.parameterized_relaxer,
        backward_crown.CONCRETIZE_ARGS_PRIMITIVE, branch_optimizer)
    nobranch_concretizer = concretization.ChunkedBackwardConcretizer(
        nobranch_transform, max_chunk_size=0)
    nobranch_algorithm = concretization.BackwardConcretizingAlgorithm(
        nobranch_concretizer)

    inp_bound = jax_verify.IntervalBound(lb, ub)
    reference_bound, _ = bound_propagation.bound_propagation(
        reference_algorithm, fun, inp_bound)
    nobranch_bound, _ = bound_propagation.bound_propagation(
        nobranch_algorithm, fun, inp_bound)

    np.testing.assert_array_equal(reference_bound.upper, nobranch_bound.upper)
    np.testing.assert_array_equal(reference_bound.lower, nobranch_bound.lower)


if __name__ == '__main__':
  absltest.main()
