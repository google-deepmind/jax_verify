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

"""Algorithms for branch/bound minimisation by partitioning activation space.

Each of the algorithms provided here maintains the current partitioning,
i.e. the set of leaves of the branching decision tree. This is represented
as a priority queue, such that the leaf (partition) with the highest value
of the output bound to be minimised appears first.

The algorithms iteratively branch by splitting the highest priority partition.
On each new sub-partition, its output bound and next branching point are
recomputed.
"""
import functools
import heapq
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from absl import logging
import jax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import intersection
from jax_verify.src.branching import branch_selection
import numpy as np


Tensor = jnp.ndarray
Index = bound_propagation.Index
Nest = bound_propagation.Nest
BranchPlane = branch_selection.BranchPlane
BranchingDecisionList = branch_selection.BranchingDecisionList
BranchEvaluationFn = Callable[[
    BranchingDecisionList,
], Tuple[
    Tensor,  # Upper bound on output.
    Optional[BranchPlane],  # Next branch plane.
]]
BoundFn = Callable[[
    BranchingDecisionList,
], Tuple[
    Tensor,  # Upper bound on output.
    Dict[Index, Tuple[Tensor, Tensor]],  # All intermediate bounds.
]]
BranchScoringFn = Callable[[
    Dict[Index, Tuple[Tensor, Tensor]],  # All intermediate bounds.
], Tuple[
    Dict[Index, Tensor],  # Scores for each neuron.
    Optional[Dict[Index, Tensor]],  # Branch values for each neuron (default 0).
]]


def upper_bound_with_branching(
    transform: bound_propagation.BoundTransform,
    branch_neuron_selector: branch_selection.BranchNeuronSelector,
    spec_fn: Callable[..., Nest[Tensor]],
    *input_bounds: Nest[bound_propagation.GraphInput],
    num_branches: int,
    stop_at_negative_ub: bool = False,
) -> Tensor:
  """Applies the given bound prop method repeatedly with branching.

  Args:
    transform: Forward bound propagation method to apply.
    branch_neuron_selector: Strategy for selecting branch planes.
    spec_fn: Function for which to find an upper bound.
    *input_bounds: Bounds on inputs to `spec_fn`.
    num_branches: Number of branches to perform. The number of leaves will be
      one more than this.
    stop_at_negative_ub: Early stop the branch and bound if the upper bound
      becomes negative.

  Returns:
    Component-wise upper-bound on output of `spec_fn`.
  """
  root_bounds = bound_utils.BoundRetriever(bound_utils.VacuousBoundTransform())
  bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(root_bounds),
      spec_fn, *input_bounds)

  def bound_fn(branching_decisions: BranchingDecisionList):
    branch_constraint = bound_utils.FixedBoundApplier(
        branch_selection.concrete_branch_bounds(
            root_bounds.concrete_bounds, branching_decisions))
    branch_bounds = bound_utils.BoundRetriever(
        intersection.IntersectionBoundTransform(transform, branch_constraint))
    output_bound, _ = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(branch_bounds),
        spec_fn, *input_bounds)
    return output_bound.upper, {
        'intermediate_bounds': branch_bounds.concrete_bounds,
    }

  branch_scoring_fn = functools.partial(
      branch_neuron_selector.score_neurons,
      branch_neuron_selector.analyse_inputs(spec_fn, *input_bounds))

  return run_branching(
      functools.partial(evaluate_branch, bound_fn, branch_scoring_fn),
      num_branches, stop_at_negative_ub)[-1]


def run_branching(
    branch_evaluation_fn: BranchEvaluationFn,
    num_branches: int,
    stop_at_negative_ub: bool = False,
) -> Tensor:
  """Use branch-and-bound to find a tight upper bound for the network.

  Args:
    branch_evaluation_fn: Callable computing an upper bound and determining the
      next branch plane, given a branch specified by decision list.
    num_branches: Number of branches to perform. The number of leaves will be
      one more than this.
    stop_at_negative_ub: Early stop the branch and bound if the upper bound
      becomes negative.

  Returns:
    Tensor of shape [num_branches, batch_size, spec_size] containing, at each
    branching iteration, upper bounds for each example in the batch, associated
    with the worst leaf branch.
  """
  # Maintain branches (leaves of the branching tree) as a priority queue,
  # enabling us to branch on the cases with highest Lagrangian first.
  all_branches = BranchQueue()
  upper_bounds = []

  def push_branch(branching_decisions: BranchingDecisionList):
    bound, next_branch_plane = branch_evaluation_fn(branching_decisions)
    all_branches.push(bound, next_branch_plane, args=(branching_decisions,))

  push_branch([])
  upper_bounds.append(all_branches.max_bounds())

  for _ in range(num_branches):
    if not all_branches.can_pop():
      break  # The worst-scoring branch cannot be split further.
    if stop_at_negative_ub and all_branches.max_bounds().max() < 0.:
      break  # We are early stopping and can already prove robustness.

    next_branch_plane, (branching_decisions,), _ = all_branches.pop()

    # Replace this branching choice with two new ones.
    push_branch(branching_decisions + [next_branch_plane.lower_branch])
    push_branch(branching_decisions + [next_branch_plane.upper_branch])

    upper_bounds.append(all_branches.max_bounds())

  return jnp.array(upper_bounds)


def evaluate_branch(
    bound_fn: BoundFn,
    branch_scoring_fn: BranchScoringFn,
    branching_decisions: BranchingDecisionList,
) -> Tuple[Tensor, Optional[BranchPlane]]:
  """Evaluates a branch by first computing its intermediate bounds.

  Use `functools.partial` to bind the `bound_fn` and `branch_scoring_fn` args,
  to make this conform to `BranchEvaluationFn`.

  Args:
    bound_fn: A function that takes as input branching decisions and input
      bounds and will compute final and intermediate bounds.
    branch_scoring_fn: A function that takes as input information produced by
      the bounding function and will return a score for all neurons, indicating
      which ones to branch on. This will typically be based on
      `branch_selection.BranchNeuronSelector.score_neurons`.
    branching_decisions: Specifies the branch in question.

  Returns:
    Upper bounds for each example in the batch, associated with the branch.
  """
  # Obtain the bound for this set of branching decisions, and get the next
  # branches to investigate.
  bound, branch_heuristics = bound_fn(branching_decisions)

  # Obtain the scores for all the branching decisions.
  branching_scores, branch_vals = branch_scoring_fn(branch_heuristics)
  next_branch_plane = branch_selection.highest_score_branch_plane(
      branching_scores, branch_vals)

  return bound, next_branch_plane


def evaluate_branch_with_filtering(
    full_bound_fn: BoundFn,
    appx_bound_fn: BoundFn,
    branch_scoring_fn: BranchScoringFn,
    branching_decisions: BranchingDecisionList,
    *,
    num_filtered_neurons: int,
    bound_agg: str,
) -> Tuple[Tensor, Optional[BranchPlane]]:
  """Evaluates a branch by filtering via approximate intermediate bounds.

  Use `functools.partial` to bind the `full_bound_fn`, `appx_bound_fn`, and
  `branch_scoring_fn` args, to make this conform to `BranchEvaluationFn`.

  Args:
    full_bound_fn: A function that takes as input branching decisions and input
      bounds and will compute final and intermediate bounds.
    appx_bound_fn: Efficient approximation of `full_bound_fn`, used to compute
      final branching scores for filtered neurons.
    branch_scoring_fn: A function that takes as input information produced by
      the bounding function and will return a score for all neurons, indicating
      which ones to branch on. This will typically be based on
      `branch_selection.BranchNeuronSelector.score_neurons`.
    branching_decisions: Specifies the branch in question.
    num_filtered_neurons: number of filtered neurons per layer.
    bound_agg: string indicating how to aggregate bounds for upper and lower
      branches that are computed via `appx_bound_fn`.

  Returns:
    Upper bounds for each example in the batch, associated with the worst leaf
    branch
  """
  # Obtain the bound for this set of branching decisions, and get the next
  # branches to investigate.
  bound, intermediate_bounds = full_bound_fn(branching_decisions)

  # Obtain the initial scores for all the branching decisions, which are
  # used to filter promising ones.
  init_branching_scores, branch_vals = branch_scoring_fn(intermediate_bounds)
  next_branch_plane = branch_selection.highest_score_branch_plane(
      init_branching_scores, branch_vals)

  # Compute scores for all filtered neurons and select the one for branching
  best_branch_score = jnp.inf
  for index in init_branching_scores:
    flat_scores = jnp.reshape(init_branching_scores[index], [-1])
    neuron_indices = jnp.argsort(-flat_scores)
    for neuron_index in neuron_indices[:num_filtered_neurons]:
      curr_branch_plane = branch_selection.make_branch_plane(
          index, neuron_index, branch_vals)

      lower_branch_bound, _ = appx_bound_fn(
          branching_decisions + [curr_branch_plane.lower_branch])

      upper_branch_bound, _ = appx_bound_fn(
          branching_decisions + [curr_branch_plane.upper_branch])

      # TODO: Need to handle correctly the case where the
      # suggested branching decision is infeasible, which means that
      # the appx_bound_fn call will return -jnp.inf.
      # According to the heuristic, this would mean that this is a very
      # good branching decision, while in practice, this is quite poor.
      curr_branch_score = branch_selection.aggregate_ambiguities(bound_agg)(
          jnp.max(lower_branch_bound),
          jnp.max(upper_branch_bound))

      if curr_branch_score < best_branch_score:
        best_branch_score = curr_branch_score
        next_branch_plane = curr_branch_plane

  return bound, next_branch_plane


class BranchQueue:
  """Set of branch leaves, organised as a priority queue."""

  def __init__(self, log=True):
    self._all_branches = []
    self._j = 0
    self._log = log

  def push(
      self,
      bounds: Tensor,
      next_branch_plane: Optional[BranchPlane],
      *,
      args: Optional[Sequence[Any]] = None,
      jax_args: Optional[Sequence[Any]] = None):
    """Pushes the branch to the priority queue.

    Args:
      bounds: Per-example upper bound for this branch.
      next_branch_plane: Neuron to split for the next branching.
      args: Auxiliary arguments to store with the branch.
      jax_args: Auxiliary arguments to store with the branch that are jax
        objects. We're separating them so that we can hold them off device.
    """
    args = args or ()
    jax_args = jax_args or ()
    # Move tensors off the device.
    # There may be many branches, and there would be a risk of running out of
    # device memory if every entry in the queue held their tensors on-device.
    bounds, next_branch_plane, jax_args = _device_pop(
        (bounds, next_branch_plane, jax_args))
    if self._log:
      logging.info('Branch %d: %s', self._j, ''.join(
          f'[Sample {idx}] Bound {np.amax(bd):.04f} '
          for idx, bd in enumerate(bounds)))
    # Queue entries are tuples beginning with `(priority, j, ...)`,
    # where `j` is a unique identifier in case priorities are equal.
    priority = -np.max(bounds)
    entry = priority, self._j, bounds, next_branch_plane, args, jax_args
    heapq.heappush(self._all_branches, entry)
    self._j += 1

  def can_pop(self) -> bool:
    """Returns whether the branch with the highest Lagrangian can be split."""
    top_branch = self._all_branches[0]
    next_branch_plane = top_branch[3]
    return next_branch_plane is not None

  def pop(self) -> Tuple[
      BranchPlane,
      Sequence[Any],
      Sequence[Any]]:
    """Pops the branch with the highest upper bound (averaged over batch).

    Returns:
      next_branch_plane: Neuron to split for the next branching.
      args: Auxiliary arguments passed to `push`.
      jax_args: Auxiliary arguments passed to `push` that are jax objects.
    """
    entry = heapq.heappop(self._all_branches)
    _, j, bounds, next_branch_plane, args, jax_args = entry
    if self._log:
      logging.info('Splitting branch %d: %s', j, ''.join(
          f'[Sample {idx}] Bound {np.amax(bd):.04f} '
          for idx, bd in enumerate(bounds)))
    return next_branch_plane, args, jax_args

  def max_bounds(self) -> Tensor:
    """Returns the effective upper bound: the worst case over all leaves."""
    # Extract bounds from each (priority, j, bounds, ...) entry.
    bounds = jnp.array([branch[2] for branch in self._all_branches])
    return jnp.amax(bounds, axis=0)


def _device_pop(x: Nest[jnp.ndarray]) -> Nest[np.ndarray]:
  """Moves a tensor (or nest thereof) off the device.

  Any other references to the tensor will become invalid.

  Args:
    x: Nest of Jax tensors.
  Returns:
    Nest of numpy tensors corresponding to `x`.
  """
  x_np = jax.device_get(x)
  jax.tree_map(lambda x: x.delete() if isinstance(x, jnp.ndarray) else None, x)
  return x_np
