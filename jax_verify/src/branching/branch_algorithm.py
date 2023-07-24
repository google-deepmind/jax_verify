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
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import jax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import graph_traversal
from jax_verify.src import intersection
from jax_verify.src.branching import branch_selection
from jax_verify.src.types import Index, Nest, SpecFn, Tensor  # pylint: disable=g-multiple-import
import numpy as np


BoundsOnMax = Tuple[Tensor, Tensor]
BranchEvaluationFn = Callable[[
    branch_selection.BranchingDecisionList,
], Tuple[
    BoundsOnMax,
    Optional[branch_selection.BranchPlane],  # Next branch plane.
]]
JittableBranchEvaluationFn = Callable[[
    branch_selection.JittableBranchingDecisions,
], Tuple[
    Tuple[Tensor, Tensor],  # Bounds on the branch.
    Tuple[
        branch_selection.JittableBranchingDecisions,  # Next lower branch.
        branch_selection.JittableBranchingDecisions,  # Next upper branch.
    ],
]]
BoundFn = Callable[[
    branch_selection.BranchingDecisionList,
], Tuple[
    Tensor,  # Upper bound on output.
    Mapping[Index, Tuple[Tensor, Tensor]],  # All intermediate bounds.
]]
BranchScoringFn = Callable[[
    Mapping[Index, Tuple[Tensor, Tensor]],  # All intermediate bounds.
], Tuple[
    Mapping[Index, Tensor],  # Scores for each neuron.
    Optional[Mapping[Index, Tensor]],  # Branch values for each neuron (or 0).
]]


def upper_bound_with_branching(
    transform: bound_propagation.BoundTransform,
    branch_neuron_selector: branch_selection.BranchNeuronSelector,
    spec_fn: SpecFn,
    *input_bounds: Nest[graph_traversal.GraphInput],
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

  def bound_fn(branching_decisions: branch_selection.BranchingDecisionList):
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

  _, all_upper_bounds, _ = run_branching(
      functools.partial(evaluate_branch, bound_fn, branch_scoring_fn),
      num_branches, stop_at_negative_ub)
  return all_upper_bounds[-1]


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
    with the worst leaf branch, Tensor with number of solved subproblems
  """
  # Maintain branches (leaves of the branching tree) as a priority queue,
  # enabling us to branch on the cases with highest Lagrangian first.
  all_branches = BranchQueue()
  upper_bounds = []

  def push_branch(branching_decisions: branch_selection.BranchingDecisionList):
    bounds, next_branch_plane = branch_evaluation_fn(
        branching_decisions)
    all_branches.push(bounds, next_branch_plane, args=(branching_decisions,))

  push_branch([])
  upper_bounds.append(all_branches.max_bounds())

  for _ in range(num_branches):
    if not all_branches.can_pop():
      break  # The worst-scoring branch cannot be split further.
    if stop_at_negative_ub and (all_branches.max_bounds().max() < 0.
                                or np.amax(all_branches.empirical_max()) > 0.):
      break  # We are early stopping and can already prove/disprove robustness.

    next_branch_plane, (branching_decisions,), _ = all_branches.pop()

    # Replace this branching choice with two new ones.
    push_branch(branching_decisions + [next_branch_plane.lower_branch])
    push_branch(branching_decisions + [next_branch_plane.upper_branch])

    upper_bounds.append(all_branches.max_bounds())

  empirical_max = all_branches.empirical_max()
  batch_size = empirical_max.shape[0]
  return empirical_max, jnp.array(  # pytype: disable=bad-return-type  # jax-ndarray
      upper_bounds), jnp.ones(batch_size) * all_branches.n_subproblems()


def run_batched_branching(
    batch_size: int,
    max_branching_depth: int,
    max_index_depth: int,
    batched_branch_evaluation_fn: JittableBranchEvaluationFn,
    num_evals: int,
    stop_at_negative_ub: bool = False,
) -> Tensor:
  """Use branch-and-bound to find a tight upper bound for the network.

  This method evaluates several sub-domains at once in order to be quicker.

  Args:
    batch_size: How many subdomains to evaluate at the same time.
    max_branching_depth: How many constraints can we impose on a single
      subdomain.
    max_index_depth: What is the maximum level of depth in the index.
    batched_branch_evaluation_fn: Function to evaluate a batch of set of
      branching decision. This takes as input the branching decision in jitted
      form, and returns:
        - (lower bound, upper bound) on the minimum over the subdomain.
        - The next branching decision to perform on this subdomain, in jitted
          form.
   num_evals: How many branching evaluations to perform.
   stop_at_negative_ub: Early stop the branch and bound if the upper bound
     becomes negative or the lower bound becomes positive.
  Returns:
    Tensor of shape [num_branches, batch_size, spec_size] containing, at each
      branching iteration, upper bounds for each example in the batch,
      associated with the worst leaf branch,
      Tensor with number of solved subproblems
  """
  # Maintain branches (leaves of the branching tree) as a priority queue,
  # enabling us to branch on the cases with highest Lagrangian first.
  all_branches = BranchQueue()
  upper_bounds = []

  empty_branching_decisions = branch_selection.branching_decisions_tensors(
      [], max_branching_depth, max_index_depth)
  array_empty_bdecs = jax.tree_map(lambda x: jnp.expand_dims(x, 0),
                                   empty_branching_decisions)

  def push_branches(
      bdec_list: Sequence[branch_selection.BranchingDecisionList]):
    list_bdec_tensors = [branch_selection.branching_decisions_tensors(
        bdecs, max_branching_depth, max_index_depth) for bdecs in bdec_list]
    array_bdecs_tensors = jax.tree_map(
        lambda *bdec_ten: jnp.stack([*bdec_ten], axis=0), *list_bdec_tensors)
    nb_branches_to_eval = len(bdec_list)

    nb_branches_to_pad = batch_size - nb_branches_to_eval
    if nb_branches_to_pad:
      array_padding_bdecs = jax.tree_map(
          lambda x: jnp.repeat(x, nb_branches_to_pad, 0), array_empty_bdecs)
      to_eval_branches = jax.tree_map(lambda x, y: jnp.concatenate([x, y]),
                                      array_bdecs_tensors, array_padding_bdecs)
    else:
      to_eval_branches = array_bdecs_tensors

    ((lower_bounds, upper_bounds),
     (next_lower_branches, next_upper_branches)) = batched_branch_evaluation_fn(
         to_eval_branches)
    # For each existing subdomain, find out which position to insert a new
    # constraint.
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    next_node_indices = [tuple(idx)
                         for idx in next_lower_branches.node_indices.tolist()]
    next_neuron_indices = next_lower_branches.neuron_indices.tolist()
    next_lower_vals = next_lower_branches.branch_vals.tolist()
    next_upper_vals = next_upper_branches.branch_vals.tolist()

    for subprob_idx, parent_bdecs in enumerate(bdec_list):
      bounds = (lower_bounds[subprob_idx], upper_bounds[subprob_idx])

      lower_branch_dec = branch_selection.BranchDecision(
          next_node_indices[subprob_idx], next_neuron_indices[subprob_idx],
          next_lower_vals[subprob_idx], False)
      upper_branch_dec = branch_selection.BranchDecision(
          next_node_indices[subprob_idx], next_neuron_indices[subprob_idx],
          next_upper_vals[subprob_idx], True)

      next_branch_plane = branch_selection.BranchPlane(lower_branch_dec,
                                                       upper_branch_dec)
      all_branches.push(bounds, next_branch_plane, args=(parent_bdecs,))

  push_branches([[]])

  for _ in range(num_evals):
    subdomains = []

    while len(subdomains) < batch_size:
      if not all_branches.can_pop():
        break  # Queue empty / worst scoring branch cannot be split further.
      next_branch_plane, (branching_decisions,), _ = all_branches.pop()
      subdomains.append(branching_decisions + [next_branch_plane.lower_branch])
      subdomains.append(branching_decisions + [next_branch_plane.upper_branch])

    if subdomains:
      push_branches(subdomains)
    else:
      # We don't have any subdomain to evaluate.
      break

    max_bound = all_branches.max_bounds()
    upper_bounds.append(max_bound)
    emp_max = np.amax(all_branches.empirical_max() > 0.)
    if stop_at_negative_ub and (max_bound.max() < 0. or emp_max > 0.):
      # We can disprove robustness, or we can prove robustness for all
      # remaining subdomains.
      break

  empirical_max = all_branches.empirical_max()
  batch_size = empirical_max.shape[0]
  return empirical_max, jnp.array(  # pytype: disable=bad-return-type  # jax-ndarray
      upper_bounds), jnp.ones(batch_size) * all_branches.n_subproblems()


def evaluate_branch(
    bound_fn: BoundFn,
    branch_scoring_fn: BranchScoringFn,
    branching_decisions: branch_selection.BranchingDecisionList,
) -> Tuple[BoundsOnMax, Optional[branch_selection.BranchPlane]]:
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
    bounds_on_max: Lower and upper bounds on the max attainable for each example
      in the batch.
    next_branch_plane: The next branching decision to make.
  """
  # Obtain the bound for this set of branching decisions, and get the next
  # branches to investigate.
  upper_bound, branch_heuristics = bound_fn(branching_decisions)

  # Obtain the scores for all the branching decisions.
  branching_scores, branch_vals = branch_scoring_fn(branch_heuristics)
  next_branch_plane = branch_selection.highest_score_branch_plane(
      branching_scores, branch_vals)

  emp_max_est = branch_heuristics.get(
      'emp_max_est', -jnp.inf * jnp.ones_like(upper_bound))

  return (emp_max_est, upper_bound), next_branch_plane  # pytype: disable=bad-return-type  # jax-ndarray


def evaluate_branch_with_filtering(
    full_bound_fn: BoundFn,
    appx_bound_fn: BoundFn,
    branch_scoring_fn: BranchScoringFn,
    branching_decisions: branch_selection.BranchingDecisionList,
    *,
    num_filtered_neurons: int,
    bound_agg: str,
) -> Tuple[BoundsOnMax, Optional[branch_selection.BranchPlane]]:
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
    bounds_on_max: Lower and upper bounds on the max attainable for each example
      in the batch.
    next_branch_plane: The next branching decision to make.
  """
  # Obtain the bound for this set of branching decisions, and get the next
  # branches to investigate.
  upper_bound, branch_heuristics = full_bound_fn(branching_decisions)

  # Obtain the initial scores for all the branching decisions, which are
  # used to filter promising ones.
  init_branching_scores, branch_vals = branch_scoring_fn(branch_heuristics)
  next_branch_plane = branch_selection.highest_score_branch_plane(
      init_branching_scores, branch_vals)

  # Compute scores for all filtered neurons and select the one for branching
  best_branch_score = jnp.inf
  for index in init_branching_scores:
    flat_scores = jnp.reshape(init_branching_scores[index], [-1])
    neuron_indices = jnp.argsort(-flat_scores)
    for neuron_index in neuron_indices[:num_filtered_neurons]:
      neuron_score = flat_scores[neuron_index]
      if not jnp.isinf(neuron_score):
        curr_branch_plane = branch_selection.make_branch_plane(
            index, neuron_index, branch_vals)

        lower_branch_bound, _ = appx_bound_fn(
            [*branching_decisions, curr_branch_plane.lower_branch])

        upper_branch_bound, _ = appx_bound_fn(
            [*branching_decisions, curr_branch_plane.upper_branch])

        curr_branch_score = branch_selection.aggregate_ambiguities(bound_agg)(
            jnp.max(lower_branch_bound),
            jnp.max(upper_branch_bound))

        if curr_branch_score < best_branch_score:
          best_branch_score = curr_branch_score
          next_branch_plane = curr_branch_plane
      else:
        # This is an infeasible choice, so we do not need to waste time
        # evaluating it.
        # In addition, in case we have poor convergence, it's entirely possible
        # that by bad luck, this would be the best results, in which case we
        # would end up in an infinite loop.
        # (This branch gets picked, nothing gets changed, the bound remain the
        # same and we keep on landing in the same place).
        pass

  emp_max_est = branch_heuristics.get(
      'emp_max_est', -jnp.inf * jnp.ones_like(upper_bound))

  return (emp_max_est, upper_bound), next_branch_plane  # pytype: disable=bad-return-type  # jax-ndarray


class BranchQueue:
  """Set of branch leaves, organised as a priority queue."""

  def __init__(self, log=True, drop_negative_ub=False):
    self._all_branches = []
    self._j = 0
    self._log = log
    self._empirical_max = -np.inf
    self._highest_negative_ub = -np.inf
    self._bound_shape = None
    self._drop_negative_ub = drop_negative_ub

  def push(
      self,
      bounds: BoundsOnMax,
      next_branch_plane: Optional[Union[bool, branch_selection.BranchPlane]],
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
    (emp_max, upper_bound), next_branch_plane, jax_args = _device_pop(
        (bounds, next_branch_plane, jax_args))
    if self._log:
      logging.info('Branch %d: %s', self._j, ''.join(
          f'[Sample {idx}] Bound {np.amax(bd):.04f} '
          for idx, bd in enumerate(upper_bound)))
      if np.amax(emp_max) > np.amax(self._empirical_max):
        logging.info('New Empirical max: %.04f', np.amax(emp_max))
    self._bound_shape = bounds[1].shape
    # Queue entries are tuples beginning with `(priority, j, ...)`,
    # where `j` is a unique identifier in case priorities are equal.
    priority = -np.max(upper_bound)
    entry = priority, self._j, upper_bound, next_branch_plane, args, jax_args
    if (priority < 0.):
      # This is a bound that will have to be split.
      heapq.heappush(self._all_branches, entry)
    elif not self._drop_negative_ub:
      # We are not trying to save memory and are keeping all the bounds.
      heapq.heappush(self._all_branches, entry)
    else:
      # This is a bound that we want to drop, it's a negative upper bound.
      # We'll just update our running tally of what would be the limiting bound
      # when we cross zero.
      self._highest_negative_ub = np.maximum(self._highest_negative_ub,
                                             upper_bound)
    self._j += 1
    # Update the highest value seen.
    self._empirical_max = np.maximum(self._empirical_max, emp_max)

  def can_pop(self) -> bool:
    """Returns whether the branch with the highest Lagrangian can be split."""
    if self._all_branches:
      top_branch = self._all_branches[0]
      next_branch_plane = top_branch[3]
      return next_branch_plane is not None
    else:
      return False

  def pop(self) -> Tuple[
      branch_selection.BranchPlane,
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
    if self._drop_negative_ub and not self._all_branches:
      # We have dropped all the negative bounds, and have run out of new leaves
      # to branch. The remaining bound will be the one that we kept in the
      # running tally of highest negative upper bound.
      return self._highest_negative_ub
    if np.prod(self._bound_shape) == 1:
      # If we have a single output being bounded, we know where it is in the
      # priority queue.
      worst_priority = self._all_branches[0][0]
      return np.reshape(-worst_priority, self._bound_shape)
    else:
      # If there is multiple outputs, then we need to go through the whole queue
      # because otherwise we don't know which is the worst bound for a given
      # coordinate.
      bounds = np.stack(tuple(branch[2] for branch in self._all_branches))
      # Extract bounds from each (priority, j, bounds, ...) entry.
      return np.amax(bounds, axis=0)

  def empirical_max(self) -> Tensor:
    return self._empirical_max

  def n_subproblems(self) -> int:
    return self._j


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
