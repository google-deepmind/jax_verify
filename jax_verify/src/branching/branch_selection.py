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

"""Strategies for selecting the neuron on which to branch."""

import abc
import enum
import functools
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src.branching import backpropagation
from jax_verify.src.linear import linear_relaxations


Tensor = jnp.ndarray
Nest = bound_propagation.Nest
TensorFun = Callable[..., Tensor]
Index = bound_propagation.Index
Bound = bound_propagation.Bound
GraphInput = bound_propagation.GraphInput
InputBound = bound_propagation.InputBound
BranchDecision = NamedTuple('BranchDecision', [
    ('node_index', Index),
    ('neuron_index', int),
    ('branch_val', float),
    ('is_upper', int),
])
BranchPlane = NamedTuple('BranchPlane', [
    ('lower_branch', BranchDecision),
    ('upper_branch', BranchDecision)
])
BranchingDecisionList = List[BranchDecision]
JittableBranchingDecisions = NamedTuple('JittableBranchingDecisions', [
    ('node_indices', Tensor),
    ('neuron_indices', Tensor),
    ('branch_vals', Tensor),
    ('is_upper_branch', Tensor)])
InputAnalysis = Any
ScoringInputs = Any
BranchHeuristicInputs = Dict[str, Any]
BranchVal = Union[Tensor, Tuple[Tensor, Tensor]]
JittableInputBound = bound_propagation.JittableInputBound

relaxation_area = lambda lb, ub: -ub * lb
mid_point = lambda lb, ub: (lb + ub) / 2.0


class AnalysisType(enum.Enum):
  RELU_INDICES = 'relu_indices'
  SENSITIVITY = 'sensitivity'
  BOUND_INFO = 'bound_info'
  GRAD_SENSITIVITY = 'grad_sensitivity'


class BranchNeuronSelector(metaclass=abc.ABCMeta):
  """Strategy for selecting the neuron on which to branch."""

  @abc.abstractmethod
  def analyse_inputs(
      self,
      spec_fn: TensorFun,
      *init_bound: Nest[GraphInput],
  ) -> InputAnalysis:
    """Performs pre-computation for an input.

    The output of this will remain the same throughout the verification
    procedure, and not be dependent on the current state of the branch and
    bound algorithm.

    Args:
      spec_fn: Network under verification, including specification (objective).
      *init_bound: Interval bounds on the input tensor(s).

    Returns:
      input_analysis: Pre-computed input-dependent data.
    """

  @abc.abstractmethod
  def score_neurons(
      self,
      input_analysis: InputAnalysis,
      heuristic_inputs: BranchHeuristicInputs,
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    """Compute branching scores for all neurons.

    Args:
      input_analysis: Result of the analysis of the inputs.
      heuristic_inputs: Statistics depending on the current state of the branch
        and bound tree, based on which the branching decision needs to be made.
        This will for example contain the intermediate bounds of the networks,
        but may contain additional information.
    Returns:
      scores: Score for each graph node.
      branch_vals: Critical values of all neurons for each graph node. If `None`
        then simply use zero for all critical values.
    """


def selector_scoring_fn(
    spec_fn: TensorFun,
    branch_neuron_selector: BranchNeuronSelector,
    heuristic_inputs: BranchHeuristicInputs,
    *jit_input_bounds: Nest[JittableInputBound]
) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
  """Score the neurons in a jittable fashion for a BranchNeuronSelector.

  The best way to use this function is to bind the first two arguments at the
  beginning, and then to jit it, so as to have the neuron scoring be jittable.

  Args:
    spec_fn: Function whose bounds we are trying to obtain.
    branch_neuron_selector: Strategy that we use for branching.
    heuristic_inputs: Statistics depending on the current state of the branch
      and bound tree.
    *jit_input_bounds: Bounds on the inputs.
  Returns:
    scores_and_branch_vals: Score and branching values for each possible
      branching decision throughout the network.
  """
  input_bounds = bound_propagation.unjit_inputs(*jit_input_bounds)
  input_analysis = branch_neuron_selector.analyse_inputs(
      spec_fn, *input_bounds)
  return branch_neuron_selector.score_neurons(input_analysis, heuristic_inputs)


class NodeSelector(BranchNeuronSelector, metaclass=abc.ABCMeta):
  """Component capable of handling branching for a specific type of Nodes."""

  @abc.abstractmethod
  def analysis_type(self) -> AnalysisType:
    """Returns the type of analysis required by the Selector.

    The value returned from this function will be used to determine what the
    inputs to `score_handled_nodes` should be.
    Every NodeSelector that returns the same value will receive the same inputs
    to compute the scores with. This allows to share computation between
    different NodeSelector that have similar requirements, rather than
    recomputing the `preprocess_heurisitics` several times.
    """

  def score_neurons(
      self,
      input_analysis: InputAnalysis,
      heuristic_inputs: BranchHeuristicInputs,
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    score_inps = self.preprocess_heuristics(input_analysis, heuristic_inputs)
    return self.score_handled_nodes(score_inps)

  def preprocess_heuristics(
      self,
      input_analysis: InputAnalysis,
      heuristic_inputs: BranchHeuristicInputs,
  ) -> ScoringInputs:
    """Performs preprocessing of the analysis.

    This may be shared by several NodeSelector if they have the same type.
    Through the intermediate bounds that are given, this will depend on the
    current node in the branch and bound tree that is being branched on.

    By default, if this function is not overloaded, it will return exactly its
    arguments.

    Args:
      input_analysis: Result of the analysis of the inputs.
      heuristic_inputs: Statistics depending on the current state of the branch
        and bound tree, based on which the branching decision needs to be made.
        This will for example contain the intermediate bounds of the networks,
        but may contain additional information.
    Returns:
      scoring_inputs: Inputs that the score_handled_nodes function require.
    """
    return input_analysis, heuristic_inputs

  @abc.abstractmethod
  def score_handled_nodes(
      self,
      score_inputs: ScoringInputs,
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    """Returns the branching decisions as well as a score for each of them."""


class CompositeNodeSelector(BranchNeuronSelector):
  """Combine NodeSelectors that propose branches on different network parts."""

  def __init__(self,
               selectors: Sequence[NodeSelector],
               coefficients: Sequence[float]):
    super().__init__()
    self._selectors = selectors
    self._coefficients = coefficients

  def analyse_inputs(self, spec_fn: Callable[..., Tensor],
                     *init_bounds) -> InputAnalysis:
    input_analyses = {}
    for selector in self._selectors:
      if selector.analysis_type() not in input_analyses:
        input_analyses[selector.analysis_type()] = selector.analyse_inputs(
            spec_fn, *init_bounds)
    return input_analyses

  def score_neurons(
      self,
      input_analysis: InputAnalysis,
      heuristic_inputs: BranchHeuristicInputs,
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    weigh_scores = lambda weight, x: weight * x

    scoring_inputs = {}
    scores = {}
    branch_vals = {}
    for selector, coeff in zip(self._selectors, self._coefficients):
      analysis_type = selector.analysis_type()
      if selector.analysis_type not in scoring_inputs:
        scoring_inputs[analysis_type] = selector.preprocess_heuristics(
            input_analysis[analysis_type], heuristic_inputs)

      selector_scores, selector_branch_vals = selector.score_handled_nodes(
          scoring_inputs[analysis_type])

      mul_by_coeff = functools.partial(weigh_scores, coeff)
      weighted_selector_scores = jax.tree_map(mul_by_coeff,
                                              selector_scores)
      if set(scores) & set(weighted_selector_scores):
        raise ValueError('Several branching heuristics are operating on the'
                         'same nodes.')

      scores.update(weighted_selector_scores)
      branch_vals.update(selector_branch_vals)

    return scores, branch_vals


class ReluSelector(NodeSelector):
  """Strategy for selecting the neuron on which to branch."""

  def analysis_type(self) -> AnalysisType:
    return AnalysisType.RELU_INDICES

  def analyse_inputs(
      self,
      spec_fn: Callable[..., Tensor],
      *init_bound: Nest[GraphInput],
  ) -> InputAnalysis:
    """Performs pre-computation for an input."""
    graph_nodes = computation_graph_nodes(spec_fn, *init_bound)
    relu_preact_indices = primitive_preact_indices(
        graph_nodes, [synthetic_primitives.relu_p])
    if not relu_preact_indices:
      raise ValueError('Cannot branch because this network has no ReLUs.')
    return relu_preact_indices

  def score_handled_nodes(
      self,
      scoring_inputs: ScoringInputs
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    """Compute ambiguity scores for ReLU pre-activation."""
    preact_indices, heuristic_inputs = scoring_inputs
    bounds = heuristic_inputs['intermediate_bounds']

    def masked_ambiguity(lower: Tensor, upper: Tensor) -> Tensor:
      ambiguous = (lower < 0.) & (upper > 0.)
      return jnp.where(ambiguous, relaxation_area(lower, upper), -jnp.inf)

    return (
        {index: masked_ambiguity(*bounds[index]) for index in preact_indices},
        None,
    )


class BoundInfoSelector(NodeSelector):
  """Branching strategies specialised depending on the type of Node."""

  def analysis_type(self) -> AnalysisType:
    return AnalysisType.BOUND_INFO

  def analyse_inputs(
      self,
      spec_fn: Callable[..., Tensor],
      *init_bound: Nest[GraphInput],
  ) -> InputAnalysis:
    """Performs pre-computation for an input."""
    return bound_input_info(*init_bound)


class InputSelector(BoundInfoSelector):
  """Strategy for selecting the input dimension on which to branch."""

  def __init__(
      self,
      ambiguity: Callable[[Tensor, Tensor], Tensor]):
    super().__init__()

    def unfixed_ambiguity(lower: Tensor, upper: Tensor) -> Tensor:
      unfixed = upper != lower
      return jnp.where(unfixed, ambiguity(lower, upper), -jnp.inf)

    self._ambiguity = unfixed_ambiguity

  def score_handled_nodes(
      self,
      scoring_inputs: ScoringInputs,
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, Tensor]]]:
    """Compute ambiguity scores for inputs."""
    input_analysis, heuristic_inputs = scoring_inputs
    input_indices = input_analysis['input_indices']
    bounds = heuristic_inputs['intermediate_bounds']

    return (
        {index: self._ambiguity(*bounds[index]) for index in input_indices},
        {index: mid_point(*bounds[index]) for index in input_indices},
    )


class SensitivitySelector(NodeSelector, metaclass=abc.ABCMeta):
  """Neuron selection strategy based on estimated influence on output objective.
  """

  def __init__(
      self,
      ambiguity_agg: str = 'min',
      sensitivity_of_max: bool = True):
    """Initialises the neuron selector for a given network.

    Args:
      ambiguity_agg: Method ('min', 'max', 'avg') for combining the two
        possible ambiguity measures of a ReLU relaxation. The two possibilities
        arise from the choice of triangle: its vertices are (lb,0), (ub,ub),
        and _either_ (ub,0) _or_ (lb,lb).
      sensitivity_of_max: Whether to compute the neuron sensitivity with respect
        to the worst bound. By default, will compute the sensitivity with
        respect to all outputs.
    """
    super().__init__()
    self._ambiguity_agg = ambiguity_agg
    self._sensitivity_of_max = sensitivity_of_max

  def analysis_type(self) -> AnalysisType:
    return AnalysisType.SENSITIVITY

  def analyse_inputs(
      self,
      spec_fn: Callable[..., Tensor],
      *init_bounds: Nest[GraphInput],
  ) -> InputAnalysis:
    """Performs pre-computation for an input."""
    graph_nodes = computation_graph_nodes(spec_fn, *init_bounds)
    preact_indices = primitive_preact_indices(graph_nodes,
                                              [synthetic_primitives.relu_p])
    bound_info = bound_input_info(*init_bounds)
    input_indices = bound_info['input_indices']

    if (not preact_indices) and (not input_indices):
      raise ValueError('Cannot branch because this network has no ReLUs'
                       ' and no branchable inputs.')
    # Extract the bias terms.
    inspector = bound_utils.GraphInspector()
    output_node, _ = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(inspector),
        spec_fn, *init_bounds)
    biases = {}
    for node in inspector.nodes.values():
      # Forward-propagate the bias.
      if node.is_input():
        input_bound: InputBound = node.args[0]
        bias = jnp.zeros_like(input_bound.lower)
      else:
        is_bound = lambda b: isinstance(b, bound_propagation.Bound)
        arg_biases = [
            jnp.zeros_like(biases[arg.index]) if is_bound(arg) else arg
            for arg in node.args]
        bias = node.primitive.bind(*arg_biases, **node.kwargs)
      # Network inputs and ReLU outputs have no bias.
      if node.is_input() or node.primitive == synthetic_primitives.relu_p:
        bias = jnp.zeros_like(bias)
      biases[node.index] = bias

    return (spec_fn, init_bounds, graph_nodes, bound_info,
            output_node.index, biases)

  def preprocess_heuristics(
      self,
      input_analysis: InputAnalysis,
      heuristic_inputs: BranchHeuristicInputs,
  ) -> Tuple[InputAnalysis, BranchHeuristicInputs, Dict[Index, Tensor]]:
    """Compute scores to determine the influential ReLUs on which to branch."""
    (spec_fn, init_bounds, graph_nodes, bound_info,
     output_node_index, _) = input_analysis

    bounds = heuristic_inputs['intermediate_bounds']

    preact_indices = primitive_preact_indices(graph_nodes,
                                              [synthetic_primitives.relu_p])
    input_indices = bound_info['input_indices']

    output_sensitivity = None
    if self._sensitivity_of_max:
      output_upper = bounds[output_node_index][1]
      worst_output = (output_upper == output_upper.max())
      output_sensitivity = -worst_output.astype(jnp.float32)
    sensitivity_targets = preact_indices + input_indices
    sensitivity_algorithm = backpropagation.SensitivityAlgorithm(
        bound_utils.FixedBoundApplier(bounds),
        sensitivity_targets, output_sensitivity)
    bound_propagation.bound_propagation(
        sensitivity_algorithm, spec_fn, *init_bounds)

    sensitivities = {index: sens for index, sens in zip(
        sensitivity_targets, sensitivity_algorithm.target_sensitivities)}

    return input_analysis, heuristic_inputs, sensitivities


class SmartReluSelector(SensitivitySelector):
  """Neuron selection strategy based on estimated influence on output objective.
  """

  def score_handled_nodes(
      self,
      scoring_inputs: Tuple[InputAnalysis,
                            BranchHeuristicInputs,
                            Dict[Index, Tensor]],
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    """Compute neuron scores for a given node, given sensitivity and bounds.

    This implements Equation (9) in "Branch and Bound for Piecewise Linear
    Neural Network Verification", https://arxiv.org/pdf/1909.06588.pdf.

    Args:
      scoring_inputs: Tuple composed of the results of the input analysis, the
        statistics at that point of the branch-and-bound tree, as well as the
        computed sensitivities
    Returns:
      scores: Scores for branching on each ReLU of the network.
      branch_vals: None, which by default corresponds to branching over the zero
        values.
    """
    input_analysis, heuristic_inputs, sensitivities = scoring_inputs
    _, _, graph_nodes, _, _, biases = input_analysis
    bounds = heuristic_inputs['intermediate_bounds']
    relu_preact_indices = primitive_preact_indices(
        graph_nodes, [synthetic_primitives.relu_p])

    scores = {}
    branch_vals = {}
    for index in relu_preact_indices:
      lower_bound, upper_bound = bounds[index]
      bias = biases.get(index, 0.)
      sensitivity = sensitivities[index]

      # Only consider 'ambiguous' ReLUs: those whose input bounds straddle zero.
      ambiguous = jnp.logical_and(lower_bound < 0., upper_bound > 0.)
      # For excluded ReLUs, adjust bounds to a safe non-zero value.
      upper_bound = jnp.maximum(upper_bound, jnp.finfo(jnp.float32).eps)

      bias_score = aggregate_ambiguities(self._ambiguity_agg)(
          sensitivity * bias,
          (sensitivity * lower_bound / upper_bound) * bias)
      intercept_score = -lower_bound * jnp.maximum(sensitivity, 0.)

      scores[index] = jnp.where(ambiguous,
                                jnp.abs(bias_score + intercept_score),
                                -jnp.inf)
      branch_vals[index] = jnp.zeros_like(lower_bound)

    return scores, branch_vals


class SensitivityReluSelector(SensitivitySelector):
  """Neuron selection strategy based on estimated influence on output objective.
  """

  def score_handled_nodes(
      self,
      scoring_inputs: Tuple[InputAnalysis,
                            BranchHeuristicInputs,
                            Dict[Index, Tensor]],
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    """Compute neuron scores for a given node, given sensitivity and bounds.

    This uses the sensitivity to estimate the impact of a change in what the K&W
    bound would be, based on a local approximation.
    This does not take into account:
      - The impact on subsequent bounds being tightened.
      - The impact on sensitivities going further towards the beginnning of the
        network.

    It only consider the impact involving the sensitivity at node preact_index.
    The formula for the K&W bound is:

    bound = -sum_{k=1}^n nu_{k+1}b_k - u_0 [nuhat_1]+ + l_0 [nuhat_1]-
            + sum_{k=2}^n u_k * l_k/(u_k - l_k) * [nuhat_k]+

    where nu / nuhat are the sensitivities.
    for ambiguous relu, nu = u/(u-l) * nuhat
    for positive relu, nu = nuhat
    for negative relu, nu = 0

    Args:
      scoring_inputs: Tuple composed of the results of the input analysis, the
        statistics at that point of the branch-and-bound tree, as well as the
        computed sensitivities
    Returns:
      scores: Scores for branching on each ReLU of the network.
      branch_vals: None, which by default corresponds to branching over the zero
        values.
    """

    input_analysis, heuristic_inputs, sensitivities = scoring_inputs
    bounds = heuristic_inputs['intermediate_bounds']
    _, _, graph_nodes, _, _, biases = input_analysis
    relu_preact_indices = primitive_preact_indices(
        graph_nodes, [synthetic_primitives.relu_p])

    scores = {}
    branch_vals = {}
    for index in relu_preact_indices:
      lower_bound, upper_bound = bounds[index]
      bias = biases.get(index, 0.)
      sensitivity = sensitivities[index]

      # Only consider 'ambiguous' ReLUs: those whose input bounds straddle zero.
      ambiguous = jnp.logical_and(lower_bound < 0., upper_bound > 0.)
      # For excluded ReLUs, adjust bounds to a safe non-zero value.
      upper_bound = jnp.maximum(upper_bound, jnp.finfo(jnp.float32).eps)

      nuhat = sensitivity * (upper_bound - lower_bound) / upper_bound
      # In the case of an ambiguous ReLU, the contribution to the bound is:
      # nu = sensitivity = u / (u-l) * nuhat
      score_amb = (sensitivity * bias
                   + jnp.maximum(sensitivity, 0.) * lower_bound)
      # If we are blocking the ReLU to be on, the contribution to the bound is:
      # (as nu = nuhat)
      score_on = nuhat * bias + jnp.maximum(nuhat, 0.) * lower_bound
      # If we are blocking the ReLU to be off, the contribution to the bound is:
      # (as nu = 0)
      score_off = jnp.maximum(nuhat, 0.) * lower_bound

      # We're going to compare the estimated improvements in bounds that we
      # would gain by splitting on the ReLU, aggregating them in the chosen way.
      # score_amb is the contribution to the upper bound as the ReLU was
      # ambiguous so its difference with score_on and score_off represent the
      # expected improvement we would get by splitting.
      improvements = aggregate_ambiguities(self._ambiguity_agg)(
          jnp.maximum(score_amb - score_on, 0.),
          jnp.maximum(score_amb - score_off, 0.))

      scores[index] = jnp.where(ambiguous, improvements, -jnp.inf)
      branch_vals[index] = jnp.zeros_like(lower_bound)

    return scores, branch_vals


class SensitivityLinfSelector(SensitivitySelector):
  """Input selection strategy based on estimated influence on output objective.
  """

  def score_handled_nodes(
      self,
      scoring_inputs: Tuple[InputAnalysis,
                            BranchHeuristicInputs,
                            Dict[Index, Tensor]],
  ) -> Tuple[Dict[Index, Tensor], Optional[Dict[Index, BranchVal]]]:
    """Compute Input branching scores given sensitivity and bounds.

    This only relies on the sensitivity of the output node and ignores the
    impact on intermediate bounds.

    Args:
      scoring_inputs: Tuple composed of the results of the input analysis, the
        statistics at that point of the branch-and-bound tree, as well as the
        computed sensitivities
    Returns:
      scores: Scores for branching on each L_inf bound input of the network.
      branch_vals: Mid-point for the interval bounds
    """
    input_analysis, heuristic_inputs, sensitivities = scoring_inputs
    bounds = heuristic_inputs['intermediate_bounds']
    _, _, _, bound_info, _, _ = input_analysis

    scores = {}
    branch_vals = {}

    for index in bound_info['input_indices']:
      if index in bound_info['index_to_nze']:
        # Skip the L0 bounds
        continue
      lower_bound, upper_bound = bounds[index]
      sensitivity = sensitivities[index]

      unfixed = lower_bound != upper_bound

      # Given that the sensitivity is based on propagating the same way
      # through the upper and lower bound of activation, we only have
      # one relaxation.
      # The bound it corresponds to is sensitivity * inp.
      # Assuming the sensitivity is constant, depending on the sign of the
      # coordinate, it will either increase the final bound in the lower branch,
      # or in the upper branch, based on the amount of movement.
      improvement = jnp.abs(sensitivity) * (upper_bound - lower_bound) / 2

      scores[index] = jnp.where(unfixed, improvement, -jnp.inf)
      branch_vals[index] = 0.5 + (lower_bound + upper_bound)

    return scores, branch_vals


def aggregate_ambiguities(
    ambiguity_agg: str) -> Callable[[Tensor, Tensor], Tensor]:
  if ambiguity_agg == 'min':
    return jnp.minimum
  elif ambiguity_agg == 'max':
    return jnp.maximum
  elif ambiguity_agg == 'avg':
    return lambda x, y: (x+y)/2.
  else:
    raise ValueError('Unknown ambiguity aggregation method: {ambiguity_agg}')


def bound_input_info(
    *init_bound: Nest[graph_traversal.GraphInput],
) -> Dict[str, Any]:
  """Returns useful information on inputs suitable for branching."""

  is_bound = lambda b: isinstance(b, InputBound)
  input_count = 0
  index = graph_traversal.IndexCounter()

  flat_bounds, _ = jax.tree_util.tree_flatten(init_bound, is_leaf=is_bound)

  input_indices = []
  index_to_input = {}

  for bound in flat_bounds:
    if is_bound(bound):
      if isinstance(bound, bound_propagation.IntervalBound):
        index_to_input[index.as_tuple()] = input_count
      else:
        raise ValueError('Unknown input bound type')
      input_indices.append(index.as_tuple())
      index.incr()
    input_count = input_count + 1

  return {
      'input_indices': input_indices,
      'index_to_input': index_to_input,
  }


def _select_highest_scoring_neuron(
    scores: Dict[Index, Tensor]) -> Optional[Tuple[Index, int]]:
  """Identifies the most fitting neuron on which to branch.

  Args:
    scores: Scores for neurons in candidate graph nodes, indicating
      fitness for branching.

  Returns:
    `None` if no suitable neuron found, otherwise:
    index: Node containing the selected neuron.
    neuron_idx: Location within the flattened tensor of the selected neuron.
  """
  indices, flat_scores = zip(*scores.items())

  # Select the pre-activation neuron with the greatest ambiguity.
  # First select the node.
  max_scores = [jnp.amax(score) for score in flat_scores]
  argmax = jnp.argmax(jnp.array(max_scores))
  index = indices[argmax]
  if max_scores[argmax] == -jnp.inf:
    return None  # No feasible branching

  # Next select the highest-scoring neuron within that node.
  neuron_idx = jnp.argmax(flat_scores[argmax])
  return jax.device_get(index), jax.device_get(neuron_idx)


def computation_graph_nodes(
    spec_fn: Callable[..., Tensor],
    *init_bound: Nest[GraphInput],
) -> Dict[Index, bound_utils.GraphNode]:
  """Extract a mapping from index to primitives."""
  inspector = bound_utils.GraphInspector()
  bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(inspector),
      spec_fn, *init_bound)
  return inspector.nodes


def primitive_preact_indices(graph_nodes: Dict[Index, bound_utils.GraphNode],
                             target_primitives: Sequence[jax.core.Primitive],
                             ) -> Sequence[Index]:
  """Extract the index of the input to the required primitives."""
  preact_indices = []
  for _, node in graph_nodes.items():
    if node.primitive in target_primitives:
      preact_indices.append(node.args[0].index)
  return preact_indices


def highest_score_branch_plane(
    scores: Dict[Index, Tensor],
    branch_vals: Optional[Dict[Index, BranchVal]],
) -> Optional[BranchPlane]:
  """Determines the neuron on which to branch.

  Args:
    scores: Score for each graph node.
    branch_vals: Critical values of all neurons for each graph node. If `None`
      then simply use zero for all critical values.

  Returns:
    `None` if no suitable neuron found, otherwise:
    node_index: Graph node containing the selected neuron.
    neuron_index: Location within the flattened tensor of the selected neuron.
    branch_val: Critical value of the selected neuron on which to branch.
  """
  indices = _select_highest_scoring_neuron(scores)
  if indices is None:
    return None
  index, neuron_idx = indices
  return make_branch_plane(index, neuron_idx, branch_vals)


def make_branch_plane(
    index: Index,
    neuron_idx: int,
    branch_vals: Optional[Dict[Index, BranchVal]],
) -> BranchPlane:
  """Returns branch plane with the specified content."""

  def neuron_branch_val(branch_vals):
    if jnp.isscalar(branch_vals):
      return branch_vals
    else:
      flat_branch_vals = jnp.reshape(branch_vals, (-1,))
      return flat_branch_vals[neuron_idx]

  if branch_vals is not None:
    index_branch_vals = branch_vals[index]
    if isinstance(index_branch_vals, tuple):
      lower_branch_vals, upper_branch_vals = index_branch_vals
    else:
      lower_branch_vals = upper_branch_vals = index_branch_vals

    lower_branch_val = neuron_branch_val(lower_branch_vals)
    upper_branch_val = neuron_branch_val(upper_branch_vals)
  else:
    # This is the common default case, applicable to ReLUs.
    lower_branch_val = upper_branch_val = 0.
  return BranchPlane(
      BranchDecision(index, neuron_idx, lower_branch_val, -1),
      BranchDecision(index, neuron_idx, upper_branch_val, 1))


def concrete_branch_bounds(
    root_bounds: Dict[Index, Tuple[Tensor, Tensor]],
    branching_decisions: BranchingDecisionList,
) -> Dict[Index, Tuple[Tensor, Tensor]]:
  """Materialises a branch's decision path as concrete interval bounds.

  Args:
    root_bounds: Initial concrete bounds for each graph node; may be vacuous.
    branching_decisions: Specifies a branch's decision path.

  Returns:
    Copy of root_bounds, refined according to the current branch constraints.
  """
  branch_bounds = dict(root_bounds)
  for node, neuron_idx, branch_val, side in branching_decisions:
    lower, upper = branch_bounds[node]
    if side > 0:
      lower = _set_element(lower, neuron_idx, branch_val)
    else:
      upper = _set_element(upper, neuron_idx, branch_val)
    branch_bounds[node] = lower, upper
  return branch_bounds


def _set_element(x: Tensor, i: int, v: Tensor) -> Tensor:
  """Returns a copy of `x` with its `i`th element set to v."""
  shape = x.shape
  x = jnp.reshape(x, [-1])
  y = x.at[i].set(v)
  return jnp.reshape(y, shape)


def max_index_depth(
    spec_fn: Callable[[Nest[Tensor]], Nest[Tensor]],
    *input_bounds: GraphInput) -> int:
  """Geth the maximum number of integers to use to identify a node."""
  inspector = bound_utils.GraphInspector()
  bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(inspector),
      spec_fn, *input_bounds)
  return max(len(node.index) for node in inspector.nodes.values())


def branching_decisions_tensors(
    branching_decisions: BranchingDecisionList,
    initial_max_branching_depth: int,
    max_idx_depth: int
) -> JittableBranchingDecisions:
  """Create the tensor version of `branching_decisions`.

  Args:
    branching_decisions: List of branching decisions, where branching
      decisions are a branching plane and an integer indicating whether it is
      the upper or lower branch.
    initial_max_branching_depth: Length of the tensors to use to represent the
      branching decisions in jitted form. Padding will be added to the
      `branching_decisions` elements to make it to that length. If the length
      of `branching_decisions` is larger than this, we will instead pad to the
      nearest power of two multiple of this.
    max_idx_depth: Max number of coordinates that we will use to identify a
      node in the computation graph.
  Returns:
    Tensorial representation of `branching_decisions`.
  """
  node_indices = []
  neuron_indices = []
  branch_vals = []
  is_upper_branch = []

  for node_index, neuron_index, branch_val, side in branching_decisions:
    node_indices.append(node_index + (0,)*(max_idx_depth - len(node_index)))
    neuron_indices.append(neuron_index)
    branch_vals.append(branch_val)
    is_upper_branch.append(side > 0)

  # Pad the tensors to one of the accepted lengths.
  factor = len(branching_decisions) / initial_max_branching_depth
  safe_factor = max(1, factor)
  next_2pow = math.ceil(math.log2(safe_factor))
  tensor_len = initial_max_branching_depth * (2 ** next_2pow)

  while len(node_indices) < tensor_len:
    node_indices.append((0,)*max_idx_depth)
    neuron_indices.append(-1)
    branch_vals.append(0.)
    is_upper_branch.append(True)

  return JittableBranchingDecisions(
      jnp.array(node_indices, dtype=jnp.int32),
      jnp.array(neuron_indices, dtype=jnp.int32),
      jnp.array(branch_vals, dtype=jnp.float32),
      jnp.array(is_upper_branch, dtype=jnp.bool_))


def enforce_jittable_branching_decisions(
    branching_decisions: JittableBranchingDecisions,
    index: Index, bound: Bound,
) -> Bound:
  """Ensure that a bound is consistent with a list of branching decisions.

  Args:
    branching_decisions: List of constraints that need to be enforced.
    index: Index of the node at which this bounds lives.
    bound: Bound that need to be consistent with the constraints.
  Returns:
    enforced_bound: Modified version of `bound` guaranteed to respect the
      constraints
  """

  lay_idxs, neur_idxs, branch_vals, is_upper = branching_decisions
  max_idx_depth = lay_idxs.shape[1]
  index_tensor = jnp.array(index + (0,) * (max_idx_depth - len(index)))
  layer_mask = (lay_idxs == index_tensor).all(axis=1)

  update_lower = layer_mask & is_upper
  update_upper = layer_mask & ~is_upper

  update_low_val = jnp.where(update_lower, branch_vals, -jnp.inf)
  update_upper_val = jnp.where(update_upper, branch_vals, jnp.inf)

  flat_lower = jnp.reshape(bound.lower, [-1])
  enforced_flat_lower = flat_lower.at[neur_idxs].max(update_low_val,
                                                     mode='drop')
  enforced_lower = jnp.reshape(enforced_flat_lower, bound.lower.shape)

  flat_upper = jnp.reshape(bound.upper, [-1])
  enforced_flat_upper = flat_upper.at[neur_idxs].min(update_upper_val,
                                                     mode='drop')
  enforced_upper = jnp.reshape(enforced_flat_upper, bound.upper.shape)

  return bound_propagation.IntervalBound(enforced_lower, enforced_upper)
