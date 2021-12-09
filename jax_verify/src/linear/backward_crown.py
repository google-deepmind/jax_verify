# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
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

"""Implementation of Backward Crown / Fastlin.
"""
import functools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import graph_traversal
from jax_verify.src import ibp
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils
from jax_verify.src.linear import linear_bound_utils
import optax


LinearExpression = linear_bound_utils.LinearExpression
Index = bound_propagation.Index
Bound = bound_propagation.Bound
Primitive = bound_propagation.Primitive
Tensor = bound_propagation.Tensor
GraphInput = graph_traversal.GraphInput
LayerInput = bound_utils.LayerInput
Nest = bound_propagation.Nest
ParameterizedNodeRelaxation = linear_bound_utils.ParameterizedNodeRelaxation


def _sum_linear_backward_bounds(linbound_seq: Sequence[LinearExpression]
                                ) -> LinearExpression:
  if len(linbound_seq) == 1:
    return linbound_seq[0]
  else:
    return linbound_seq[0] + _sum_linear_backward_bounds(linbound_seq[1:])


def _backpropagate_linear_functions(
    linfun: linear_bound_utils.LinFun,
    outval: LinearExpression,
    *invals: GraphInput)-> List[Optional[LinearExpression]]:
  """Propagate a linear function backward through the linfun function.

  Args:
    linfun: Linear function to propagate through.
    outval: Coefficients of a linear functions over the output of linfun.
    *invals: Tensor or Bounds that are inputs to linfun
  Returns:
    new_in_args: Coefficients of a linear functions over the input of linfun,
      representing the same linear function as was represented by outval.
  """
  # Figure out the bias of the linear transformation, which will need to be
  # added to the offset.
  zero_in_args = [jnp.zeros(arg.shape) for arg in invals
                  if isinstance(arg, Bound)]
  nb_bound_inputs = len(zero_in_args)

  linfun_onlybound_input = utils.bind_nonbound_args(linfun, *invals)
  linfun_bias, vjp_fun = jax.vjp(linfun_onlybound_input, *zero_in_args)

  # Let's evaluate what offset this would correspond to, based on the what the
  # outval is.
  broad_bias = jnp.expand_dims(linfun_bias, 0)
  dims_to_reduce = tuple(range(1, broad_bias.ndim))

  # We're splitting the offset between all the bounds that we propagate
  # backward. This contains both the offset that was already present on the
  # bounds being propagated backward and the ones coming from this level
  # of the relaxation.
  total_offset = (outval.offset + jnp.sum(outval.lin_coeffs * broad_bias,
                                          dims_to_reduce))
  shared_offset = total_offset / nb_bound_inputs

  # Let's vmap the target dimension, so as to backpropagate for all targets.
  vjp_fun = jax.vmap(vjp_fun, in_axes=0, out_axes=0)

  in_args_lin_coeffs = vjp_fun(outval.lin_coeffs)

  new_in_args = []
  bound_arg_pos = 0
  for arg in invals:
    if isinstance(arg, Bound):
      new_in_args.append(LinearExpression(in_args_lin_coeffs[bound_arg_pos],
                                          shared_offset))
      bound_arg_pos += 1
    else:
      new_in_args.append(None)

  return new_in_args


def _handle_linear_relaxation(
    lb_fun: linear_bound_utils.LinFun,
    ub_fun: linear_bound_utils.LinFun,
    outval: LinearExpression,
    *invals: LayerInput) -> List[Optional[LinearExpression]]:
  """Propagate a linear function backward through a linear relaxation.

  This is employed when we have a non-linear primitive, once we have obtained
  its linear lower bounding and linear upper bounding function.
  We backpropagate through this linear relaxation of a non-linear primitive.

  Args:
    lb_fun: Linear lower bound of the primitive to propagate backwards through.
    ub_fun: Linear upper bound of the primitive to progagate backwards through.
    outval: Coefficients of a linear function over the output of the primitive
      relaxed by lb_fun and ub_fun.
    *invals: Tensor or Bounds that are inputs to the primitive relaxed by lb_fun
      and ub_fun.
  Returns:
    new_in_args: Coefficients of a linear function over the input of the
      primitive, representing the same linear function as was represented by
      outval.
  """
  # We're going to split the linear function over the output into two parts,
  # depending on the sign of the coefficients.
  # The one with positive coefficients will be backpropagated through the
  # lower bound, the one with the negative coefficients will be propagated
  # through the upper bound.
  # The offset can go on either as it is not actually backpropagated, we just
  # need to make sure that it is not double-counted.
  pos_outval = LinearExpression(jnp.maximum(outval.lin_coeffs, 0.),
                                outval.offset)
  neg_outval = LinearExpression(jnp.minimum(outval.lin_coeffs, 0.),
                                jnp.zeros_like(outval.offset))

  through_pos_inlinfuns = _backpropagate_linear_functions(
      lb_fun, pos_outval, *invals)
  through_neg_inlinfuns = _backpropagate_linear_functions(
      ub_fun, neg_outval, *invals)

  new_in_args = []
  for pos_contrib, neg_contrib in zip(through_pos_inlinfuns,
                                      through_neg_inlinfuns):
    # The None should be in the same position, whether through the lower or
    # upper bound.
    assert (pos_contrib is None) == (neg_contrib is None)
    if pos_contrib is None:
      new_in_args.append(None)
    else:
      new_in_args.append(pos_contrib + neg_contrib)

  return new_in_args


class LinearBoundBackwardTransform(
    bound_utils.BackwardConcretizingTransform[LinearExpression]):
  """Transformation to propagate linear bounds backwards and concretize them."""

  def __init__(
      self,
      relaxer: linear_bound_utils.LinearBoundsRelaxer,
      primitive_needs_concrete_bounds: Tuple[Primitive, ...]):
    self.relaxer = relaxer
    self._primitive_needs_concrete_bounds = primitive_needs_concrete_bounds

  def concretize_args(self, primitive: Primitive) -> bool:
    return primitive in self._primitive_needs_concrete_bounds

  def aggregate(self, eqn_outvals: Sequence[LinearExpression]
                ) -> LinearExpression:
    return _sum_linear_backward_bounds(eqn_outvals)

  def primitive_backtransform(
      self,
      context: graph_traversal.TransformContext,
      primitive: Primitive,
      eqn_outval: LinearExpression,
      *args: LayerInput,
      **params) -> Sequence[Sequence[Optional[LinearExpression]]]:
    if (primitive in bound_propagation.AFFINE_PRIMITIVES
        or primitive in bound_propagation.RESHAPE_PRIMITIVES):
      lin_fun = functools.partial(primitive.bind, **params)
      in_linfun = _backpropagate_linear_functions(lin_fun, eqn_outval, *args)
    else:
      # This is not an affine primitive. We need to go through a relaxation.
      # Obtain the linear bounds.
      index = context.index
      lb_linrelaxfun, ub_linrelaxfun = self.relaxer.linearize_primitive(
          index, primitive, *args, **params)
      in_linfun = _handle_linear_relaxation(lb_linrelaxfun, ub_linrelaxfun,
                                            eqn_outval, *args)
    return list(zip(in_linfun))

  def concrete_bound_chunk(
      self,
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
      env: Dict[jax.core.Var, LayerInput],
      node_ref: jax.core.Var,
      obj: Tensor,
  ) -> Tensor:
    initial_linear_expression = identity(obj)

    flat_inputs, _ = jax.tree_util.tree_flatten(inputs)
    bound_inputs = [inp for inp in flat_inputs
                    if isinstance(inp, bound_propagation.Bound)]
    input_nodes_indices = [(i,) for i in range(len(bound_inputs))]
    inputs_linfuns, _ = graph.backward_propagation(
        self, env, {node_ref: initial_linear_expression}, input_nodes_indices)

    flat_bound = jnp.zeros(())
    for input_linfun, inp_bound in zip(inputs_linfuns, bound_inputs):
      if input_linfun is not None:
        # Only concretize when the input_linfun is not None. It is possible,
        # especially when computing intermediate bounds, that not all of the
        # inputs will have an impact on each bound to compute.
        # Example:
        #  a -> Linear -> Relu -> sum -> out
        #  b -------------------/
        # When computing the bound on the input to the ReLU, the backward
        # bound on b will be None, and can be safely ignored.
        inp_contrib = concretize_backward_bound(input_linfun, inp_bound)

        flat_bound = flat_bound + inp_contrib

    return flat_bound


class _RelaxationScanner(graph_traversal.BackwardGraphTransform[LayerInput]):
  """Identifies the node relaxations relevant to the graph."""

  def __init__(
      self,
      relaxer: linear_bound_utils.ParameterizedLinearBoundsRelaxer,
  ):
    self._relaxer = relaxer
    self._node_relaxations = {}

  @property
  def node_relaxations(self) -> Dict[Index, ParameterizedNodeRelaxation]:
    return self._node_relaxations

  def aggregate(self, eqn_outvals: Sequence[LayerInput]) -> LayerInput:
    # In the case of fan-out, the same forward value should have been
    # encountered on every possible backward path.
    assert all(eqn_outval is eqn_outvals[0] for eqn_outval in eqn_outvals)
    return eqn_outvals[0]

  def primitive_backtransform(
      self,
      context: graph_traversal.TransformContext,
      primitive: Primitive,
      eqn_outval: LayerInput,
      *args: LayerInput,
      **params) -> Sequence[Sequence[Optional[LayerInput]]]:
    if not (primitive in bound_propagation.AFFINE_PRIMITIVES
            or primitive in bound_propagation.RESHAPE_PRIMITIVES):
      # This is not an affine primitive. We need to go through a relaxation.
      # Obtain the linear bounds.
      arg_shapes = [arg.shape for arg in args]
      self._node_relaxations[context.index] = (
          self._relaxer.parameterized_linearizer(
              context.index, primitive, *arg_shapes, **params))

    # We're using this back-transform to traverse the nodes, rather than to
    # compute anything. Arbitrarily return the forward bounds associated with
    # each node.
    return [[arg if isinstance(arg, Bound) else None] for arg in args]


class OptimizingLinearBoundBackwardTransform(
    bound_utils.BackwardConcretizingTransform[LinearExpression]):
  """Transformation to propagate linear bounds backwards and concretize them."""

  def __init__(
      self,
      relaxer: linear_bound_utils.ParameterizedLinearBoundsRelaxer,
      primitive_needs_concrete_bounds: Tuple[Primitive, ...],
      opt: optax.GradientTransformation,
      num_opt_steps: int,
  ):
    """Constructs a per-node concretizer that performs an inner optimisation.

    Args:
      relaxer: Specifies the parameterised linear relaxation to use for each
        primitive operation.
      primitive_needs_concrete_bounds: Which primitive operations need to be
        concretised.
      opt: Optimiser used to minimise the upper bounds (and the negative lower
        bounds) with respect to the linear relaxation parameters.
      num_opt_steps: Number of optimisation steps.
    """
    self._relaxer = relaxer
    self._primitive_needs_concrete_bounds = primitive_needs_concrete_bounds
    self._opt = opt
    self._num_opt_steps = num_opt_steps

  def concretize_args(self, primitive: Primitive) -> bool:
    return primitive in self._primitive_needs_concrete_bounds

  def aggregate(self, eqn_outvals: Sequence[LinearExpression]
                ) -> LinearExpression:
    raise NotImplementedError()

  def primitive_backtransform(
      self,
      context: graph_traversal.TransformContext,
      primitive: Primitive,
      eqn_outval: LinearExpression,
      *args: LayerInput,
      **params) -> Sequence[Sequence[Optional[LinearExpression]]]:
    raise NotImplementedError()

  def concrete_bound_chunk(
      self,
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
      env: Dict[jax.core.Var, LayerInput],
      node_ref: jax.core.Var,
      obj: Tensor,
  ) -> Tensor:
    # Analyse the relevant parts of the graph.
    flat_inputs, _ = jax.tree_util.tree_flatten(inputs)
    bound_inputs = [inp for inp in flat_inputs
                    if isinstance(inp, bound_propagation.Bound)]
    input_nodes_indices = [(i,) for i in range(len(bound_inputs))]
    scanner = _RelaxationScanner(self._relaxer)
    graph.backward_propagation(
        scanner, env, {node_ref: env[node_ref]}, input_nodes_indices)

    # Allow lookup of any node's input bounds, for parameter initialisation.
    graph_inspector = bound_utils.GraphInspector()
    bound_propagation.ForwardPropagationAlgorithm(
        graph_inspector).propagate(graph, inputs)

    def input_bounds(index: Index) -> Sequence[LayerInput]:
      graph_node = graph_inspector.nodes[index]
      return [env[graph.jaxpr_node(arg.index)]
              if isinstance(arg, bound_utils.GraphNode) else arg
              for arg in graph_node.args]

    # Define optimisation for a single neuron's bound. (We'll vmap afterwards.)
    # This ensures that each neuron uses independent relaxation parameters.
    def optimized_concrete_bound(one_obj):
      def concrete_bound(relax_params):
        return self._bind(
            scanner.node_relaxations, relax_params).concrete_bound_chunk(
                graph, inputs, env, node_ref, jnp.expand_dims(one_obj, 0))

      # Define function to optimise: summary tightness of guaranteed bounds.
      def objective(relax_params):
        lb_min = concrete_bound(relax_params)
        return jnp.sum(-lb_min)

      val_and_grad_fn = jax.value_and_grad(objective)

      # Optimise the relaxation parameters.
      initial_params = self._initial_params(scanner, input_bounds)
      initial_state = (initial_params, self._opt.init(initial_params),
                       initial_params, jnp.inf)
      def update_state(_, state):
        params, opt_state, best_params, best_val = state
        params_val, params_grad = val_and_grad_fn(params)
        # Compute the next step in the optimization process.
        updates, next_opt_state = self._opt.update(params_grad, opt_state)
        next_params = optax.apply_updates(params, updates)
        next_params = self._project_params(scanner, next_params)
        # Update the best params seen.
        params_improved = params_val < best_val
        update_best_params = lambda p, best: jnp.where(params_improved, p, best)

        next_best_params = jax.tree_multimap(update_best_params,
                                             params, best_params)
        next_best_val = jnp.minimum(best_val, params_val)
        return next_params, next_opt_state, next_best_params, next_best_val
      _, _, relax_params, _ = jax.lax.fori_loop(
          0, self._num_opt_steps, update_state, initial_state)

      # Evaluate the relaxation at these parameters.
      return concrete_bound(jax.lax.stop_gradient(relax_params))

    return jax.vmap(optimized_concrete_bound)(obj)

  def _initial_params(self, scanner, input_bounds):
    return {index: node_relaxation.initial_params(*input_bounds(index))
            for index, node_relaxation in scanner.node_relaxations.items()}

  def _project_params(self, scanner, unc_params):
    return {
        index: node_relaxation.project_params(unc_params[index])
        for index, node_relaxation in scanner.node_relaxations.items()}

  def _bind(
      self,
      node_relaxations: Dict[Index, ParameterizedNodeRelaxation],
      relax_params: Dict[Index, Tensor],
  ) -> LinearBoundBackwardTransform:
    return LinearBoundBackwardTransform(
        linear_bound_utils.BindRelaxerParams(node_relaxations, relax_params),
        self._primitive_needs_concrete_bounds)


class ChunkedBackwardConcretizer(bound_utils.BackwardConcretizer):
  """Concretizer that invokes the given transform in chunks for each layer."""

  def __init__(
      self,
      concretizing_transform: bound_utils.BackwardConcretizingTransform[
          LinearExpression],
      max_chunk_size: int = 0):
    self._concretizing_transform = concretizing_transform
    self._max_chunk_size = max_chunk_size

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    return self._concretizing_transform.should_handle_as_subgraph(primitive)

  def concretize_args(self, primitive: Primitive) -> bool:
    return self._concretizing_transform.concretize_args(primitive)

  def concrete_bound(
      self,
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
      env: Dict[jax.core.Var, LayerInput],
      node_ref: jax.core.Var,
  ) -> ibp.IntervalBound:
    """Perform backward linear bound computation for the node `index`.

    Args:
      graph: Graph to perform Backward Propagation on.
      inputs: Bounds on the inputs.
      env: Environment containing intermediate bound and shape information.
      node_ref: Reference of the node to obtain a bound for.
    Returns:
      concrete_bound: IntervalBound on the activation at `node_ref`.
    """
    node = graph_traversal.read_env(env, node_ref)

    def bound_fn(obj: Tensor) -> Tuple[Tensor, Tensor]:
      # Handle lower bounds and upper bounds independently in the same chunk.
      obj = jnp.concatenate([obj, -obj], axis=0)

      all_bounds = self._concretizing_transform.concrete_bound_chunk(
          graph, inputs, env, node_ref, obj)

      # Separate out the lower and upper bounds.
      lower_bound, neg_upper_bound = jnp.split(all_bounds, 2, axis=0)
      upper_bound = -neg_upper_bound
      return lower_bound, upper_bound

    return ibp.IntervalBound(
        *utils.chunked_bounds(node.shape, self._max_chunk_size, bound_fn))


def identity(obj: Tensor) -> LinearExpression:
  """Returns identity linear expression for lower bound of objective."""
  initial_lin_coeffs = obj
  initial_offsets = jnp.zeros(obj.shape[:1])
  return LinearExpression(initial_lin_coeffs, initial_offsets)


def concretize_backward_bound(backward_linexp: LinearExpression,
                              act_bound: Bound) -> Tensor:
  """Compute the lower bound value of a backward bound.

  Args:
    backward_linexp: Coefficients of linear functions. The leading batch
      dimension corresponds to different functions that need to be concretized.
    act_bound: Bound on the activations of that layer. Its shape should
      match the coefficients of the linear functions to concretize.
  Returns:
    bound: A concretized bound on the value of the functions represented by
      backward_linexp.
  """
  return _concretize_linear_function_interval_bounds(
    backward_linexp, act_bound)


def _concretize_linear_function_interval_bounds(
    backward_linexp: LinearExpression,
    act_bound: bound_propagation.IntervalBound) -> Tensor:
  """Compute the lower bound of a linear function under interval constraints."""
  act_lower = jnp.expand_dims(act_bound.lower, 0)
  act_upper = jnp.expand_dims(act_bound.upper, 0)

  dims_to_reduce = tuple(range(1, act_lower.ndim))

  return backward_linexp.offset + jnp.sum(
      jnp.minimum(backward_linexp.lin_coeffs, 0.) * act_upper +
      jnp.maximum(backward_linexp.lin_coeffs, 0.) * act_lower,
      dims_to_reduce)


CONCRETIZE_ARGS_PRIMITIVE = (
    synthetic_primitives.leaky_relu_p,
    synthetic_primitives.relu_p,
    synthetic_primitives.sigmoid_p,
    synthetic_primitives.posbilinear_p,
    synthetic_primitives.posreciprocal_p,
    lax.abs_p,
    lax.exp_p
)

backward_crown_transform = LinearBoundBackwardTransform(
    linear_bound_utils.crown_rvt_relaxer, CONCRETIZE_ARGS_PRIMITIVE)
backward_fastlin_transform = LinearBoundBackwardTransform(
    linear_bound_utils.fastlin_rvt_relaxer, CONCRETIZE_ARGS_PRIMITIVE)
backward_crown_concretizer = ChunkedBackwardConcretizer(
    backward_crown_transform)
backward_fastlin_concretizer = ChunkedBackwardConcretizer(
    backward_fastlin_transform)


def crownibp_bound_propagation(
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[GraphInput]) -> Nest[LayerInput]:
  """Performs Crown-IBP as described in https://arxiv.org/abs/1906.06316.

  We first perform IBP to obtain intermediate bounds and then propagate linear
  bounds backwards.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBounds, bounds on the inputs of the function.
  Returns:
    output_bounds: Bounds on the outputs of the function obtained by Crown-IBP
  """
  crown_ibp_algorithm = bound_utils.BackwardAlgorithmForwardConcretization(
      ibp.bound_transform, backward_crown_concretizer)
  output_bounds, _ = bound_propagation.bound_propagation(
      crown_ibp_algorithm, function, *bounds)
  return output_bounds


def backward_crown_bound_propagation(
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[GraphInput]) -> Nest[LayerInput]:
  """Performs CROWN as described in https://arxiv.org/abs/1811.00866.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  backward_crown_algorithm = bound_utils.BackwardConcretizingAlgorithm(
      backward_crown_concretizer)
  output_bound, _ = bound_propagation.bound_propagation(
      backward_crown_algorithm, function, *bounds)
  return output_bound


def backward_rvt_bound_propagation(
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[GraphInput]) -> Nest[LayerInput]:
  """Performs CROWN as described in https://arxiv.org/abs/1811.00866.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  backward_crown_algorithm = bound_utils.BackwardConcretizingAlgorithm(
      backward_crown_concretizer)
  expand_softmax_simplifier_chain = synthetic_primitives.simplifier_composition(
      synthetic_primitives.activation_simplifier,
      synthetic_primitives.hoist_constant_computations,
      synthetic_primitives.expand_softmax_simplifier,
      synthetic_primitives.group_linear_sequence,
      synthetic_primitives.group_posbilinear)
  output_bound, _ = bound_propagation.bound_propagation(
      backward_crown_algorithm, function, *bounds,
      graph_simplifier=expand_softmax_simplifier_chain)
  return output_bound


def backward_fastlin_bound_propagation(
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[GraphInput]) -> Nest[LayerInput]:
  """Performs FastLin as described in https://arxiv.org/abs/1804.09699.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  backward_fastlin_algorithm = bound_utils.BackwardConcretizingAlgorithm(
      backward_fastlin_concretizer)
  output_bound, _ = bound_propagation.bound_propagation(
      backward_fastlin_algorithm, function, *bounds)
  return output_bound


class JointOptimizationConcretizationAlgorithm(
    bound_propagation.PropagationAlgorithm[Bound]
):
  """Algorithm to jointly optimize all the bounds in the network."""

  def __init__(self,
               relaxer: linear_bound_utils.ParameterizedLinearBoundsRelaxer,
               opt: optax.GradientTransformation,
               num_opt_steps: int,
               max_chunk_size: int = 0):
    self._relaxer = relaxer
    self._opt = opt
    self._num_opt_steps = num_opt_steps
    self._max_chunk_size = max_chunk_size

  def propagate(self,
                graph: bound_propagation.PropagationGraph,
                inputs: Nest[GraphInput]):

    # Inspect the graph to figure out what are the nodes needing concretization.
    graph_inspector = bound_utils.GraphInspector()
    inspector_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        graph_inspector)
    gn_outvals, env = inspector_algorithm.propagate(graph, inputs)

    flat_inputs, _ = jax.tree_util.tree_flatten(inputs)
    flat_bounds = [inp for inp in flat_inputs
                   if isinstance(inp, bound_propagation.Bound)]
    input_nodes_indices = [(i,) for i in range(len(flat_bounds))]

    # For every node that requires relaxations, we will use a RelaxationScanner
    # to collect the node that it requires.
    relaxations = {}
    def collect_relaxations(graph_node):
      if graph_node.index not in relaxations:
        index_to_concretize = graph_node.index
        jaxpr_node = graph.jaxpr_node(index_to_concretize)
        scanner = _RelaxationScanner(self._relaxer)
        graph.backward_propagation(
            scanner, env, {jaxpr_node: graph_node},
            input_nodes_indices)
        relaxations[index_to_concretize] = scanner.node_relaxations

    for node in graph_inspector.nodes.values():
      node_primitive = node.primitive
      if node_primitive and node_primitive in CONCRETIZE_ARGS_PRIMITIVE:
        for node_arg in node.args:
          collect_relaxations(node_arg)

    # Iterate over the outputs, making notes of their index so that we can use
    # them to specify the objective function, and collecting the relaxations we
    # need to define to use them.
    objective_nodes = []
    for gn in gn_outvals:
      collect_relaxations(gn)
      jaxpr_node = graph.jaxpr_node(gn.index)
      objective_nodes.append(jaxpr_node)

    env_with_final_bounds = self.jointly_optimize_relaxations(
        relaxations, graph, inputs, env, objective_nodes)

    outvals = [env_with_final_bounds[jaxpr_node_opted]
               for jaxpr_node_opted in objective_nodes]

    return outvals, env_with_final_bounds

  def jointly_optimize_relaxations(
      self,
      relaxations: Dict[Index, Dict[Index, ParameterizedNodeRelaxation]],
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
      env: Dict[jax.core.Var, LayerInput],
      objective_nodes: Sequence[jax.core.Var]):
    """Perform the joint optimization of all the bounds.

    For a network that is (index in parentheses):
    Inp -> Linear(1) -> Relu(2) -> Linear(3) -> Relu(4) -> Linear(5)

    We would have relaxations be a dict of the form:
    {
      (1,): {},  # When we concretize 1, we don't need any relaxations
      (3,): {(2,): relaxation} # Concretizing 3, we need to relax 2
      (5,): {(2,): relaxation, (4,): relaxation}
    }
    Args:
      relaxations: Dict mapping each index to a relaxation dict mapping the
          preceding primitives to their parameterized relaxer.
      graph: Graph to perform the backward propagation on to obtain bounds.
      inputs: Bounds on the inputs
      env: Environment containing shape information
      objective_nodes: List of jaxpr_nodes indicating which bound to use as
       objective functions.
    Returns:
      env_with_bounds: Environment with concretized bounds.
    """
    # Initialize the parameters for the optimization
    default_init = lambda relax: relax.initial_params(*((None,) * relax.arity))
    initial_params = jax.tree_map(default_init, relaxations)
    initial_state = self._opt.init(initial_params)
    param_and_state = initial_params, initial_state

    # Define a function that compute all bounds that we have parameterized.
    # This will concretize each intermediate bounds using the parameters
    # corresponding to that level of the relaxation.
    def compute_all_bounds(params: Dict[Index, Dict[Index, Nest[Tensor]]]):
      specific_env = env.copy()
      for inter_index, node_relaxations in relaxations.items():
        relax_params = params[inter_index]
        jaxpr_node = graph.jaxpr_node(inter_index)
        backward_transform = LinearBoundBackwardTransform(
            linear_bound_utils.BindRelaxerParams(node_relaxations,
                                                 relax_params),
            CONCRETIZE_ARGS_PRIMITIVE)
        chunked_backward_transform = ChunkedBackwardConcretizer(
            backward_transform, self._max_chunk_size)
        concrete_bound = chunked_backward_transform.concrete_bound(
            graph, inputs, specific_env, jaxpr_node)
        specific_env[jaxpr_node] = concrete_bound
      return specific_env

    # Define the objective function of the optimization. This will be the
    # range of the final bound, as indicated by the objective_nodes argument.
    def objective_fun(params: Dict[Index, Dict[Index, Nest[Tensor]]]):
      env_with_bounds = compute_all_bounds(params)
      obj = 0
      for jaxpr_node_to_opt in objective_nodes:
        bound_to_opt = env_with_bounds[jaxpr_node_to_opt]
        obj = obj + jnp.sum(bound_to_opt.upper - bound_to_opt.lower)
      return obj

    grad_fn = jax.grad(objective_fun)

    # Define the optimization step, and call it as a fori-loop
    def update_fun(_, param_and_state):
      params, opt_state = param_and_state
      updates, next_opt_state = self._opt.update(grad_fn(params), opt_state)
      next_params = optax.apply_updates(params, updates)
      next_params = jax.tree_multimap(
          lambda relax, param: relax.project_params(param),
          relaxations, next_params)
      return next_params, next_opt_state
    relax_params, _ = jax.lax.fori_loop(
        0, self._num_opt_steps, update_fun, param_and_state)

    # Compute the bounds corresponding to the final set of optimized
    # parameters, and extract the final bounds that we were optimizing.
    env_with_final_bounds = compute_all_bounds(relax_params)

    return env_with_final_bounds
