# coding=utf-8
# Copyright 2021 The jax_verify Authors.
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
import math
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
import numpy as np
import optax


LinearExpression = linear_bound_utils.LinearExpression
Index = bound_propagation.Index
Bound = bound_propagation.Bound
Primitive = bound_propagation.Primitive
Tensor = bound_propagation.Tensor
GraphInput = graph_traversal.GraphInput
LayerInput = bound_utils.LayerInput
Nest = bound_propagation.Nest


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
      node_ref: jax.core.Var,
      graph: bound_propagation.PropagationGraph,
      env: Dict[jax.core.Var, LayerInput],
      initial_lin_coeffs: Tensor,
      initial_offsets: Tensor,
      *bounds: Nest[GraphInput],
  ) -> Tuple[Tensor, Tensor]:
    all_bounds_initial_lin_coeffs = jnp.concatenate(
        [initial_lin_coeffs, -initial_lin_coeffs], axis=0)
    all_bounds_initial_offsets = jnp.concatenate(
        [initial_offsets, -initial_offsets], axis=0)

    initial_linear_expression = LinearExpression(
        all_bounds_initial_lin_coeffs, all_bounds_initial_offsets)

    bound_inputs = [inp for inp in bounds
                    if isinstance(inp, bound_propagation.Bound)]
    input_nodes_indices = [(i,) for i in range(len(bound_inputs))]

    inputs_linfuns, _ = graph.backward_propagation(
        self, env, {node_ref: initial_linear_expression}, input_nodes_indices)

    flat_bound_lower = jnp.zeros(())
    flat_bound_upper = jnp.zeros(())
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
        flat_bound_lower = flat_bound_lower + inp_contrib.lower
        flat_bound_upper = flat_bound_upper + inp_contrib.upper

    return flat_bound_lower, flat_bound_upper


class _RelaxationScanner(graph_traversal.BackwardGraphTransform[LayerInput]):
  """Identifies the node relaxations relevant to the graph."""

  def __init__(
      self,
      relaxer: linear_bound_utils.ParameterizedLinearBoundsRelaxer,
  ):
    self._relaxer = relaxer
    self._node_relaxations = {}

  @property
  def node_relaxations(self):
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
      self._node_relaxations[context.index] = self._relaxer.linearize_primitive(
          context.index, primitive, *args, **params)

    # We're using this back-transform to traverse the nodes, rather than to
    # compute anything. Arbitrarily return the forward bounds associated with
    # each node.
    return args


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
      node_ref: jax.core.Var,
      graph: bound_propagation.PropagationGraph,
      env: Dict[jax.core.Var, LayerInput],
      initial_lin_coeffs: Tensor,
      initial_offsets: Tensor,
      *bounds: Nest[GraphInput],
  ) -> Tuple[Tensor, Tensor]:
    # Analyse the relevant parts of the graph.
    bound_inputs = [inp for inp in bounds
                    if isinstance(inp, bound_propagation.Bound)]
    input_nodes_indices = [(i,) for i in range(len(bound_inputs))]
    scanner = _RelaxationScanner(self._relaxer)
    graph.backward_propagation(
        scanner, env, {node_ref: env[node_ref]}, input_nodes_indices)

    # Define function to optimise: summary tightness of guaranteed bounds.
    def objective(relax_params):
      lb_min, ub_max = self._bind(
          scanner.node_relaxations, relax_params).concrete_bound_chunk(
              node_ref, graph, env,
              initial_lin_coeffs, initial_offsets, *bounds)
      return jnp.sum(ub_max - lb_min)

    grad_fn = jax.grad(objective)

    # Optimise the relaxation parameters.
    initial_params = {
        index: node_relaxation.initial_params()
        for index, node_relaxation in scanner.node_relaxations}
    initial_state = initial_params, self._opt.init(initial_params)
    def update_state(state):
      params, opt_state = state
      updates, next_opt_state = self._opt.update(grad_fn(params), opt_state)
      next_params = optax.apply_updates(params, updates)
      next_params = {
          index: node_relaxation.project_params(next_params[index])
          for index, node_relaxation in scanner.node_relaxations}
      return next_params, next_opt_state
    relax_params, _ = jax.lax.fori_loop(
        0, self._num_opt_steps, update_state, initial_state)

    # Evaluate the relaxation at these parameters.
    relax_params = jax.lax.stop_gradient(relax_params)
    return self._bind(
        scanner.node_relaxations, relax_params).concrete_bound_chunk(
            node_ref, graph, env, initial_lin_coeffs, initial_offsets, *bounds)

  def _bind(
      self,
      node_relaxations: Dict[
          Index, linear_bound_utils.ParameterizedNodeRelaxation],
      relax_params: Dict[Index, Tensor],
  ) -> LinearBoundBackwardTransform:
    return LinearBoundBackwardTransform(
        linear_bound_utils.BindRelaxerParams(node_relaxations, relax_params),
        self._primitive_needs_concrete_bounds)


class LinearBoundBackwardConcretizer(bound_utils.BackwardConcretizer):
  """Transformation to propagate linear bounds backwards and concretize them."""

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

  def concrete_bound(self, node_ref: jax.core.Var,
                     graph: bound_propagation.PropagationGraph,
                     env: Dict[jax.core.Var, LayerInput],
                     *bounds: Nest[GraphInput]) -> Bound:
    """Perform backward linear bound computation for the node `index`.

    Args:
      node_ref: Reference of the node to obtain a bound for.
      graph: Graph to perform Backward Propagation on.
      env: Environment containing intermediate bound and shape information.
      *bounds: Bounds on the inputs.
    Returns:
      concrete_bound: IntervalBound on the activation at `node_ref`.
    """
    if node_ref in graph.inputs:
      return bounds[graph.inputs.index(node_ref)]

    node = graph_traversal.read_env(env, node_ref)
    nb_act = np.prod(node.shape)

    def bound_chunk(chunk_index: int) -> Tuple[Tensor, Tensor]:
      initial_lin_coeffs, initial_offsets = create_opt_problems(
          node.shape, chunk_index, self._max_chunk_size)

      return self._concretizing_transform.concrete_bound_chunk(
          node_ref, graph, env, initial_lin_coeffs, initial_offsets, *bounds)

    if (self._max_chunk_size == 0) or (nb_act <= self._max_chunk_size):
      flat_lbs, flat_ubs = bound_chunk(0)
    else:
      nb_bound_chunk = math.ceil(nb_act / self._max_chunk_size)
      chunk_indices = jnp.arange(nb_bound_chunk)
      (map_lbs, map_ubs) = jax.lax.map(bound_chunk, chunk_indices)
      # Remove the padding elements
      flat_lbs = jnp.reshape(map_lbs, (-1,))[:nb_act]
      flat_ubs = jnp.reshape(map_ubs, (-1,))[:nb_act]

    concrete_bound = ibp.IntervalBound(
        jnp.reshape(flat_lbs, node.shape),
        jnp.reshape(flat_ubs, node.shape)
    )
    return concrete_bound


def create_opt_problems(
    bound_shape,
    chunk_index,
    nb_parallel_nodes):
  """Create the linear coefficients and constants."""
  obj = utils.objective_chunk(bound_shape, chunk_index, nb_parallel_nodes)
  # Make the objective for all the samples in the batch

  lin_coeffs = obj
  offsets = jnp.zeros(obj.shape[:1])

  return lin_coeffs, offsets


def concretize_backward_bound(backward_linexp: LinearExpression,
                              act_bound: Bound) -> Bound:
  """Compute the value of a backward bound.

  Args:
    backward_linexp: Coefficients of linear functions defined over a layer.
    act_bound: Bound on the activations of that layer.
  Returns:
    bound: A concretized bound on the value of the functions represented by
      backward_linexp.
  """
  act_lower = jnp.expand_dims(act_bound.lower, 0)
  act_upper = jnp.expand_dims(act_bound.upper, 0)

  dims_to_reduce = tuple(range(1, act_lower.ndim))

  lin_coeffs = backward_linexp.lin_coeffs
  all_bounds = (backward_linexp.offset
                + jnp.sum(jnp.minimum(lin_coeffs, 0.) * act_upper
                          + jnp.maximum(lin_coeffs, 0.) * act_lower,
                          dims_to_reduce))
  lower_bound, neg_upper_bound = jnp.split(all_bounds, 2, axis=0)
  upper_bound = -neg_upper_bound

  return ibp.IntervalBound(lower_bound, upper_bound)


CONCRETIZE_ARGS_PRIMITIVE = (
    synthetic_primitives.leaky_relu_p,
    synthetic_primitives.relu_p,
    synthetic_primitives.posbilinear_p,
    synthetic_primitives.posreciprocal_p,
    lax.abs_p,
    lax.exp_p
)

backward_crown_transform = LinearBoundBackwardTransform(
    linear_bound_utils.crown_rvt_relaxer, CONCRETIZE_ARGS_PRIMITIVE)
backward_fastlin_transform = LinearBoundBackwardTransform(
    linear_bound_utils.fastlin_rvt_relaxer, CONCRETIZE_ARGS_PRIMITIVE)
backward_crown_concretizer = LinearBoundBackwardConcretizer(
    backward_crown_transform)
backward_fastlin_concretizer = LinearBoundBackwardConcretizer(
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
