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

"""Bound propagation utilities."""
import abc
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import graph_traversal
from jax_verify.src import utils

Index = graph_traversal.Index
GraphInput = graph_traversal.GraphInput
Bound = bound_propagation.Bound
IntervalBound = bound_propagation.IntervalBound
Nest = bound_propagation.Nest
Tensor = jnp.ndarray
LayerInput = bound_propagation.LayerInput
Primitive = bound_propagation.Primitive
T = TypeVar('T', bound=graph_traversal.TransformedNode)


class BackwardConcretizingTransform(
    bound_propagation.BackwardGraphTransform[T],
    Generic[T],  # Explicitly restate this, to aid PyType resolution.
    metaclass=abc.ABCMeta):
  """Abstract class for a Backward Transformation that can concretize bounds."""

  @abc.abstractmethod
  def concretize_args(self, primitive: Primitive) -> bool:
    """Return whether the arguments needs to be concretized.

    Args:
      primitive: Primitive that we are encountering.
    """

  @abc.abstractmethod
  def concrete_bound_chunk(
      self,
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
      env: Dict[jax.core.Var, LayerInput],
      node: jax.core.Var,
      obj: Tensor,
  ) -> Tensor:
    """Computes concrete bounds for a chunk of neurons in the given layer.

    Args:
      graph: Graph to perform Backward Propagation on.
      inputs: Bounds on the inputs.
      env: Environment containing intermediate bound and shape information.
      node: Graph node to obtain a bound for.
      obj: One-hot tensor of shape (chunk_size, *node_shape) specifying, for
        each index in the chunk, an element of the objective. Non-zero entries
        may be +1 or -1 to request lower or upper bounds respectively.

    Returns:
      Bound of shape (chunk_size,) on the activation at `node`.
    """


class BackwardConcretizer(metaclass=abc.ABCMeta):
  """Abstract producer of concretize bounds by back-propagation."""

  @abc.abstractmethod
  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    """Returns whether the primitive should be handled via its sub-graph."""

  @abc.abstractmethod
  def concretize_args(self, primitive: Primitive) -> bool:
    """Return whether the arguments needs to be concretized.

    Args:
      primitive: Primitive that we are encountering.
    """

  @abc.abstractmethod
  def concrete_bound(
      self,
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
      env: Dict[jax.core.Var, LayerInput],
      node: jax.core.Var,
  ) -> jax_verify.IntervalBound:
    """Perform backward linear bound computation for the node `index`.

    Args:
      graph: Graph to perform Backward Propagation on.
      inputs: Bounds on the inputs.
      env: Environment containing intermediate bound and shape information.
      node: Graph node to obtain a bound for.

    Returns:
      concrete_bound: IntervalBound on the activation at `node`.
    """


class ChunkedBackwardConcretizer(BackwardConcretizer):
  """Concretizer that invokes the given transform in chunks for each layer."""

  def __init__(self,
               concretizing_transform: BackwardConcretizingTransform[Any],
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
  ) -> jax_verify.IntervalBound:
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

    return jax_verify.IntervalBound(
        *utils.chunked_bounds(node.shape, self._max_chunk_size, bound_fn))


class BackwardConcretizingAlgorithm(
    bound_propagation.PropagationAlgorithm[Bound]):
  """Abstract Backward graph propagation method with forward concretization.

  A trace through the network is first obtained, then backward bound
  computations are performed for each of the node requiring to be concretized,
  such as intermediate nodes that are inputs to non-linearity or output values.

  Note that the resulting environment of intermediate bounds (returned by
  `propagate` alongside the concrete output bound) is sparse, in the sense that
  many nodes will only contain a placeholder `GraphNode` with vacuous bounds.
  The only nodes guaranteed to contain genuine bounds are the args to
  concretised nodes, and the final outputs. In addition, the graph inputs are
  included in concrete form for convenience.
  """

  def __init__(
      self,
      backward_concretizer: BackwardConcretizer,
      bound_postprocess_fn: Callable[
          [Index, LayerInput], LayerInput] = lambda _, x: x):
    self._backward_concretizer = backward_concretizer
    self._bound_postprocess_fn = bound_postprocess_fn

  def propagate(
      self, graph: bound_propagation.PropagationGraph, inputs: Nest[GraphInput]
  ) -> Tuple[Nest[LayerInput], Dict[jax.core.Var, LayerInput]]:
    subgraph_decider = self._backward_concretizer.should_handle_as_subgraph
    graph_inspector = bound_utils.GraphInspector(subgraph_decider)
    inspector_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        graph_inspector)
    gn_outvals, env = inspector_algorithm.propagate(graph, inputs)

    def lazily_concretize(index, *, is_output):
      jaxpr_node = graph.jaxpr_node(index)
      if isinstance(env[jaxpr_node], bound_utils.GraphNode):
        # This node has not yet been concretized. Perform concretization.
        concrete_bound = self._backward_concretizer.concrete_bound(
            graph, inputs, env, jaxpr_node)
        if not is_output:
          concrete_bound = self._bound_postprocess_fn(index, concrete_bound)
        env[jaxpr_node] = concrete_bound

    # Iterate over the nodes in order so that we get intermediate bounds in
    # the order where we need them.
    for node in graph_inspector.nodes.values():
      if (not node.is_input() and
          self._backward_concretizer.concretize_args(node.primitive)):
        for node_arg in node.args:
          if isinstance(node_arg, Bound):
            lazily_concretize(node_arg.index, is_output=False)

    # Iterate over the outputs, making sure to concretize all of them.
    outvals = []
    for gn in gn_outvals:
      lazily_concretize(gn.index, is_output=True)
      outvals.append(env[graph.jaxpr_node(gn.index)])

    # Fill in the bounds for the inputs.
    # This is unnecessary for backward methods themselves, but may be useful
    # if the resulting `env` is used as a set of base bounds for a forward
    # method.
    flat_inputs, _ = jax.tree_util.tree_flatten(inputs)
    for in_jaxpr_node, in_bound in zip(graph.inputs, flat_inputs):
      if isinstance(env[in_jaxpr_node], bound_utils.GraphNode):
        env[in_jaxpr_node] = in_bound

    return outvals, env


class BackwardAlgorithmForwardConcretization(
    bound_propagation.PropagationAlgorithm[Bound]):
  """Abstract Backward graph propagation with forward concretization."""

  def __init__(self, forward_transform: bound_propagation.BoundTransform,
               backward_concretizer: BackwardConcretizer):
    self._forward_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        forward_transform)
    self._backward_concretizer = backward_concretizer

  def propagate(
      self,
      graph: bound_propagation.PropagationGraph,
      inputs: Nest[GraphInput],
  ) -> Tuple[Nest[LayerInput], Dict[jax.core.Var, LayerInput]]:
    # Perform the forward propagation so that all intermediate bounds are
    # concretized.
    _, env = self._forward_algorithm.propagate(graph, inputs)

    # Iterate over the outputs, computing each one separately.
    outvals = []
    for out_var in graph.outputs:
      concrete_outvar_bound = self._backward_concretizer.concrete_bound(
          graph, inputs, env, out_var)
      outvals.append(concrete_outvar_bound)
      env[out_var] = concrete_outvar_bound
    return outvals, env
