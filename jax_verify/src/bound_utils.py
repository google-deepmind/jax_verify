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

"""Bound propagation utilities."""
import abc
from typing import Dict, Generic, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives


GraphInput = graph_traversal.GraphInput
Bound = bound_propagation.Bound
Index = bound_propagation.Index
TransformContext = bound_propagation.TransformContext
Nest = bound_propagation.Nest
Tensor = jnp.ndarray
LayerInput = Union[Bound, Tensor]
Primitive = bound_propagation.Primitive
T = TypeVar('T')


class FixedBoundApplier(bound_propagation.BoundTransform):
  """Fixed bound constraints.

  Use with `IntersectionBoundTransform` to apply branching decisions
  to existing bounds.
  """

  def __init__(self, fixed_bounds: Dict[Index, Tuple[Tensor, Tensor]]):
    self._fixed_bounds = fixed_bounds

  def input_transform(
      self, context: TransformContext, lower_bound: Tensor, upper_bound: Tensor
  ) -> bound_propagation.Bound:
    return jax_verify.IntervalBound(*self._fixed_bounds[context.index])

  def primitive_transform(
      self, context: TransformContext,
      primitive: jax.core.Primitive, *args, **kwargs
  ) -> bound_propagation.Bound:
    if (context.index not in self._fixed_bounds and
        isinstance(primitive, synthetic_primitives.FakePrimitive)):
      # Bound is missing at the synthetic primitive level.
      # Try and infer the bound from its sub-graph.
      subgraph = kwargs['jax_verify_subgraph']
      return context.subgraph_handler(self, subgraph, *args)

    return jax_verify.IntervalBound(*self._fixed_bounds[context.index])


class BoundRetriever(Generic[T], bound_propagation.GraphTransform[T]):
  """Retrieves bounds' concrete values.

  The concrete values of the bound is only obtained when the bounds are queried.
  """

  def __init__(self, base_transform: bound_propagation.GraphTransform[T]):
    self._base_transform = base_transform
    self._base_bounds = {}

  def input_transform(
      self, context: TransformContext, lower_bound: Tensor, upper_bound: Tensor
  ) -> T:
    bound = self._base_transform.input_transform(
        context, lower_bound, upper_bound)
    self._base_bounds[context.index] = bound
    return bound

  def primitive_transform(
      self, context: TransformContext,
      primitive: jax.core.Primitive, *args, **kwargs
  ) -> T:
    bound = self._base_transform.equation_transform(
        context, primitive, *args, **kwargs)
    self._base_bounds[context.index] = bound
    return bound

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    return self._base_transform.should_handle_as_subgraph(primitive)

  @property
  def concrete_bounds(self):
    return {index: (bound.lower, bound.upper)
            for index, bound in self._base_bounds.items()}

  @property
  def base_transform(self):
    return self._base_transform


class BoundRetrieverAlgorithm(bound_propagation.PropagationAlgorithm[Bound]):
  """Algorithm to collect concrete bounds.

  Compared to a BoundRetriever Transform, this allows to obtain the final bounds
  in the environment, such as when bounds in the environment are modified
  by different transforms.
  """

  def __init__(self,
               base_algorithm: bound_propagation.PropagationAlgorithm[Bound]):
    self._base_algorithm = base_algorithm
    self._base_bounds = {}

  def propagate(
      self, graph: bound_propagation.PropagationGraph,
      bounds: Nest[GraphInput],
  ) -> Tuple[Nest[Bound], Dict[jax.core.Var, LayerInput]]:
    outvals, env = self._base_algorithm.propagate(graph, bounds)
    self._base_bounds = {index: env[graph.jaxpr_node(index)]
                         for index in graph.indices}
    return outvals, env

  @property
  def concrete_bounds(self):
    return {index: (bound.lower, bound.upper)
            for index, bound in self._base_bounds.items()}


class VacuousBoundTransform(bound_propagation.BoundTransform):
  """Generates vacuously loose bounds."""

  def input_transform(
      self, context: TransformContext, lower_bound: Tensor, upper_bound: Tensor
  ) -> bound_propagation.Bound:
    return _vacuous_bounds(lower_bound)

  def primitive_transform(
      self, context: TransformContext,
      primitive: jax.core.Primitive, *args, **kwargs
  ) -> bound_propagation.Bound:
    template_args = [
        jnp.zeros_like(arg.lower)
        if isinstance(arg, bound_propagation.Bound) else arg
        for arg in args]
    return _vacuous_bounds(primitive.bind(*template_args, **kwargs))


def _vacuous_bounds(template: Tensor) -> bound_propagation.Bound:
  ones = jnp.ones_like(template)
  return jax_verify.IntervalBound(-float('inf') * ones, +float('inf') * ones)


class GraphNode(bound_propagation.Bound):
  """Node of a Jax computation graph."""

  def __init__(self, index: Index, primitive, *args, **kwargs):
    self.index = index
    self.primitive = primitive
    self.args = args
    self.kwargs = kwargs

    if self.is_input():
      self._shape = args[0].shape
    else:
      template_args = [
          jnp.zeros(arg.shape) if isinstance(arg, GraphNode) else arg
          for arg in args]
      kwarged_fun = lambda x: primitive.bind(*x, **kwargs)
      self._shape = jax.eval_shape(kwarged_fun, template_args).shape

  def is_input(self):
    return self.primitive is None

  @property
  def shape(self):
    return self._shape

  @property
  def lower(self):
    raise NotImplementedError()

  @property
  def upper(self):
    raise NotImplementedError()


class GraphInspector(bound_propagation.GraphTransform[GraphNode]):
  """Graph traverser that exposes the nodes."""

  def __init__(self, subgraph_decider=None):
    self.nodes = {}
    self._subgraph_decider = subgraph_decider

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    if self._subgraph_decider:
      return self._subgraph_decider(primitive)
    else:
      return super().should_handle_as_subgraph(primitive)

  def input_transform(
      self, context: TransformContext, lower_bound: Tensor, upper_bound: Tensor
  ) -> GraphNode:
    self.nodes[context.index] = GraphNode(context.index,
                                          None, lower_bound, upper_bound)
    return self.nodes[context.index]

  def primitive_transform(
      self, context: TransformContext,
      primitive: jax.core.Primitive, *args, **kwargs
  ) -> GraphNode:
    self.nodes[context.index] = GraphNode(context.index,
                                          primitive, *args, **kwargs)
    return self.nodes[context.index]


class BackwardConcretizingTransform(bound_propagation.BackwardGraphTransform[T],
                                    metaclass=abc.ABCMeta):
  """Abstract class for a Backward Transformation that can concretize bounds."""

  @abc.abstractmethod
  def concretize_args(self, primitive: Primitive) -> bool:
    """Return whether the arguments needs to be concretized.

    Args:
      primitive: Primitive that we are encountering.
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
  def concrete_bound(self, node: jax.core.Var,
                     graph: bound_propagation.PropagationGraph,
                     env: Dict[jax.core.Var, LayerInput],
                     *bounds: Nest[GraphInput]) -> Bound:
    """Perform backward linear bound computation for the node `index`.

    Args:
      node: Graph node to obtain a bound for.
      graph: Graph to perform Backward Propagation on.
      env: Environment containing intermediate bound and shape information.
      *bounds: Bounds on the inputs.
    Returns:
      concrete_bound: IntervalBound on the activation at `node`.
    """


class BackwardConcretizingAlgorithm(
    bound_propagation.PropagationAlgorithm[Bound]):
  """Abstract Backward graph propagation method with forward concretization.

  A trace through the network is first obtained, then backward bound
  computations are performed for each of the node requiring to be concretized,
  such as intermediate nodes that are inputs to non-linearity or output values.
  """

  def __init__(self, backward_concretizer: BackwardConcretizer):
    self._backward_concretizer = backward_concretizer

  def propagate(self,
                graph: bound_propagation.PropagationGraph,
                *bounds: Nest[GraphInput]
                ) -> Tuple[Nest[LayerInput], Dict[jax.core.Var, LayerInput]]:
    subgraph_decider = self._backward_concretizer.should_handle_as_subgraph
    graph_inspector = GraphInspector(subgraph_decider)
    inspector_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        graph_inspector)
    gn_outvals, env = inspector_algorithm.propagate(graph, bounds)

    flat_bounds, _ = jax.tree_util.tree_flatten(bounds)
    for node in graph_inspector.nodes.values():
      # Iterate over the nodes in order so that we get intermediate bounds in
      # the order where we need them.
      node_primitive = node.primitive
      if (node_primitive and
          self._backward_concretizer.concretize_args(node.primitive)):
        for node_arg in node.args:
          if isinstance(node_arg, GraphNode):
            node_index_to_concretize = node_arg.index
            jaxpr_node = graph.jaxpr_node(node_index_to_concretize)
            node_to_concretize = env[jaxpr_node]
            if isinstance(node_to_concretize, GraphNode):
              # This node has not yet been concretized. Perform concretization.
              concrete_bound = self._backward_concretizer.concrete_bound(
                  jaxpr_node, graph, env, *flat_bounds)
              env[jaxpr_node] = concrete_bound

    # Iterate over the outputs, making sure to concretize all of them.
    outvals = []
    for gn in gn_outvals:
      jaxpr_node = graph.jaxpr_node(gn.index)
      env_node = env[jaxpr_node]
      # This node has not been concretized. Perform concretization.
      if isinstance(env_node, GraphNode):
        concrete_bound = self._backward_concretizer.concrete_bound(
            jaxpr_node, graph, env, *flat_bounds)
        env[jaxpr_node] = concrete_bound
      else:
        concrete_bound = env_node
      outvals.append(concrete_bound)

    return outvals, env


class BackwardAlgorithmForwardConcretization(
    bound_propagation.PropagationAlgorithm[Bound]
):
  """Abstract Backward graph propagation with forward concretization."""

  def __init__(self, forward_transform: bound_propagation.BoundTransform,
               backward_concretizer: BackwardConcretizer):
    self._forward_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        forward_transform)
    self._backward_concretizer = backward_concretizer

  def propagate(
      self,
      graph: bound_propagation.PropagationGraph,
      *bounds: Nest[GraphInput],
  ) -> Tuple[Nest[LayerInput], Dict[jax.core.Var, LayerInput]]:
    # Perform the forward propagation so that all intermediate bounds are
    # concretized.
    _, env = self._forward_algorithm.propagate(graph, *bounds)

    flat_bounds, _ = jax.tree_util.tree_flatten(bounds)
    # Iterate over the outputs, computing each one separately.
    outvals = []
    for out_var in graph.outputs:
      concrete_outvar_bound = self._backward_concretizer.concrete_bound(
          out_var, graph, env, *flat_bounds)
      outvals.append(concrete_outvar_bound)
      env[out_var] = concrete_outvar_bound
    return outvals, env
