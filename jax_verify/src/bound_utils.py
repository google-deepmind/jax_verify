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
from typing import Dict, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives


GraphInput = graph_traversal.GraphInput
InputBound = graph_traversal.InputBound
Bound = bound_propagation.Bound
Index = bound_propagation.Index
TransformContext = bound_propagation.TransformContext
Nest = bound_propagation.Nest
Tensor = jnp.ndarray
LayerInput = bound_propagation.LayerInput
Primitive = bound_propagation.Primitive
T = TypeVar('T', bound=graph_traversal.TransformedNode)


class FixedBoundApplier(bound_propagation.BoundTransform):
  """Fixed bound constraints.

  Use with `IntersectionBoundTransform` to apply branching decisions
  to existing bounds.
  """

  def __init__(self, fixed_bounds: Dict[Index, Tuple[Tensor, Tensor]]):
    self._fixed_bounds = fixed_bounds

  def input_transform(
      self, context: TransformContext, input_bound: InputBound
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
      self, context: TransformContext, input_bound: InputBound
  ) -> T:
    bound = self._base_transform.input_transform(context, input_bound)
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
  def concrete_bounds(self) -> Dict[Index, Tuple[Tensor, Tensor]]:
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
      inputs: Nest[GraphInput],
  ) -> Tuple[Nest[Bound], Dict[jax.core.Var, LayerInput]]:
    outvals, env = self._base_algorithm.propagate(graph, inputs)
    self._base_bounds = {
        index: graph_traversal.read_env(env, graph.jaxpr_node(index))
        for index in graph.indices}
    return outvals, env

  @property
  def concrete_bounds(self) -> Dict[Index, Tuple[Tensor, Tensor]]:
    return {index: (bound.lower, bound.upper)
            for index, bound in self._base_bounds.items()}


class VacuousBoundTransform(bound_propagation.BoundTransform):
  """Generates vacuously loose bounds."""

  def input_transform(
      self, context: TransformContext, input_bound: InputBound
  ) -> bound_propagation.Bound:
    return _vacuous_bounds(input_bound.lower)

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
      input_bound: InputBound = args[0]
      self._shape = input_bound.lower.shape
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
    return -float('inf') * jnp.ones(self._shape)

  @property
  def upper(self):
    return float('inf') * jnp.ones(self._shape)


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
      self, context: TransformContext, input_bound: InputBound
  ) -> GraphNode:
    self.nodes[context.index] = GraphNode(context.index, None, input_bound)
    return self.nodes[context.index]

  def primitive_transform(
      self, context: TransformContext,
      primitive: jax.core.Primitive, *args, **kwargs
  ) -> GraphNode:
    self.nodes[context.index] = GraphNode(context.index,
                                          primitive, *args, **kwargs)
    return self.nodes[context.index]
