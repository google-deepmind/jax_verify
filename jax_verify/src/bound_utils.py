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

"""Bound propagation utilities."""
from typing import Generic, Mapping, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src.types import Index, Nest, Primitive, SpecFn, Tensor  # pylint: disable=g-multiple-import


T = TypeVar('T', bound=graph_traversal.TransformedNode)


class FixedBoundApplier(bound_propagation.BoundTransform):
  """Fixed bound constraints.

  Use with `IntersectionBoundTransform` to apply branching decisions
  to existing bounds.
  """

  def __init__(self, fixed_bounds: Mapping[Index, Tuple[Tensor, Tensor]]):
    self._fixed_bounds = fixed_bounds

  def input_transform(
      self,
      context: bound_propagation.TransformContext,
      input_bound: graph_traversal.InputBound,
  ) -> bound_propagation.Bound:
    return jax_verify.IntervalBound(*self._fixed_bounds[context.index])

  def primitive_transform(
      self,
      context: bound_propagation.TransformContext,
      primitive: Primitive,
      *args: bound_propagation.LayerInput,
      **kwargs,
  ) -> bound_propagation.Bound:
    if (context.index not in self._fixed_bounds and
        isinstance(primitive, synthetic_primitives.FakePrimitive)):
      # Bound is missing at the synthetic primitive level.
      # Try and infer the bound from its sub-graph.
      subgraph = kwargs['jax_verify_subgraph']
      bound, = context.subgraph_handler(self, subgraph, *args)
      return bound

    return jax_verify.IntervalBound(*self._fixed_bounds[context.index])


class BoundRetriever(Generic[T], graph_traversal.GraphTransform[T]):
  """Retrieves bounds' concrete values.

  The concrete values of the bound is only obtained when the bounds are queried.
  """

  def __init__(self, base_transform: graph_traversal.GraphTransform[T]):
    self._base_transform = base_transform
    self._base_bounds = {}

  def input_transform(
      self,
      context: graph_traversal.TransformContext[T],
      input_bound: graph_traversal.InputBound,
  ) -> T:
    bound = self._base_transform.input_transform(context, input_bound)
    self._base_bounds[context.index] = bound
    return bound

  def primitive_transform(
      self,
      context: graph_traversal.TransformContext[T],
      primitive: Primitive,
      *args: graph_traversal.LayerInput[T],
      **kwargs,
  ) -> T:
    bound, = self._base_transform.equation_transform(
        context, primitive, *args, **kwargs)
    self._base_bounds[context.index] = bound
    return bound

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    return self._base_transform.should_handle_as_subgraph(primitive)

  @property
  def concrete_bounds(self) -> Mapping[Index, Tuple[Tensor, Tensor]]:
    return {index: (bound.lower, bound.upper)
            for index, bound in self._base_bounds.items()}

  @property
  def base_transform(self):
    return self._base_transform


class BoundRetrieverAlgorithm(
    bound_propagation.PropagationAlgorithm[bound_propagation.Bound]):
  """Algorithm to collect concrete bounds.

  Compared to a BoundRetriever Transform, this allows to obtain the final bounds
  in the environment, such as when bounds in the environment are modified
  by different transforms.
  """

  def __init__(
      self,
      base_algorithm: bound_propagation.PropagationAlgorithm[
          bound_propagation.Bound]):
    self._base_algorithm = base_algorithm
    self._base_bounds = {}

  def propagate(
      self,
      graph: graph_traversal.PropagationGraph,
      inputs: Nest[graph_traversal.GraphInput],
  ) -> Tuple[
      Nest[bound_propagation.Bound],
      Mapping[jax.core.Var, bound_propagation.LayerInput],
  ]:
    outvals, env = self._base_algorithm.propagate(graph, inputs)
    self._base_bounds = {
        index: graph_traversal.read_env(env, graph.jaxpr_node(index))
        for index in graph.indices}
    return outvals, env

  @property
  def concrete_bounds(self) -> Mapping[Index, Tuple[Tensor, Tensor]]:
    return {index: (bound.lower, bound.upper)
            for index, bound in self._base_bounds.items()}


class VacuousBoundTransform(bound_propagation.BoundTransform):
  """Generates vacuously loose bounds."""

  def input_transform(
      self,
      context: bound_propagation.TransformContext,
      input_bound: graph_traversal.InputBound
  ) -> bound_propagation.Bound:
    return _vacuous_bounds(input_bound.lower)

  def primitive_transform(
      self,
      context: bound_propagation.TransformContext,
      primitive: Primitive,
      *args: bound_propagation.LayerInput,
      **kwargs,
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
      input_bound: graph_traversal.InputBound = args[0]
      self._shape = input_bound.lower.shape
      self._dtype = input_bound.lower.dtype
    else:
      kwarged_fun = lambda x: primitive.bind(*x, **kwargs)
      shape_and_type = jax.eval_shape(kwarged_fun, args)
      self._shape = shape_and_type.shape
      self._dtype = shape_and_type.dtype

  def is_input(self):
    return self.primitive is None

  @property
  def shape(self):
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  @property
  def lower(self):
    return -float('inf') * jnp.ones(self._shape, self._dtype)

  @property
  def upper(self):
    return float('inf') * jnp.ones(self._shape, self._dtype)


class GraphInspector(graph_traversal.GraphTransform[GraphNode]):
  """Graph traverser that exposes the nodes."""

  def __init__(self, subgraph_decider=None):
    self._nodes = {}
    self._subgraph_decider = subgraph_decider

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    if self._subgraph_decider:
      return self._subgraph_decider(primitive)
    else:
      return super().should_handle_as_subgraph(primitive)

  def input_transform(
      self,
      context: graph_traversal.TransformContext[GraphNode],
      input_bound: graph_traversal.InputBound,
  ) -> GraphNode:
    self._nodes[context.index] = GraphNode(context.index, None, input_bound)
    return self._nodes[context.index]

  def primitive_transform(
      self,
      context: graph_traversal.TransformContext[GraphNode],
      primitive: Primitive,
      *args: graph_traversal.LayerInput[GraphNode],
      **kwargs,
  ) -> GraphNode:
    self._nodes[context.index] = GraphNode(
        context.index, primitive, *args, **kwargs)
    return self._nodes[context.index]

  @property
  def nodes(self) -> Mapping[Index, GraphNode]:
    return self._nodes


def computation_graph_nodes(
    spec_fn: SpecFn,
    *init_bound: Nest[graph_traversal.GraphInput],
) -> Mapping[Index, GraphNode]:
  """Extract a mapping from index to primitives."""
  inspector = GraphInspector()
  bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(inspector),
      spec_fn, *init_bound)
  return inspector.nodes
