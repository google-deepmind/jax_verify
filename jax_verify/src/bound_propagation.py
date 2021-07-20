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

"""Propagate the bounds through the network.

This is accomplished by traversing the JaxPR representation of the computation
and translating the computational graph.
"""
import abc
from typing import Callable, Dict, Generic, Tuple, TypeVar, Union

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives


Tensor = jnp.ndarray
Nest = graph_traversal.Nest
Primitive = graph_traversal.Primitive
Index = graph_traversal.Index
GraphInput = graph_traversal.GraphInput
InputBound = graph_traversal.InputBound
JittableInputBound = graph_traversal.JittableInputBound
TransformContext = graph_traversal.TransformContext
PropagationGraph = graph_traversal.PropagationGraph
GraphTransform = graph_traversal.GraphTransform
OpwiseGraphTransform = graph_traversal.OpwiseGraphTransform
UpdatedGraphTransform = graph_traversal.UpdatedGraphTransform
BackwardGraphTransform = graph_traversal.BackwardGraphTransform
BackwardOpwiseTransform = graph_traversal.BackwardOpwiseTransform
Repr = TypeVar('Repr', bound=graph_traversal.TransformedNode)


class Bound(graph_traversal.TransformedNode):
  """Abstract propagated bound."""

  @abc.abstractproperty
  def lower(self) -> Tensor:
    """Concrete lower bound."""

  @abc.abstractproperty
  def upper(self) -> Tensor:
    """Concrete upper bound."""

  @property
  def shape(self) -> Tuple[int]:
    """Shape of the bound."""
    return self.lower.shape

  def unwrap(self) -> 'Bound':
    """Underlying bound of method-specific type, without extra constraints.

    Usually this returns `self`. However, subclasses that wrap the bound to
    provide additional information (for example externally imposed interval
    constraints) will return the wrapped bound.

    Returns:
      Underlying bound arising directly from bound propagation.
    """
    return self


class IntervalBound(Bound, InputBound):
  """Represent an interval where some activations might be valid."""

  def __init__(self, lower_bound: Tensor, upper_bound: Tensor):  # pylint: disable=super-init-not-called
    # Pylint complains that the __init__ method of the base class Bound is not
    # called, despite the fact that Bound does not have an __init__ method.
    self._lower_bound = lower_bound
    self._upper_bound = upper_bound

  @property
  def lower(self) -> Tensor:
    return self._lower_bound

  @property
  def upper(self) -> Tensor:
    return self._upper_bound

  def update_bounds(self, lower, upper):
    self._lower_bound = lower
    self._upper_bound = upper


def unwrapping(fn):
  """Create a wrapper function to unwrap the bound arguments.

  Use as a decorator. If a propagation function has been defined assuming
  its input bound arguments are of the method-specific type
  (e.g. `fastlin.LinearBound`), then applying this decorator will allow the
  function to accept wrapped bound arguments with extra constraints
  (e.g. `intersection.ConstrainedBound`) which it will ignore.

  Args:
    fn: Function accepting possibly wrapped bound arguments.

  Returns:
    Function accepting bound arguments of the method-specific type.
  """
  unwrap = lambda x: x.unwrap() if isinstance(x, Bound) else x
  def fn_unwrapped(*args, **kwargs):
    return fn(*[unwrap(arg) for arg in args], **kwargs)
  return fn_unwrapped


BoundTransform = GraphTransform[Bound]
OpwiseBoundTransform = OpwiseGraphTransform[Bound]
UpdatedBoundTransform = UpdatedGraphTransform[Bound]


class PropagationAlgorithm(Generic[Repr], metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def propagate(
      self,
      graph: PropagationGraph,
      bounds: Nest[GraphInput],
  ) -> Tuple[Nest[Repr], Dict[jax.core.Var, Union[Repr, Tensor]]]:
    """Propagate the given input bounds on the given graph."""


class ForwardPropagationAlgorithm(PropagationAlgorithm[Repr],
                                  metaclass=abc.ABCMeta):
  """Abstract Forward graph propagation method."""

  def __init__(self, graph_transform: GraphTransform):
    self._graph_transform = graph_transform

  def propagate(
      self,
      graph: PropagationGraph,
      bounds: Nest[GraphInput],
  ) -> Tuple[Nest[Repr], Dict[jax.core.Var, Union[Repr, Tensor]]]:
    """Propagate forward the given input bounds on the given graph."""
    return graph.forward_propagation(self._graph_transform, bounds)


def bound_propagation(
    prop_alg: PropagationAlgorithm[Repr],
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[GraphInput],
    graph_simplifier=synthetic_primitives.default_simplifier,
) -> Tuple[
    Nest[Union[Repr, Tensor]],
    Dict[jax.core.Var, Union[Repr, Tensor, Bound]]]:
  """Performs Bound Propagation on the model implemented by `function`.

  Args:
    prop_alg: Algorithm specifying how to traverse the graph and how to
      transform each node.
    function: Pure function inputs -> outputs. If the function to propagate
      through has a more complex signature, the use of `functools.partial` can
      solve that problem.
    *bounds: Nest of `IntervalBound` objects containing the lower and upper
      bounds on all the inputs, or `Tensor`s containing known inputs directly.
    graph_simplifier: Function transforming the JaxPR graph into a simpler
      graph. Default value is a function identifying specific activation
      functions, followed by grouping of linear sequences and quadratic forms.
  Returns:
    bounds: Bounds over all the outputs of the function, with the same structure
      as the output of `function`
    env: Mapping from the node of the computations to their representation.
  """
  # Replace all the jittable bounds by standard bound object.
  is_jittable_bound = lambda b: isinstance(b, JittableInputBound)
  bounds = jax.tree_util.tree_map(
      lambda b: IntervalBound(b.lower, b.upper) if is_jittable_bound(b) else b,
      bounds, is_leaf=is_jittable_bound)

  # Parse the computation graph.
  placeholder_inputs = jax.tree_util.tree_map(
      lambda b: b.lower if isinstance(b, Bound) else b,
      bounds)
  jaxpr_maker = jax.make_jaxpr(function)
  parsed = jaxpr_maker(*placeholder_inputs)
  output_shapes = jax.eval_shape(function, *placeholder_inputs)

  flat_is_bound, _ = jax.tree_util.tree_flatten(
      jax.tree_util.tree_map(lambda b: isinstance(b, Bound), bounds))
  inp_is_bound = {var: is_bound
                  for var, is_bound in zip(parsed.jaxpr.invars, flat_is_bound)}
  simplified_graph = synthetic_primitives.simplify_graph(
      graph_simplifier, parsed.jaxpr, inp_is_bound)
  graph = PropagationGraph(simplified_graph, parsed.literals)

  outvals, env = prop_alg.propagate(graph, bounds)

  # Make outvals into the same tree structure than the output of the function.
  tree_structure = jax.tree_util.tree_structure(output_shapes)
  outvals = jax.tree_util.tree_unflatten(tree_structure, outvals)

  return outvals, env


RESHAPE_PRIMITIVES = [
    lax.reshape_p,
    lax.squeeze_p,
    lax.transpose_p,
    lax.broadcast_in_dim_p,
    lax.concatenate_p,
    lax.gather_p,
    lax.scatter_p,
]


BILINEAR_PRIMITIVES = [
    lax.mul_p,
    lax.dot_general_p,
    lax.conv_general_dilated_p,
]
# Note that synthetic_primitives.posbilinear_p is not present in the list of
# BILINEAR_PRIMITIVES. This is because it's only created by the bilinear
# primitive simplifier and therefore will always be tagging specific bilinear
# operation (and not just bilinear primitive which will be affine because one
# argument is a weight tensor). This allows us to avoid including posbilinear
# into AFFINE_PRIMITIVES.


AFFINE_PRIMITIVES = [
    lax.scatter_add_p,
    lax.add_p,
    lax.sub_p,
    lax.reduce_sum_p,
    synthetic_primitives.linear_p,
] + BILINEAR_PRIMITIVES
# lax.div_p can also be treated as an affine primitive, subject to checking
# that its second arg (the divisor) is a constant.
