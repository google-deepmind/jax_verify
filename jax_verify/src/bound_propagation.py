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

"""Propagate the bounds through the network.

This is accomplished by traversing the JaxPR representation of the computation
and translating the computational graph.
"""
import abc
import collections
from typing import Generic, Mapping, Sequence, Tuple, TypeVar, Union

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src.types import Nest, Primitive, SpecFn, Tensor  # pylint: disable=g-multiple-import


Repr = TypeVar('Repr', bound=graph_traversal.TransformedNode)


# (lower, upper) input bound represented as a Tensor nest so that it can
# be passed as a parameter to a Jax jitted function.
# `bound_type` should be a dictionary using the desired bound class as a key
# (the value that the key maps to is unimportant). This way, jax does not
# complain about it not being a jax type.
# Example: {jax_verify.IntervalBound: None}
# Additional arguments can be provided in `kwargs`,
# The Bound class should implement a `from_jittable` class method, to
# instantiate the object based on the jittable bound.
JittableInputBound = collections.namedtuple(
    'JittableInputBound', ['lower', 'upper', 'bound_type', 'kwargs'])
JittableGraphInput = Union[JittableInputBound, Tensor]


class Bound(graph_traversal.TransformedNode, metaclass=abc.ABCMeta):
  """Abstract propagated bound."""

  @property
  @abc.abstractmethod
  def lower(self) -> Tensor:
    """Concrete lower bound."""

  @property
  @abc.abstractmethod
  def upper(self) -> Tensor:
    """Concrete upper bound."""

  @property
  def shape(self) -> Tuple[int, ...]:
    """Shape of the node."""
    return self.lower.shape

  @property
  def dtype(self) -> jnp.dtype:
    """Data type of the node."""
    return self.lower.dtype

  def unwrap(self) -> 'Bound':
    """Underlying bound of method-specific type, without extra constraints.

    Usually this returns `self`. However, subclasses that wrap the bound to
    provide additional information (for example externally imposed interval
    constraints) will return the wrapped bound.

    Returns:
      Underlying bound arising directly from bound propagation.
    """
    return self


LayerInput = graph_traversal.LayerInput[Bound]
TransformContext = graph_traversal.TransformContext[Bound]


class IntervalBound(Bound, graph_traversal.InputBound):
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

  @classmethod
  def from_jittable(cls, jittable_bound: JittableInputBound):
    return cls(jittable_bound.lower, jittable_bound.upper)

  def to_jittable(self) -> JittableInputBound:
    return JittableInputBound(self.lower, self.upper, {IntervalBound: None}, {})

  def project_onto_bound(self, tensor: Tensor) -> Tensor:
    return jnp.clip(tensor, a_min=self._lower_bound, a_max=self._upper_bound)


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


BoundTransform = graph_traversal.GraphTransform[Bound]


class PropagationAlgorithm(Generic[Repr], metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def propagate(
      self,
      graph: graph_traversal.PropagationGraph,
      bounds: Nest[graph_traversal.GraphInput],
  ) -> Tuple[Nest[Repr], Mapping[jax.core.Var, Union[Repr, Tensor]]]:
    """Propagate the given input bounds on the given graph."""


class ForwardPropagationAlgorithm(PropagationAlgorithm[Repr]):
  """Forward graph propagation method."""

  def __init__(self, graph_transform: graph_traversal.GraphTransform[Repr]):
    self._graph_transform = graph_transform

  def propagate(
      self,
      graph: graph_traversal.PropagationGraph,
      bounds: Nest[graph_traversal.GraphInput],
  ) -> Tuple[Nest[Repr], Mapping[jax.core.Var, Union[Repr, Tensor]]]:
    """Propagate forward the given input bounds on the given graph."""
    return graph.forward_propagation(self._graph_transform, bounds)


def jit_inputs(
    *inputs: Nest[graph_traversal.GraphInput],
) -> Sequence[Nest[JittableGraphInput]]:
  """Replace all the bound objects by jittable bounds."""
  is_bound = lambda b: isinstance(b, Bound)
  jit_bound = lambda b: b.to_jittable()
  return jax.tree_util.tree_map(
      lambda b: jit_bound(b) if is_bound(b) else b,
      inputs, is_leaf=is_bound)


def unjit_inputs(
    *inputs: Nest[JittableGraphInput],
) -> Sequence[Nest[graph_traversal.GraphInput]]:
  """Replace all the jittable bounds by standard bound objects."""
  is_jittable_bound = lambda b: isinstance(b, JittableInputBound)
  unjit_bound = lambda b: next(iter(b.bound_type)).from_jittable(b)
  return jax.tree_util.tree_map(
      lambda b: unjit_bound(b) if is_jittable_bound(b) else b,
      inputs, is_leaf=is_jittable_bound)


def bound_propagation(
    prop_alg: PropagationAlgorithm[Repr],
    function: SpecFn,
    *bounds: Nest[graph_traversal.GraphInput],
    graph_simplifier=synthetic_primitives.default_simplifier,
) -> Tuple[
    Nest[Union[Repr, Tensor]],
    Mapping[jax.core.Var, Union[Repr, Tensor, Bound]]]:
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
  # Parse the computation graph.
  placeholder_inputs = jax.tree_util.tree_map(
      lambda b: b.lower if isinstance(b, graph_traversal.InputBound) else b,
      bounds)
  parsed = synthetic_primitives.make_jaxpr_nojit(function, *placeholder_inputs)
  output_shapes = jax.eval_shape(function, *placeholder_inputs)

  flat_is_bound, _ = jax.tree_util.tree_flatten(
      jax.tree_util.tree_map(
          lambda b: isinstance(b, graph_traversal.InputBound), bounds))
  inp_is_bound = {var: is_bound
                  for var, is_bound in zip(parsed.jaxpr.invars, flat_is_bound)}
  simplified_graph = synthetic_primitives.simplify_graph(
      graph_simplifier, parsed.jaxpr, inp_is_bound)
  graph = graph_traversal.PropagationGraph(simplified_graph, parsed.literals)

  outvals, env = prop_alg.propagate(graph, bounds)

  # Make outvals into the same tree structure than the output of the function.
  tree_structure = jax.tree_util.tree_structure(output_shapes)
  outvals = jax.tree_util.tree_unflatten(tree_structure, outvals)

  return outvals, env


RESHAPE_PRIMITIVES: Sequence[Primitive] = [
    lax.copy_p,
    lax.reshape_p,
    lax.slice_p,
    lax.dynamic_slice_p,
    lax.squeeze_p,
    lax.transpose_p,
    lax.broadcast_in_dim_p,
    lax.concatenate_p,
    lax.gather_p,
    lax.scatter_p,
    *([lax.select_p] if hasattr(lax, 'select_p') else []),
    *([lax.select_n_p] if hasattr(lax, 'select_n_p') else []),
    synthetic_primitives.convert_float32_p,
]


BILINEAR_PRIMITIVES: Sequence[Primitive] = [
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


AFFINE_PRIMITIVES: Sequence[Primitive] = [
    lax.scatter_add_p,
    lax.add_p,
    lax.sub_p,
    lax.reduce_sum_p,
    lax.neg_p,
    synthetic_primitives.linear_p,
    *BILINEAR_PRIMITIVES,
]
# lax.div_p can also be treated as an affine primitive, subject to checking
# that its second arg (the divisor) is a constant.
