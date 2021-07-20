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

"""Traverses a network, applying a transformation to each node in the graph.

This is accomplished by traversing the JaxPR representation of the computation.
"""
import abc
import collections
import functools
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import dataclasses
import jax
import jax.numpy as jnp

from jax_verify.src import synthetic_primitives


Tensor = jnp.ndarray
T = TypeVar('T')
Nest = Union[T, Sequence[T], Dict[Any, T]]
Primitive = Union[jax.core.Primitive, synthetic_primitives.FakePrimitive]
Index = Tuple[int]


class InputBound(metaclass=abc.ABCMeta):
  """Abstract input bound."""

  @abc.abstractproperty
  def lower(self) -> Tensor:
    """Concrete lower bound."""

  @abc.abstractproperty
  def upper(self) -> Tensor:
    """Concrete upper bound."""


# (lower, upper) input bound represented as a Tensor nest so that it can
# be passed as a parameter to a Jax jitted function.
JittableInputBound = collections.namedtuple(
    'JittableInputBound', ['lower', 'upper'])


GraphInput = Union[InputBound, JittableInputBound, Tensor]


class TransformedNode(metaclass=abc.ABCMeta):
  """Abstract transformed node, e.g. a propagated bound."""


Repr = TypeVar('Repr')
BackRepr = TypeVar('BackRepr')


@dataclasses.dataclass
class TransformContext:
  """Transform context.

  Attributes:
    index: Integer path identifying the input node.
    subgraph_handler: Function to recursively handle a sub-graph.
  """

  index: Index
  subgraph_handler: Optional[Callable[..., Any]]


class GraphTransform(Generic[Repr], metaclass=abc.ABCMeta):
  """Abstract forward Node transformation method."""

  @abc.abstractmethod
  def input_transform(
      self,
      context: TransformContext,
      lower_bound: Tensor,
      upper_bound: Tensor,
  ) -> Repr:
    """Constructs input representations from lower/upper bound tensors.

    Args:
      context: Transform context containing node index.
      lower_bound: Original concrete lower bound on the input.
      upper_bound: Original concrete upper bound on the input.

    Returns:
      Method-specific representation for the inputs.
    """

  @abc.abstractmethod
  def primitive_transform(
      self,
      context: TransformContext,
      primitive: Primitive,
      *args: Union[Repr, Tensor],
      **params,
  ) -> Repr:
    """Applies the given primitive operation to its arguments' representations.

    Args:
      context: Transform context containing node index.
      primitive: Primitive Jax operation to transform.
      *args: Arguments of the primitive. Arguments are expressed as
        method-specific representations if they have any dependence on the
        network's original inputs, or tensors otherwise.
      **params: Keyword arguments of the primitive.

    Returns:
      Method-specific representation for the operation's output.
    """

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    """Returns whether the primitive should be handled via its sub-graph.

    If not, it will be handled by treating it as a single primitive.

    This function is intended to be overridden. The default implementation
    specifies that all synthetic primitives are to be handled as primitives.

    Args:
      primitive: Primitive Jax operation to transform.

    Returns:
      Whether to handle as a single primitive, as opposed to recursively
      processing the sub-graph's child nodes.
    """
    return primitive in (
        jax.custom_derivatives.custom_jvp_call_jaxpr_p, jax.xla.xla_call_p)

  def equation_transform(
      self,
      context: TransformContext,
      primitive: Primitive,
      *args: Union[Repr, Tensor],
      **params,
  ) -> Repr:
    """Applies the given primitive operation to its arguments' representations.

    By default this invokes `primitive_transform()` which will be overridden
    for primitive-specific behaviour. However, this function has an additional
    `subgraph_handler` argument allowing primitive-specific customisation of
    sub-graph handling.

    Args:
      context: Transform context containing node index.
      primitive: Primitive Jax operation to transform.
      *args: Arguments of the primitive. Arguments are expressed as
        method-specific representations if they have any dependence on the
        network's original inputs, or tensors otherwise.
      **params: Keyword arguments of the primitive.

    Returns:
      Method-specific representation for the operation's output,
    """
    if self.should_handle_as_subgraph(primitive):
      if primitive == jax.custom_derivatives.custom_jvp_call_jaxpr_p:
        subgraph = params['fun_jaxpr'].jaxpr
      elif primitive == jax.xla.xla_call_p:
        subgraph = params['call_jaxpr']
      else:
        subgraph = params['jax_verify_subgraph']
      return context.subgraph_handler(self, subgraph, *args)
    return self.primitive_transform(context, primitive, *args, **params)


class BackwardGraphTransform(Generic[BackRepr], metaclass=abc.ABCMeta):
  """Abstract Backward graph propagation method."""

  @abc.abstractmethod
  def primitive_backtransform(
      self,
      context: TransformContext,
      primitive: Primitive,
      eqn_outval: BackRepr,
      *args: Union[TransformedNode, Tensor],
      **params) -> Sequence[Sequence[Optional[BackRepr]]]:
    """Propagate backward to the `*args` inputs of `primitive`."""

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    """Returns whether the primitive should be handled via its sub-graph.

    If not, it will be handled by treating it as a single primitive.

    This function is intended to be overridden. The default implementation
    specifies that all synthetic primitives are to be handled as primitives.

    Args:
      primitive: Primitive Jax operation to transform.

    Returns:
      Whether to handle as a single primitive, as opposed to recursively
      processing the sub-graph's child nodes.
    """
    return primitive in (
        jax.custom_derivatives.custom_jvp_call_jaxpr_p, jax.xla.xla_call_p)

  def equation_backtransform(
      self,
      context: TransformContext,
      primitive: Primitive,
      eqn_outval: BackRepr,
      *args: Union[TransformedNode, Tensor],
      **params,
  ) -> Sequence[Sequence[Optional[BackRepr]]]:
    """Applies the given primitive operation to its arguments' representations.

    Normally this invokes `primitive_backtransform()` which will be overridden
    for primitive-specific behaviour. However, as can be customised in
    `should_handle_as_subgraph`, this may decide instead to process the
    subgraph's child nodes recursively.

    Args:
      context: Transform context containing node index.
      primitive: Primitive Jax operation to transform.
      eqn_outval: Backward representation of the output of the primitive.
      *args: Arguments of the primitive. Arguments need to either be a bound
        or a tensor if they do not depend on the input.
      **params: Keyword arguments of the primitive.

    Returns:
      Method-specific representation for each of the operation's input, or None
      if the input is a Tensor.
    """
    if self.should_handle_as_subgraph(primitive):
      if primitive == jax.custom_derivatives.custom_jvp_call_jaxpr_p:
        subgraph = params['fun_jaxpr'].jaxpr
      elif primitive == jax.xla.xla_call_p:
        subgraph = params['call_jaxpr']
      else:
        subgraph = params['jax_verify_subgraph']
      return context.subgraph_handler(self, subgraph, eqn_outval)
    return self.primitive_backtransform(
        context, primitive, eqn_outval, *args, **params)

  @abc.abstractmethod
  def aggregate(self, eqn_outvals: Sequence[BackRepr]) -> BackRepr:
    """Aggregate the representations coming from different branches."""


class BackwardOpwiseTransform(BackwardGraphTransform[BackRepr]):
  """Backward Propagation method defined by functions for each primitive op."""

  def __init__(
      self,
      primitive_backtransform: Dict[
          Primitive, Callable[..., Sequence[Optional[BackRepr]]]],
      aggregation_fun: Callable[[Sequence[BackRepr]], BackRepr]):
    self._primitive_backtransform = primitive_backtransform
    self._aggregation_fun = aggregation_fun

  def primitive_backtransform(
      self,
      context: TransformContext,
      primitive: synthetic_primitives.PrimitiveLike,
      eqn_outval: BackRepr,
      *args: Union[TransformedNode, Tensor],
      **params) -> Sequence[Sequence[Optional[BackRepr]]]:
    if primitive not in self._primitive_backtransform:
      raise NotImplementedError(f'Unknown Primitive: {primitive}.')
    # Promote each output to a single-length list.
    # This is for consistency with the sub-graph handler, in which output is
    # returned as a list according to the multiple paths through the graph.
    return list(zip(self._primitive_backtransform[primitive](
        context, eqn_outval, *args, **params)))

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    if (isinstance(primitive, synthetic_primitives.FakePrimitive) and
        primitive not in self._primitive_backtransform):
      return True
    return super().should_handle_as_subgraph(primitive)

  def aggregate(self, eqn_outvals: Sequence[BackRepr]) -> BackRepr:
    """Aggregate the representations coming from different branches."""
    return self._aggregation_fun(eqn_outvals)


class OpwiseGraphTransform(GraphTransform[Repr]):
  """Bound propagation method defined by functions for each primitive op."""

  def __init__(
      self,
      input_transform: Callable[[Index, Tensor, Tensor], Repr],
      primitive_transform: Dict[Primitive, Callable[..., Repr]]):
    self._input_transform = input_transform
    self._primitive_transform = primitive_transform

  def input_transform(self, context, lower_bound, upper_bound):
    return self._input_transform(context, lower_bound, upper_bound)

  def primitive_transform(self, context, primitive, *args, **params):
    if primitive not in self._primitive_transform:
      raise NotImplementedError(f'Unknown Primitive: {primitive}')
    return self._primitive_transform[primitive](context, *args, **params)

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    if (isinstance(primitive, synthetic_primitives.FakePrimitive) and
        primitive not in self._primitive_transform):
      return True
    return super().should_handle_as_subgraph(primitive)


class UpdatedGraphTransform(GraphTransform[Repr]):
  """Graph transform with a base transform and updated primitive transform."""

  def __init__(
      self,
      base_transform: GraphTransform[Repr],
      updated_primitive_transform: Dict[Primitive, Callable[..., Repr]]):
    self._base_transform = base_transform
    self._updated_primitive_transform = updated_primitive_transform

  def input_transform(self, context, lower_bound, upper_bound):
    return self._base_transform.input_transform(
        context, lower_bound, upper_bound)

  def primitive_transform(self, context, primitive, *args, **params):
    if primitive in self._updated_primitive_transform:
      return self._updated_primitive_transform[primitive](
          context.index, *args, **params)
    return self._base_transform.equation_transform(
        context, primitive, *args, **params)

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    if (isinstance(primitive, synthetic_primitives.FakePrimitive) and
        primitive not in self._updated_primitive_transform):
      return True
    return self._base_transform.should_handle_as_subgraph(primitive)


JaxVar = Union[jax.core.Var, jax.core.Literal]


class _Ref:

  def __init__(self, target_var: JaxVar):
    self.target_var: JaxVar = target_var


class IndexCounter:
  """Maintains and navigates a path consisting of integers."""

  def __init__(self, initial_index: Optional[Index] = None):
    self._index = [0] if initial_index is None else list(initial_index)

  def incr(self):
    self._index[-1] += 1

  def decr(self):
    self._index[-1] -= 1

  def begin_child(self, initial_index: int = 0):
    self._index.append(initial_index)

  def end_child(self):
    del self._index[-1]

  def child_index(self) -> int:
    return self._index[-1]

  def as_tuple(self) -> Index:
    return tuple(self._index)


def read_env(env: Dict[jax.core.Var, Union[Repr, Tensor, _Ref]],
             var: JaxVar) -> Union[Repr, Tensor]:
  """Read the value from the environment."""
  if isinstance(var, jax.core.Literal):
    # Literals are values baked into the Jaxpr, e.g. ReLU's '0' arg to 'max'.
    return var.val
  else:
    val = env[var]
    if isinstance(val, _Ref):
      # The value is a reference to another variable. Follow the reference.
      return read_env(env, val.target_var)
    elif isinstance(val, list):
      # This is a list so we want to ensure that we resolve the contents if they
      # contain _Ref, and collect everything into a single list.
      flattened_list = []
      for elt in val:
        if isinstance(elt, _Ref):
          flattened_list.extend(read_env(env, elt.target_var))
        elif elt is not None:
          flattened_list.append(elt)
      return flattened_list
    else:
      return val


class PropagationGraph:
  """Holds a computational graph and the environment holding the variables.
  """

  def __init__(self, graph: jax.core.Jaxpr, literals: Sequence[Tensor]):
    self._graph = graph
    self._literals = literals
    self._index_to_node: Dict[Index, jax.core.Var] = {}
    self._last_index = None

  @property
  def inputs(self):
    return self._graph.invars

  @property
  def outputs(self):
    return self._graph.outvars

  def jaxpr_node(self, index: Index) -> jax.core.Var:
    return self._index_to_node[index]

  @property
  def indices(self) -> Sequence[Index]:
    return self._index_to_node.keys()

  def forward_propagation(
      self,
      transform: GraphTransform[Repr],
      bounds: Nest[GraphInput],
  ) -> Tuple[Nest[Repr], Dict[jax.core.Var, Union[Repr, Tensor, _Ref]]]:
    """Performs forward propagation on the parsed computation graph.

    This is accomplished by traversing the JaxPR representation of the
    computation and translating the computational graph, using the
    `primitive_transform` replacement for each jax primitive.

    Args:
      transform: Basic Jax primitive ops' equivalents for operating on
        the representation (e.g. bound propagation).
      bounds: Nest of `InputBound` objects containing the lower and upper
        bounds on all the inputs, or `Tensor`s containing known inputs directly.
    Returns:
      outvals: Propagated values corresponding to the graph output.
      env: Dictionary holding the computed bounds on nodes of the graph.
    """

    # Initialize the environment based on the provided bounds.
    env = {}
    is_bound = lambda b: isinstance(b, (TransformedNode, JittableInputBound))
    flat_bounds, _ = jax.tree_util.tree_flatten(bounds, is_leaf=is_bound)
    invals = []
    index = IndexCounter()
    for bound, invar in zip(flat_bounds, self._graph.invars):
      if is_bound(bound):
        input_val = transform.input_transform(
            TransformContext(index.as_tuple(), None),
            bound.lower, bound.upper)
        self._index_to_node[index.as_tuple()] = invar
        index.incr()
      else:
        # Input is a fixed tensor.
        input_val = bound
      invals.append(input_val)
    env.update(zip(self._graph.invars, invals))
    env.update(zip(self._graph.constvars, self._literals))

    for eqn in self._graph.eqns:
      self._forward_prop_eqn(transform, env, index, eqn)
    self._last_index = index.as_tuple()

    outvals = jax.tree_map(functools.partial(read_env, env),
                           self._graph.outvars)
    return outvals, env

  def _forward_prop_eqn(self, transform: GraphTransform[Repr],
                        env: Dict[jax.core.Var, Union[Repr, Tensor, _Ref]],
                        index: IndexCounter,
                        eqn: jax.core.JaxprEqn):
    """Recursive step of `forward_propagation`."""
    def subgraph_handler(sub_transform, sub_graph, *invals):
      assert len(invals) == len(sub_graph.invars) == len(eqn.invars)
      use_ref = not isinstance(eqn.primitive,
                               synthetic_primitives.FakePrimitive)
      if sub_transform is transform:
        # We must use the same environment, so that transformed equations
        # in the sub-graph are included in it.
        assert all(read_env(env, invar) is inval
                   for invar, inval in zip(eqn.invars, invals))
        invals = [_Ref(invar) if use_ref else read_env(env, invar)
                  for invar in eqn.invars]
        sub_env = env
      else:
        # We must leave the environment intact, because it contains
        # equations transformed by a different transformer.
        # Set up a new environment.
        assert not use_ref
        sub_env = {}

      # Add the sub-graph's inputs to the environment.
      sub_env.update({
          sub_invar: inval for sub_invar, inval in zip(sub_graph.invars, invals)
          if isinstance(sub_invar, jax.core.Var)})
      # Recursively propagate through the sub-graph.
      index.begin_child()
      try:
        for sub_eqn in sub_graph.eqns:
          self._forward_prop_eqn(sub_transform, sub_env, index, sub_eqn)
      finally:
        index.end_child()
      # Associate the sub-graph's outputs with the enclosing equation outputs.
      eqn_outvals = [
          _Ref(sub_outvar) if use_ref else read_env(sub_env, sub_outvar)
          for sub_outvar in sub_graph.outvars]
      return eqn_outvals[0] if len(eqn.outvars) == 1 else eqn_outvals

    eqn_invars_vals = jax.tree_map(functools.partial(read_env, env), eqn.invars)
    idx = index.as_tuple()
    if any(isinstance(inval, TransformedNode) for inval in eqn_invars_vals):
      if len(eqn.outvars) != 1:
        # TODO Handle multi out primitives.
        raise NotImplementedError('Multiple output primitives not supported.')
      if (idx in self._index_to_node
          and self._index_to_node[idx] != eqn.outvars[0]):
        raise ValueError('A node with this index pointing to another Node'
                         'already exists.')
      eqn_outval = transform.equation_transform(
          TransformContext(idx, subgraph_handler),
          eqn.primitive, *eqn_invars_vals, **eqn.params)
      self._index_to_node[idx] = eqn.outvars[0]
    else:
      # No dependence on the network's inputs. Any primitive is supported, but
      # we need to be careful with regards to how to execute xla_call_p
      # primitives
      if eqn.primitive == jax.xla.xla_call_p:
        sub_graph = eqn.params['call_jaxpr']
        eqn_outval = subgraph_handler(transform, sub_graph, *eqn_invars_vals)
      else:
        eqn_outval = eqn.primitive.bind(*eqn_invars_vals, **eqn.params)

    index.incr()
    if len(eqn.outvars) == 1:
      env[eqn.outvars[0]] = eqn_outval
    else:
      env.update({
          outvar: outval
          for outvar, outval in zip(eqn.outvars, eqn_outval)})

  def _propagate_backward(
      self,
      transform: BackwardGraphTransform[BackRepr],
      forward_env: Dict[jax.core.Var, Union[TransformedNode, Tensor, _Ref]],
      backward_env: Dict[jax.core.Var, List[Union[BackRepr, _Ref]]],
      target_indices: Sequence[Index],
  ) -> Tuple[
      Sequence[Optional[BackRepr]],
      Dict[jax.core.Var, Sequence[Union[BackRepr, _Ref]]]]:
    """Performs backward propagation on the parsed computational graph.

    Args:
      transform: Backward Transformation to implement reverse bound propagation.
      forward_env: Environment providing bounds over the nodes of the network,
        as obtained by a forward propagation.
      backward_env: Backward environment to hold the bounds being propagated
        backward.
      target_indices: Indices of the nodes for which we want to obtain
        backward bounds.
    Returns:
      targets: Propagated values corresponding to the required target_indices.
      backward_env: Backward environment holding the bounds being propagated
        backward.
    """
    index = IndexCounter(self._last_index)
    limit_index = sorted(target_indices)[0]
    for eqn in reversed(self._graph.eqns):
      self._backward_prop_eqn(transform, forward_env, backward_env, index, eqn)
      if index.as_tuple() <= limit_index:
        break

    targets = []
    for target_idx in target_indices:
      node_ref = self._index_to_node[target_idx]
      if node_ref in backward_env:
        targets.append(transform.aggregate(read_env(backward_env, node_ref)))
      else:
        targets.append(None)
    return targets, backward_env

  def backward_propagation(
      self,
      transform: BackwardGraphTransform[BackRepr],
      forward_env: Dict[jax.core.Var, Union[TransformedNode, Tensor, _Ref]],
      backward_bounds: Dict[jax.core.Var, BackRepr],
      target_indices: Sequence[Index],
  ) -> Tuple[
      Sequence[Optional[BackRepr]],
      Dict[jax.core.Var, Sequence[Union[BackRepr, _Ref]]]]:
    """Performs backward prop from an intermediate node up to target nodes.

    Args:
      transform: Backward Transformation to implement reverse bound propagation.
      forward_env: Environment providing bounds over the nodes of the network,
        as obtained by a forward propagation.
      backward_bounds: Dict mapping the initial bounds that we want to propagate
        backward.
      target_indices: Indices of the nodes in the graph that we want to reach.
        It is the responsibility of the caller to ensure that valid bounds can
        be derived with only these bounds.
    Returns:
      targets: Propagated values corresponding to the required target_indices.
    """
    backward_env = collections.defaultdict(list)
    for node_ref, node_out in backward_bounds.items():
      backward_env[node_ref].append(node_out)

    return self._propagate_backward(transform, forward_env, backward_env,
                                    target_indices)

  def _backward_prop_eqn(
      self,
      transform: BackwardGraphTransform[BackRepr],
      forward_env: Dict[jax.core.Var, Union[TransformedNode, Tensor, _Ref]],
      backward_env: Dict[jax.core.Var, List[Union[BackRepr, _Ref]]],
      index: IndexCounter, eqn: jax.core.JaxprEqn):
    """Recursive step of `backward_propagation`."""

    def subgraph_handler(
        sub_transform, sub_graph, outval: BackRepr,
    ) -> Sequence[Sequence[Union[BackRepr, _Ref]]]:
      assert len(eqn.outvars) == 1
      use_ref = not isinstance(eqn.primitive,
                               synthetic_primitives.FakePrimitive)
      if outval is not None:
        # If we are actually backpropagating something, update the backward
        # environment correctly.
        outvals = [outval]
        if sub_transform is transform:
          # We must use the same environment, so that transformed equations
          # in the sub-graph are included in it.
          if use_ref:
            outvals = [[_Ref(outvar)] for outvar in eqn.outvars]
          else:
            outvals = [read_env(backward_env, outvar) for outvar in eqn.outvars]
        else:
          # We must leave the environment intact, because it contains
          # equations transformed by a different transformer.
          # Set up a new environment.
          assert not use_ref
          raise NotImplementedError(
              'Upstream backward transform attempting to process a sub-graph '
              'that was not processed by the principal backward transform.')

        # Add the sub-graph's outputs to the environment.
        backward_env.update({
            sub_outvar: outval
            for sub_outvar, outval in zip(sub_graph.outvars, outvals)})
      # Recursively back-propagate through the sub-graph.
      index.begin_child(len(sub_graph.eqns))
      try:
        for sub_eqn in reversed(sub_graph.eqns):
          self._backward_prop_eqn(transform, forward_env, backward_env,
                                  index, sub_eqn)
        if index.child_index() != 0:
          raise ValueError('Forward/backward indexing mismatch')
      finally:
        index.end_child()
      # Associate the sub-graph's inputs with the enclosing equation inputs.
      # However, if it's a synthetic primitive, then its input variables _are_
      # the sub-graph's input variables (no proxying with `_Ref`);
      # in this case just return empty lists because the input vars have already
      # received their back-propagated values during sub-graph traversal.
      eqn_invals = [
          [_Ref(sub_invar)] if use_ref else []
          for sub_invar in sub_graph.invars]
      return eqn_invals

    # Decrement the index to match the indexing in the forward propagation.
    index.decr()

    if len(eqn.outvars) != 1:
      # TODO Handle multi out primitives.
      raise NotImplementedError('Multiple outputs primitives not supported.')

    if isinstance(read_env(forward_env, eqn.outvars[0]), TransformedNode):
      if eqn.outvars[0] != self._index_to_node[index.as_tuple()]:
        raise ValueError('Forward/backward indexing mismatch')
      # Check if a repr is being propagated backward through this primitive.
      repr_on_outvar = eqn.outvars[0] in backward_env
      eqn_outval = None
      if repr_on_outvar:
        # Get the BackReprs on the outvar so that we can ensure that it's not
        # just an empty list.
        outvar_reprs = read_env(backward_env, eqn.outvars[0])
        if outvar_reprs:
          eqn_outval = transform.aggregate(outvar_reprs)
      if (eqn_outval is not None
          or transform.should_handle_as_subgraph(eqn.primitive)):
        # The two cases where we want to recurse into the primitives are:
        #   - If we have something to propagate backward on the output of the
        #     primitive (so eqn_outval is not None)
        #   - If this primitive should be handled as a subgraph, because the
        #     backward_env might contain something to backpropagate on the
        #     intermediate nodes.
        eqn_invars_vals = jax.tree_map(
            functools.partial(read_env, forward_env), eqn.invars)
        eqn_invals = transform.equation_backtransform(
            TransformContext(index.as_tuple(), subgraph_handler),
            eqn.primitive, eqn_outval, *eqn_invars_vals, **eqn.params)
      else:
        # If we do not propagate backward through this primitive, we do not want
        # to fill the backward_env with dummy None variables. This way the
        # parents of this bound will also not be executed unless a backward
        # bound was defined on them.
        eqn_invals = []
    else:
      # No dependence on the network's variable inputs.
      eqn_invals = [[None] for _ in eqn.invars]

    for in_var, in_val in zip(eqn.invars, eqn_invals):
      # If it's a literal, there are no updates to perform.
      if not isinstance(in_var, jax.core.Literal):
        backward_env[in_var].extend(in_val)
