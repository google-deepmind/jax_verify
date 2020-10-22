# coding=utf-8
# Copyright 2020 The jax_verify Authors.
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
from typing import Any, Callable, Dict, Generic, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp

from jax_verify.src import synthetic_primitives


Tensor = jnp.ndarray
T = TypeVar('T')
Nest = Union[T, Sequence[T], Dict[Any, T]]


class Bound(metaclass=abc.ABCMeta):
  """Abstract propagated bound."""

  @abc.abstractproperty
  def lower(self) -> Tensor:
    """Concrete lower bound."""

  @abc.abstractproperty
  def upper(self) -> Tensor:
    """Concrete upper bound."""

  def unwrap(self) -> 'Bound':
    """Underlying bound of method-specific type, without extra constraints.

    Usually this returns `self`. However, subclasses that wrap the bound to
    provide additional information (for example externally imposed interval
    constraints) will return the wrapped bound.

    Returns:
      Underlying bound arising directly from bound propagation.
    """
    return self


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


Repr = TypeVar('Repr')


class GraphTransform(Generic[Repr], metaclass=abc.ABCMeta):
  """Abstract graph propagation method."""

  @abc.abstractmethod
  def input_transform(
      self,
      index: int,
      lower_bound: Tensor,
      upper_bound: Tensor,
      ) -> Repr:
    """Constructs input representations from lower/upper bound tensors.

    Args:
      index: Integer identifying the input node.
      lower_bound: Original concrete lower bound on the input.
      upper_bound: Original concrete upper bound on the input.

    Returns:
      Method-specific representation for the inputs.
    """

  @abc.abstractmethod
  def primitive_transform(
      self,
      index: int,
      primitive: jax.core.Primitive,
      *args: Union[Repr, Tensor],
      **params,
      ) -> Repr:
    """Applies the given primitive operation to its arguments' representations.

    Args:
      index: Integer identifying the computation node.
      primitive: Primitive Jax operation to transform.
      *args: Arguments of the primitive. Arguments are expressed as
        method-specific representations if they have any dependence on the
        network's original inputs, or tensors otherwise.
      **params: Keyword arguments of the primitive.

    Returns:
      Method-specific representation for the operation's output.
    """


BoundTransform = GraphTransform[Bound]


class OpwiseBoundTransform(BoundTransform):
  """Bound propagation method defined by functions for each primitive op."""

  def __init__(
      self,
      input_transform: Callable[[int, Tensor, Tensor], Bound],
      primitive_transform: Dict[jax.core.Primitive, Callable[..., Bound]]):
    self._input_transform = input_transform
    self._primitive_transform = primitive_transform

  def input_transform(self, index, lower_bound, upper_bound):
    return self._input_transform(index, lower_bound, upper_bound)

  def primitive_transform(self, index, primitive, *args, **params):
    if primitive not in self._primitive_transform:
      raise NotImplementedError(f'Unknown Primitive: {primitive}')
    return self._primitive_transform[primitive](index, *args, **params)


JaxVar = Union[jax.core.Var, jax.core.Literal]


class _Ref:

  def __init__(self, target_var: JaxVar):
    self.target_var: JaxVar = target_var


def _sub_graph(eqn):
  """Exposes the subgraph in a standard format if this is a jaxpr call.

  Args:
    eqn: A JaxPr equation that may contain a subgraph
  Returns:
    sub_graph: Can be None if there is no subgraph, or a JaxPR graph
  """
  if eqn.primitive == jax.custom_derivatives.custom_jvp_call_jaxpr_p:
    return eqn.params['fun_jaxpr'].jaxpr
  elif eqn.primitive == jax.xla.xla_call_p:
    return eqn.params['call_jaxpr']
  else:
    return None


class PropagationGraph(Generic[Repr]):
  """Holds a computational graph and the environment holding the variables.
  """

  def __init__(self, graph: jax.core.Jaxpr, graph_simplifier):
    self._graph = graph
    self.env: Dict[jax.core.Var, Union[Repr, Tensor, _Ref]] = {}
    self._reverse_env: Dict[jax.core.Var, Sequence[Repr]] = {}
    self.reverse_nodes: Dict[int, Repr] = {}
    self._first_index = None
    self._last_index = None
    self._graph_simplifier = graph_simplifier

  def read(self, var: JaxVar) -> Union[Repr, Tensor]:
    """Read the value from the environment."""
    if isinstance(var, jax.core.Literal):
      # Literals are values baked into the Jaxpr, e.g. ReLU's '0' arg to 'max'.
      return var.val
    else:
      val = self.env[var]
      if isinstance(val, _Ref):
        # The value is a reference to another variable. Follow the reference.
        return self.read(val.target_var)
      else:
        return val

  def build_inputs(
      self,
      transform: GraphTransform[Repr],
      bounds: Nest[Union[Bound, Tensor]],
      literals: Sequence[Tensor]):
    """Constructs initial representation of the graph's inputs.

    This is accomplished by inspecting the parameter inputs of the JaxPR
    representation of the computation, and constructing bounds using
    `input_transform`.

    Args:
      transform: Constructs initial input representation from
        lower/upper bound tensors.
      bounds: Nest of `IntervalBound` objects containing the lower and upper
        bounds on all the inputs, or `Tensor`s containing known inputs directly.
      literals: Values to use for any constant inputs of the JaxPR.
    """
    flat_bounds, _ = jax.tree_util.tree_flatten(bounds)
    invals = []
    index = 0
    for bound in flat_bounds:
      if isinstance(bound, Bound):
        input_val = transform.input_transform(index, bound.lower, bound.upper)
        index += 1
      else:
        # Input is a fixed tensor.
        input_val = bound
      invals.append(input_val)
    self.env.update(zip(self._graph.invars, invals))
    self.env.update(zip(self._graph.constvars, literals))

    self._first_index = index

  def forward_propagation(self, transform: GraphTransform[Repr]) -> Repr:
    """Performs forward propagation on the parsed computation graph.

    This is accomplished by traversing the JaxPR representation of the
    computation and translating the computational graph, using the
    `primitive_transform` replacement for each jax primitive.

    Args:
      transform: Basic Jax primitive ops' equivalents for operating on
        the representation (e.g. bound propagation).
    Returns:
      outvals: Propagated values corresponding to the graph output.
    """
    # Indices 0..k-1 are reserved for the k inputs to the network.
    index = self._first_index
    for eqn in self._graph_simplifier(self._graph).eqns:
      index = self._forward_prop_eqn(transform, index, eqn)
    self._last_index = index

    outvals = jax.tree_map(self.read, self._graph.outvars)
    return outvals

  def _forward_prop_eqn(self, transform, index, eqn):
    """Recursive step of `forward_propagation`."""
    sub_graph = _sub_graph(eqn)
    if sub_graph is not None:
      sub_graph = self._graph_simplifier(sub_graph)
      # Add the sub-graph's inputs to the environment.
      self.env.update({
          sub_invar: _Ref(invar)
          for sub_invar, invar in zip(sub_graph.invars, eqn.invars)})
      # Recursively propagate through the sub-graph.
      for sub_eqn in sub_graph.eqns:
        index = self._forward_prop_eqn(transform, index, sub_eqn)
      # Associate the sub-graph's outputs with the enclosing equation outputs.
      self.env.update({
          outvar: _Ref(sub_outvar)
          for sub_outvar, outvar in zip(sub_graph.outvars, eqn.outvars)})
      return index
    else:
      if len(eqn.outvars) != 1:
        # TODO Handle multi out primitives.
        raise NotImplementedError('Multiple outputs primitives not supported.')
      eqn_outvar = eqn.outvars[0]

      eqn_invars_vals = jax.tree_map(self.read, eqn.invars)
      eqn_params = eqn.params
      if any(isinstance(inval, Bound) for inval in eqn_invars_vals):
        eqn_outval = transform.primitive_transform(
            index, eqn.primitive, *eqn_invars_vals, **eqn_params)
        index += 1
      else:
        # No dependence on the network's inputs. Any primitive is supported.
        eqn_outval = eqn.primitive.bind(*eqn_invars_vals, **eqn_params)

      self.env[eqn_outvar] = eqn_outval
      return index

  def backward_propagation(self, primitive_transform, aggregate_function,
                           initial_outputs):
    """Performs backward propagation on the parsed computational graph.

    Args:
      primitive_transform: Dict matching basic jax primitives to functions
        (index, *input_bounds, **params) -> output_bounds, implementing
        the primitive ops' reverse bound propagation equivalents.
      aggregate_function: Function to aggregate the contribution of several
        of the network branches to one node. For example, for a ResNet, there
        will be several branches coming from the same node so when performing
        backward propagation, several contribution will come to the same node.
      initial_outputs: Values on the output to propagate backward.
    Returns:
      invals: Propagated values corresponding to the graph input.
    """

    flat_outs, _ = jax.tree_util.tree_flatten(initial_outputs)
    self._reverse_env = collections.defaultdict(list)
    for out_var, out_val in zip(self._graph.outvars, flat_outs):
      self._reverse_env[out_var].append(out_val)

    index = self._last_index
    for eqn in reversed(self._graph_simplifier(self._graph).eqns):
      index = self._backward_prop_eqn(primitive_transform, aggregate_function,
                                      index, eqn)

    if index != self._first_index:
      raise ValueError('Forward/backward indexing mismatch')

    invals = []
    for invar in reversed(self._graph.invars):
      if isinstance(self.read(invar), Bound):
        inval = aggregate_function(self._reverse_env.get(invar))
        index -= 1
        self.reverse_nodes[index] = inval
        invals.insert(0, inval)
      else:
        # Not a variable input.
        invals.insert(0, None)
    return invals

  def _backward_prop_eqn(self, primitive_transform, aggregate_function,
                         index, eqn):
    """Recursive step of `backward_propagation`."""
    sub_graph = _sub_graph(eqn)
    if sub_graph is not None:
      sub_graph = self._graph_simplifier(sub_graph)
      # Add the sub-graph's outputs to the environment.
      for sub_outvar, outvar in zip(sub_graph.outvars, eqn.outvars):
        self._reverse_env[sub_outvar].extend(self._reverse_env[outvar])
      # Recursively back-propagate through the sub-graph.
      for sub_eqn in reversed(sub_graph.eqns):
        index = self._backward_prop_eqn(primitive_transform, aggregate_function,
                                        index, sub_eqn)
      # Associate the sub-graph's inputs with the enclosing equation inputs.
      for sub_invar, invar in zip(sub_graph.invars, eqn.invars):
        self._reverse_env[invar].extend(self._reverse_env[sub_invar])
      return index

    else:
      if len(eqn.outvars) != 1:
        # TODO Handle multi out primitives.
        raise NotImplementedError('Multiple outputs primitives not supported.')

      if isinstance(self.read(eqn.outvars[0]), Bound):
        eqn_outval = aggregate_function(self._reverse_env[eqn.outvars[0]])
        eqn_invars_vals = jax.tree_map(self.read, eqn.invars)
        if eqn.primitive not in primitive_transform:
          raise NotImplementedError(f'Unknown Primitive: {eqn.primitive}')
        backward_prop_op = primitive_transform[eqn.primitive]
        index -= 1
        self.reverse_nodes[index] = eqn_outval
        eqn_invals = backward_prop_op(index, eqn_outval, *eqn_invars_vals,
                                      **eqn.params)
      else:
        # No dependence on the network's variable inputs.
        eqn_invals = [None for _ in eqn.invars]

      for in_var, in_val in zip(eqn.invars, eqn_invals):
        if not isinstance(in_var, jax.core.Literal):
          self._reverse_env[in_var].append(in_val)
      return index


def bound_propagation(
    transform: GraphTransform[Repr],
    function: Callable[..., Tensor],
    *bounds: Nest[Union[Bound, Tensor]],
    graph_simplifier=synthetic_primitives.activation_detector,
) -> Tuple[Repr, PropagationGraph[Repr]]:
  """Performs Bound Propagation on the model implemented by `function`.

  Args:
    transform: Basic Jax primitive ops' equivalents for operating on
      the representation.
    function: Pure function inputs -> outputs. If the function to propagate
      through has a more complex signature, the use of `functools.partial` can
      solve that problem.
    *bounds: Nest of `IntervalBound` objects containing the lower and upper
      bounds on all the inputs, or `Tensor`s containing known inputs directly.
    graph_simplifier: Function transforming the JaxPR graph into a simpler
      graph. Default value is a function identifying specific activation
      functions.
  Returns:
    bounds: Bounds over all the outputs of the function, with the same structure
      as the output of `function`
    graph: A PropagationGraph object which contains the parsed representation of
      the computation as well as the environment.
  """
  # Parse the computation graph.
  placeholder_inputs = jax.tree_util.tree_map(
      lambda b: b.lower if isinstance(b, Bound) else b,
      bounds)
  jaxpr_maker = jax.make_jaxpr(function)
  parsed = jaxpr_maker(*placeholder_inputs)
  output_shapes = jax.eval_shape(function, *placeholder_inputs)

  graph = PropagationGraph(parsed.jaxpr, graph_simplifier)
  graph.build_inputs(transform, bounds, parsed.literals)
  outvals = graph.forward_propagation(transform)

  # Make outvals into the same tree structure than the output of the function.
  tree_structure = jax.tree_util.tree_structure(output_shapes)
  outvals = jax.tree_util.tree_unflatten(tree_structure, outvals)

  return outvals, graph
