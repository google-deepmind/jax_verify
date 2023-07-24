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

"""Provides methods to simplify the JaxPR computation graph.
"""
import collections
import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, TypeVar, Union
import jax
from jax import lax
from jax.experimental import pjit
from jax.interpreters import mlir
import jax.numpy as jnp
from jax_verify.src.types import Primitive, Tensor, TensorFun  # pylint: disable=g-multiple-import
import numpy as np

VarIsBoundDict = MutableMapping[jax.core.Var, bool]

T = TypeVar('T')
ListNest = Union[T, Sequence['ListNest[T]']]
GraphSimplifier = ListNest[
    Callable[[jax.core.Jaxpr, VarIsBoundDict], jax.core.Jaxpr]]
SimpleSimplifier = Callable[[jax.core.Jaxpr], jax.core.Jaxpr]


SUBGRAPH_PRIMITIVES: Sequence[Primitive] = (
    jax.custom_derivatives.custom_jvp_call_jaxpr_p,
    jax.custom_derivatives.custom_jvp_call_p,
    jax.interpreters.pxla.xla_pmap_p,
    pjit.pjit_p,
)


def jax_primitive_subgraph(primitive: Primitive, **params) -> jax.core.Jaxpr:
  """Returns the sub-graph for the given equation."""
  if primitive == pjit.pjit_p:
    return params['jaxpr'].jaxpr
  elif primitive == jax.custom_derivatives.custom_jvp_call_jaxpr_p:
    return params['fun_jaxpr'].jaxpr
  elif primitive in SUBGRAPH_PRIMITIVES:
    if isinstance(params['call_jaxpr'], jax.core.ClosedJaxpr):
      return params['call_jaxpr'].jaxpr
    else:
      assert isinstance(params['call_jaxpr'], jax.core.Jaxpr)
      return params['call_jaxpr']
  else:
    return params['jax_verify_subgraph']


def _replace_jax_primitive_subgraph(
    primitive: Primitive,
    params: MutableMapping[str, Any],
    subgraph: jax.core.Jaxpr,
):
  """Updates the sub-graph for the given equation."""
  if primitive == jax.custom_derivatives.custom_jvp_call_jaxpr_p:
    params['fun_jaxpr'] = params['fun_jaxpr'].replace(jaxpr=subgraph)
  elif primitive in SUBGRAPH_PRIMITIVES:
    if primitive == pjit.pjit_p:
      params['jaxpr'] = params['jaxpr'].replace(jaxpr=subgraph)
    elif isinstance(params['call_jaxpr'], jax.core.ClosedJaxpr):
      params['call_jaxpr'] = params['call_jaxpr'].replace(jaxpr=subgraph)
    else:
      assert isinstance(params['call_jaxpr'], jax.core.Jaxpr)
      params['call_jaxpr'] = subgraph
  else:
    params['jax_verify_subgraph'] = subgraph


def filter_jaxverify_kwargs(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
  if 'jax_verify_keepjvargs' in kwargs and kwargs['jax_verify_keepjvargs']:
    return kwargs
  else:
    return {k: v for k, v in kwargs.items() if not k.startswith('jax_verify')}


def make_jaxpr_nojit(fun, *inps, **kwargs):
  kwarged_fun = functools.partial(fun, **kwargs)
  with jax.disable_jit():
    make_jaxpr = jax.make_jaxpr(kwarged_fun)
    return make_jaxpr(*inps)


def simplify_graph(
    graph_simplifier: GraphSimplifier,
    graph: jax.core.Jaxpr,
    var_is_bound: VarIsBoundDict,
) -> jax.core.Jaxpr:
  """Recursively apply the graph simplifier to the graph and its subgraphs.

  Starts by filling in var_is_bound with the results for all variables.

  Args:
    graph_simplifier: Function simplifying the eqns in a jaxpr but ignoring
      the subgraphs
    graph: Jaxpr to simplify
    var_is_bound: Dict mapping whether a given variable is a bound or not.
  Returns:
    Simplified Jaxpr, where all the subgraphs are also simplified.
  """
  _propagate_var_is_bound(graph, var_is_bound)
  return _simplify_graph(var_is_bound, graph, graph_simplifier)


def _simplify_graph(
    var_is_bound: VarIsBoundDict,
    graph: jax.core.Jaxpr,
    graph_simplifier: GraphSimplifier,
) -> jax.core.Jaxpr:
  """Recursively apply the graph simplifier to the graph and its subgraphs.

  Args:
    var_is_bound: Dict mapping whether a given variable is a bound or not.
    graph: Jaxpr to simplify
    graph_simplifier: Function simplifying the eqns in a jaxpr but ignoring
      the subgraphs
  Returns:
    Simplified Jaxpr, where all the subgraphs are also simplified.
  """
  if isinstance(graph_simplifier, list):
    # Recursively apply each child simplifier to the entire graph.
    return functools.reduce(
        functools.partial(_simplify_graph, var_is_bound),
        graph_simplifier,
        graph)
  else:
    # Apply this 'leaf' simplifier to the graph.
    simplified_graph = graph_simplifier(graph, var_is_bound)
    # Also recursively apply it to all sub-graphs.
    for eqn in simplified_graph.eqns:
      if eqn.primitive in SUBGRAPH_PRIMITIVES:
        subgraph = jax_primitive_subgraph(eqn.primitive, **eqn.params)
        _replace_jax_primitive_subgraph(
            eqn.primitive,
            eqn.params,
            _simplify_graph(var_is_bound, subgraph, graph_simplifier),
        )
    return simplified_graph


capture_float32 = object()


class SyntheticPrimitiveSpec:
  """Traced graph of a function used to specify a synthetic primitive.."""

  def __init__(
      self,
      fn: TensorFun,
      upstream_simplifier: Optional[SimpleSimplifier],
      primitive: 'FakePrimitive',
      *arg_shapes_dtypes: Union[tuple[Sequence[int], jnp.dtype],
                                Callable[[], Tensor]],
      **params):
    """Constructs a specification from the given function.

    Traces `fn` with placeholder inputs specified by `arg_shapes_dtypes`, and
    records the resulting Jaxpr. This will subsequently be used a reference
    graph for detecting occurrences of `fn` within other graphs, allowing them
    to be replaced with `primitive`.

    The `upstream_simplifier` is important if this synthetic primitive is to be
    detected _after_ the upstream graph simplifier has already been applied to
    the main graph. In that case, occurrences of the synthetic primitive will
    already have received upstream simplifications. We must apply the same
    simplifications to this reference graph so that they match.

    Args:
      fn: Traceable function defining the computation to detect.
      upstream_simplifier: Optional simplifier to apply to the reference graph.
      primitive: Synthetic primitive to denote occurrences of `fn`.
      *arg_shapes_dtypes: Shapes and dtypes of placeholder inputs to `fn`, or a
        function producing the placeholder.
      **params: Keyword parameters qualifying `primitive`. Any parameter may
        be `capture_float32`, in which case the parameter value will be
        determined during the match by capturing the actual value passed as
        keyword param to `fn`.
    """
    # Replace `capture_float32` params with thunks that look up the actual
    # captured values passed as keyword params to `fn`.
    subscript = lambda key, captures: captures[key]
    params = {key: functools.partial(subscript, key)
                   if param is capture_float32 else param
              for key, param in params.items()}
    capture_keys = [key for key, param in params.items() if callable(param)]

    placeholder_inputs = []
    for asd in arg_shapes_dtypes:
      if isinstance(asd, tuple):
        shape, dtype = asd
        placeholder_inputs.append(jnp.zeros(shape, dtype))
      else:
        placeholder_inputs.append(asd())

    placeholder_params = {key: 7. for key in capture_keys}

    closed_jaxpr = make_jaxpr_nojit(fn, *placeholder_inputs,
                                    **placeholder_params)
    self._graph = closed_jaxpr.jaxpr

    self._capture_literals = {}
    for capture_key in capture_keys:
      # Re-trace using a different placeholder value for this param only.
      # The actual values are immaterial; we simply need them to be different
      # so that we can detect which literals correspond to this keyword param.
      alt_params = {key: 8. if key == capture_key else 7.
                    for key in capture_keys}

      closed_jaxpr = make_jaxpr_nojit(fn, *placeholder_inputs, **alt_params)
      alt_graph = closed_jaxpr.jaxpr
      for literal in _differing_literal_indices(self._graph, alt_graph):
        assert literal not in self._capture_literals
        self._capture_literals[literal] = capture_key

    if upstream_simplifier:
      self._graph = upstream_simplifier(self._graph)
    self._primitive = primitive
    self._params = params

  @property
  def graph(self) -> jax.core.Jaxpr:
    return self._graph

  @property
  def capture_literals(self) -> Mapping[tuple[int, int], str]:
    """Literals whose values are to be captured.

    Returns:
      Python ids of literals whose values are to be captured, mapping to the
      name of the spec function's keyword param that supplies the value.
    """
    return self._capture_literals

  @property
  def primitive(self) -> 'FakePrimitive':
    return self._primitive

  @property
  def params(self) -> Mapping[str, Any]:
    return self._params

  def simplify(self, graph_simplifier: SimpleSimplifier):
    """Applies an upstream graph simplifier to this reference graph.

    This is important if this synthetic primitive is to be detected _after_
    the upstream graph simplifier has already been applied to the main graph.

    In that case, occurrences of the synthetic primitive will already have
    received upstream simplifications. We must apply the same simplifications
    to this reference graph so that they match.

    Args:
      graph_simplifier: Upstream graph simplifier to apply.
    """
    self._graph = graph_simplifier(self._graph)


def _mark_outputs_whether_bounds(eqn, var_is_bound):
  non_literal_inps = [invar for invar in eqn.invars
                      if not isinstance(invar, jax.core.Literal)]
  # If any of the input is a bound, the outputs are bounds too.
  outputs_are_bounds = any(var_is_bound[invar] for invar in non_literal_inps)
  for outvar in eqn.outvars:
    var_is_bound[outvar] = outputs_are_bounds


def _propagate_var_is_bound(graph: jax.core.Jaxpr,
                            var_is_bound: VarIsBoundDict):
  """Fill in the var_is_bound dictionary to indicate which variables are bounds.

  Args:
    graph: The graph to check the variables of.
    var_is_bound: Dictionary to fill in.
  """
  for cvar in graph.constvars:
    var_is_bound[cvar] = False
  for eqn in graph.eqns:
    _mark_outputs_whether_bounds(eqn, var_is_bound)
    if eqn.primitive in SUBGRAPH_PRIMITIVES:
      eqn_subgraph = jax_primitive_subgraph(eqn.primitive, **eqn.params)
      subgraph_var_is_bound = {}
      for subgraph_invar, eqn_invar in zip(eqn_subgraph.invars, eqn.invars):
        if isinstance(eqn_invar, jax.core.Var):
          subgraph_var_is_bound[subgraph_invar] = var_is_bound[eqn_invar]
        else:
          subgraph_var_is_bound[subgraph_invar] = False
      _propagate_var_is_bound(eqn_subgraph, subgraph_var_is_bound)
      var_is_bound.update(subgraph_var_is_bound)


def detect(
    synthetic_primitive_specs: Sequence[SyntheticPrimitiveSpec],
    graph: jax.core.Jaxpr,
) -> jax.core.Jaxpr:
  """Attempts to simplify the graph by identifying specific activations.

  This is done by recognizing part of the graph and fusing them into synthetic
  primitives.

  Args:
    synthetic_primitive_specs: Specifies graph features to be detected
      as synthetic primitives.
    graph: Unprocessed JaxPR graph.
  Returns:
    Potentially modified JaxPR graph.
  """
  new_eqns = []
  eqn_idx = 0
  while eqn_idx < len(graph.eqns):
    eqn, eqn_idx = _next_equation(synthetic_primitive_specs, graph, eqn_idx)
    new_eqns.append(eqn)

  return jax.core.Jaxpr(graph.constvars, graph.invars, graph.outvars, new_eqns)


def _next_equation(
    synthetic_primitive_specs: Sequence[SyntheticPrimitiveSpec],
    graph: jax.core.Jaxpr,
    eqn_idx: int,
) -> tuple[jax.core.JaxprEqn, int]:
  """Determines the next equation in the Jaxpr, possibly a substitution.

  Args:
    synthetic_primitive_specs: Specification of what graph patterns can be
      replaced with fake primitives.
    graph: Jaxpr graph.
    eqn_idx: Index of the equation in the Jaxpr graph.
  Returns:
    eqn: Next equation in the Jaxpr, which may be a fake primitive.
    eqn_idx: Index of the following equation in the Jaxpr.
  """
  for spec in synthetic_primitive_specs:
    (
        spec_matches, match_len,
        primitive_invars, primitive_outvars, captures
    ) = _matches(spec.graph, spec.capture_literals, graph, eqn_idx)
    if spec_matches:
      sub_jaxpr = jax.core.Jaxpr(
          constvars=[], invars=primitive_invars, outvars=primitive_outvars,
          eqns=graph.eqns[eqn_idx:(eqn_idx + match_len)])
      # Replace deferred keyword params with captured literals.
      spec_params = {
          key: param(captures) if callable(param) else param
          for key, param in spec.params.items()}
      return jax.core.new_jaxpr_eqn(
          primitive_invars, primitive_outvars, spec.primitive,
          {'jax_verify_subgraph': sub_jaxpr, **spec_params},
          jax.core.no_effects,
          graph.eqns[eqn_idx].source_info), eqn_idx + match_len

  return graph.eqns[eqn_idx], eqn_idx + 1


def _equal_literal_values(lhs, rhs) -> bool:
  # Check that the literals have the same value, using the appropriate
  # comparison method.
  if isinstance(lhs, jnp.ndarray):
    return np.all(lhs.item() == rhs.item())
  else:
    # literal.val might be an int / float
    return lhs == rhs


def _matches(
    spec: jax.core.Jaxpr,
    capture_literals: Mapping[tuple[int, int], str],
    graph: jax.core.Jaxpr,
    eqn_idx: int,
) -> tuple[
    bool,
    int,
    Sequence[jax.core.Var],
    Sequence[Union[jax.core.Var, jax.core.Literal]],
    Mapping[str, Any],
]:
  """Determines whether the graph continues with the given reference graph.

  Args:
    spec: Reference Jaxpr graph specifying the fake primitive to match.
    capture_literals: Python ids of literals whose value is to be captured,
      mapping to capture param name.
    graph: Jaxpr graph.
    eqn_idx: Index of the current equation in the Jaxpr graph,
      at which to check for a match.

  Returns:
    spec_matches: Whether the graph matches the spec.
    match_len: How many (top-level) equations constitute the match.
    invars: Varables of `graph` that correspond to `spec.invars`.
    outvars: Varables of `graph` that correspond to `spec.outvars`.
    captures: Captured literal values, keyed by param name.
  """
  no_match = False, 0, [], [], {}
  eqn_idx_orig = eqn_idx
  graph_vars_by_spec_var = {}
  captures = {}

  def inputs_correspond(spec_eqn_invars, graph_eqn_invars, eqn_pos) -> bool:
    # Check that the equation's inputs correspond.
    if len(spec_eqn_invars) != len(graph_eqn_invars):
      return False
    for invar_idx, (spec_eqn_invar, graph_eqn_invar) in enumerate(zip(
        spec_eqn_invars, graph_eqn_invars)):
      if isinstance(spec_eqn_invar, jax.core.Literal):
        if not isinstance(graph_eqn_invar, jax.core.Literal):
          return False
        # Check that the literals hold values of the same type.
        if not isinstance(spec_eqn_invar.val, type(graph_eqn_invar.val)):
          return False
        if (eqn_pos, invar_idx) in capture_literals:
          # The value of this literal in the specification graph is just a
          # placeholder. Instead of an equality comparison, we capture the
          # corresponding value in the actual graph.
          key = capture_literals[(eqn_pos, invar_idx)]
          if key in captures and not _equal_literal_values(
              graph_eqn_invar.val, captures[key]):
            # Same keyword param has already captured a different value.
            return False
          captures[key] = graph_eqn_invar.val
        elif not _equal_literal_values(spec_eqn_invar.val, graph_eqn_invar.val):
          return False
      else:
        if ((spec_eqn_invar in spec.invars or spec_eqn_invar in spec.constvars)
            and spec_eqn_invar not in graph_vars_by_spec_var):
          # Encountering an input for the first time.
          graph_vars_by_spec_var[spec_eqn_invar] = graph_eqn_invar
        if graph_vars_by_spec_var[spec_eqn_invar] != graph_eqn_invar:
          return False
    return True

  for spec_eqn in spec.eqns:
    if eqn_idx >= len(graph.eqns):
      return no_match
    graph_eqn = graph.eqns[eqn_idx]
    eqn_pos = eqn_idx - eqn_idx_orig
    eqn_idx += 1

    # Check that the primitives are the same.
    if graph_eqn.primitive != spec_eqn.primitive:
      return no_match

    if spec_eqn.primitive in (jax.lax.add_p, jax.lax.mul_p):
      backup_graph_vars_by_spec_var = {**graph_vars_by_spec_var}
      backup_captures = {**captures}
      if not inputs_correspond(spec_eqn.invars, graph_eqn.invars, eqn_pos):
        # Jax will sometimes silently reverse the order of args of these
        # commutative ops, for example if one arg is a literal.
        graph_vars_by_spec_var = backup_graph_vars_by_spec_var
        captures = backup_captures
        if not inputs_correspond(spec_eqn.invars,
                                 list(reversed(graph_eqn.invars)), eqn_pos):
          return no_match
    else:
      if not inputs_correspond(spec_eqn.invars, graph_eqn.invars, eqn_pos):
        return no_match

    # Check that the equation's params are equal.
    if set(spec_eqn.params) != set(graph_eqn.params):
      return no_match
    for key in set(spec_eqn.params):
      spec_param = spec_eqn.params[key]
      graph_param = graph_eqn.params[key]
      if key in ('fun_jaxpr', 'call_jaxpr', 'jax_verify_subgraph', 'jaxpr'):
        # Recursively check that the sub-graphs match.
        if isinstance(spec_param, jax.core.ClosedJaxpr):
          assert isinstance(graph_param, jax.core.ClosedJaxpr)
          subspec = spec_param.jaxpr
          subgraph = graph_param.jaxpr
        else:
          assert isinstance(spec_param, jax.core.Jaxpr)
          assert isinstance(graph_param, jax.core.Jaxpr)
          subspec = spec_param
          subgraph = graph_param
        (
            subgraph_matches, _,
            subgraph_invars, subgraph_outvars, subgraph_captures
        ) = _matches(subspec, capture_literals, subgraph, 0)
        captures.update(subgraph_captures)
        if not subgraph_matches:
          return no_match
        if subgraph.invars != subgraph_invars:
          return no_match
        if subgraph.outvars != subgraph_outvars:
          return no_match
        # Assimilate the captured literal values from the sub-graph.
        if any(key in captures and
               not _equal_literal_values(capture, captures[key])
               for key, capture in subgraph_captures.items()):
          # Same keyword param has already captured a different value.
          return no_match
      elif key in ('shape', 'new_sizes',
                   'start_indices', 'limit_indices', 'slice_sizes'):
        # Don't check shape, but do check rank.
        if len(spec_param) != len(graph_param):
          return no_match
      elif not callable(spec_param):
        if spec_param != graph_param:
          return no_match

    # Record the correspondence between the equation's outputs.
    graph_vars_by_spec_var.update({
        spec_eqn_outvar: graph_eqn_outvar
        for spec_eqn_outvar, graph_eqn_outvar in zip(
            spec_eqn.outvars, graph_eqn.outvars)})

  # It's a match.
  # Look up the input and output variables in the graph.
  graph_invars = [
      graph_vars_by_spec_var[spec_invar] for spec_invar in spec.invars]
  assert all(graph_invar is not None for graph_invar in graph_invars)
  graph_outvars = [
      graph_vars_by_spec_var[spec_outvar]
      if isinstance(spec_outvar, jax.core.Var) else spec_outvar
      for spec_outvar in spec.outvars]
  assert all(graph_outvar is not None for graph_outvar in graph_outvars)
  return True, eqn_idx - eqn_idx_orig, graph_invars, graph_outvars, captures


def _differing_literal_indices(
    graph: jax.core.Jaxpr,
    alt_graph: jax.core.Jaxpr,
) -> Sequence[tuple[int, int]]:
  """Returns indices of literals taking different values in the two graphs."""
  literals = []
  assert len(graph.eqns) == len(alt_graph.eqns), 'Different number of equations'
  for eqn_idx, (eqn, alt_eqn) in enumerate(zip(graph.eqns, alt_graph.eqns)):
    assert eqn.primitive == alt_eqn.primitive, 'Different primitives'

    # Check that the equation's inputs correspond.
    for invar_idx, (eqn_invar, alt_eqn_invar) in enumerate(zip(eqn.invars,
                                                               alt_eqn.invars,
                                                               strict=True)):
      assert (
          isinstance(eqn_invar, jax.core.Literal) ==
          isinstance(alt_eqn_invar, jax.core.Literal)
      ), 'Different literal occurrences'
      if (isinstance(eqn_invar, jax.core.Literal) and
          not _equal_literal_values(eqn_invar.val, alt_eqn_invar.val)):
        literals.append((eqn_idx, invar_idx))

    assert set(eqn.params) == set(alt_eqn.params), 'Different param key sets'
    for key in set(eqn.params):
      param = eqn.params[key]
      alt_param = alt_eqn.params[key]
      if key in ('fun_jaxpr', 'call_jaxpr', 'jax_verify_subgraph', 'jaxpr'):
        # Recursively match the sub-graphs.
        subgraph = param.jaxpr if isinstance(
            param, jax.core.ClosedJaxpr) else param
        alt_subgraph = alt_param.jaxpr if isinstance(
            alt_param, jax.core.ClosedJaxpr) else alt_param
        literals.extend(_differing_literal_indices(subgraph, alt_subgraph))

  return literals


def _is_linear_eqn(eqn: jax.core.JaxprEqn, var_is_bound: VarIsBoundDict):
  """Identify if an eqn is a linear transformation of inputs that are bounds.

  Args:
    eqn: The equation to check.
    var_is_bound: Dictionary indicating whether each variable represent a bound
      or a Tensor.
  Returns:
    is_linear: boolean indicating whether the equation is linear.
  """
  # Handle the case where a subgraph primitive contains only linear operations.
  # We will simplify the graph and see if it amounts to a single primitive.
  if eqn.primitive in SUBGRAPH_PRIMITIVES:
    subgraph = jax_primitive_subgraph(eqn.primitive, **eqn.params)
    grouped_subgraph = group_linear_sequence(subgraph, var_is_bound)
    return (len(grouped_subgraph.eqns) == 1
            and grouped_subgraph.eqns[0].primitive is linear_p)

  # Otherwise, check simply the primitive
  prim = eqn.primitive
  non_literal_inps = [invar for invar in eqn.invars
                      if not isinstance(invar, jax.core.Literal)]
  nb_bound_input = sum(var_is_bound[invar] for invar in non_literal_inps)
  if not any(var_is_bound[outvar] for outvar in eqn.outvars):
    return False
  # Make sure that if this produces bounds,there is only one output
  assert len(eqn.outvars) == 1

  return (prim in LINEAR_OP
          or (prim in BILINEAR_OP and nb_bound_input == 1)
          or (prim is jax.lax.div_p
              and nb_bound_input == 1
              and not isinstance(eqn.invars[0], jax.core.Literal)
              and var_is_bound[eqn.invars[0]]))


def _is_posbilinear_eqn(eqn, var_is_bound):
  """Identify if an eqn is a Posbilinear transformation.

  Note that a PosBilinear primitive with only one of the inputs being a bound is
  considered a linear transformation, not a posbilinear one.

  Args:
    eqn: The equation to check.
    var_is_bound: Dictionary indicating whether each variable represent a bound
      or a Tensor.
  Returns:
    is_posbilinear: boolean indicating whether the equation is posbilinear.
  """
  # Handle the case where a subgraph primitive contains only a quadratic
  # operation. This is often how the results of einsum are generated.
  if eqn.primitive in SUBGRAPH_PRIMITIVES:
    subgraph = jax_primitive_subgraph(eqn.primitive, **eqn.params)
    grouped_subgraph = group_posbilinear(subgraph, var_is_bound)
    return (len(grouped_subgraph.eqns) == 1
            and grouped_subgraph.eqns[0].primitive is posbilinear_p)

  # Otherwise, simply check the primitive
  prim = eqn.primitive
  non_literal_inps = [invar for invar in eqn.invars
                      if not isinstance(invar, jax.core.Literal)]
  nb_bound_input = sum(var_is_bound[invar] for invar in non_literal_inps)
  return (prim in BILINEAR_OP) and (nb_bound_input == 2)


def _find_eqn(eqn_list: Sequence[jax.core.JaxprEqn], var: jax.core.Var) -> int:
  """Find the equation producing the var, and returns its index."""
  eqn_idx = 0
  for eqn_idx, eqn in enumerate(eqn_list):
    if eqn.outvars[0] == var:
      return eqn_idx
  else:
    assert False


def group_posbilinear(graph: jax.core.Jaxpr,
                      var_is_bound: VarIsBoundDict,
                      ) -> jax.core.Jaxpr:
  """Simplifier identifying the PosBilinear terms in the graph.

  A PosBilinear equation can be written in the form of:
    x^T M y
  where x and y are variable for which we have bound and M is a matrix with
  positive entries.

  Args:
    graph: Jaxpr to simplify
    var_is_bound: Dict mapping whether a given variable is a bound or not.
  Returns:
    Simplified Jaxpr, where all the PosBilinear have been identified.
  """
  new_eqns = []
  for eqn in graph.eqns:
    # Identify the posbilinear operations
    if _is_posbilinear_eqn(eqn, var_is_bound):
      non_literal_invars = [invar for invar in eqn.invars
                            if isinstance(invar, jax.core.Var)]
      posbilinear_jaxpr = jax.core.Jaxpr(
          constvars=[], invars=non_literal_invars,
          outvars=eqn.outvars, eqns=[eqn])
      new_eqns.append(jax.core.new_jaxpr_eqn(
          posbilinear_jaxpr.invars, posbilinear_jaxpr.outvars, posbilinear_p,
          {'jax_verify_subgraph': posbilinear_jaxpr,
           'jax_verify_keepjvargs': True}, jax.core.no_effects))
    else:
      new_eqns.append(eqn)

  return jax.core.Jaxpr(graph.constvars, graph.invars, graph.outvars, new_eqns)


def group_linear_sequence(graph: jax.core.Jaxpr,
                          var_is_bound: VarIsBoundDict,
                          ) -> jax.core.Jaxpr:
  """Attempt to fold linear sequences together into synthetic linear primitives.

  Args:
    graph: Unprocessed JaxPR graph.
    var_is_bound: Dict mapping whether a given variable is a bound or not.
  Returns:
    Potentially modified JaxPR graph.
  """
  consumed_by_linear = set()
  consumed_by = collections.Counter()
  is_linear_result = {}

  # Do a first pass through the graph to identify the Linear sequences
  for eqn in graph.eqns:
    is_linear = _is_linear_eqn(eqn, var_is_bound)
    outvar = eqn.outvars[0]
    is_linear_result[outvar] = is_linear
    for invar in eqn.invars:
      if not isinstance(invar, jax.core.Literal) and var_is_bound[invar]:
        consumed_by[invar] += 1
        if is_linear:
          consumed_by_linear.add(invar)

  for outvar in graph.outvars:
    if isinstance(outvar, jax.core.Var):
      consumed_by[outvar] += 1

  # Now collect the equations, merging the Linear sequences together
  new_eqns = []
  to_be_folded = {}
  for eqn in graph.eqns:
    outvar = eqn.outvars[0]

    if is_linear_result[outvar]:
      # This is a Linear operation. Let's construct it, possibly including
      # previous linear operations that were waiting to be folded in.
      lin_invars = []
      linear_eqns = []
      for invar in eqn.invars:
        if isinstance(invar, jax.core.Var):
          # Filter out the literals, which should not be registered as inputs to
          # a subgraph.
          if invar in to_be_folded:
            subg = to_be_folded[invar]
            for sub_invar in subg.invars:
              lin_invars.append(sub_invar)
            linear_eqns.extend(subg.eqns)
            del to_be_folded[invar]
          else:
            lin_invars.append(invar)
      # Remove duplicates, preserving order.
      lin_invars = list(dict.fromkeys(lin_invars))

      lin_outvars = [outvar]
      if eqn.primitive is linear_p:
        linear_eqns.extend(eqn.params['jax_verify_subgraph'].eqns)
      else:
        linear_eqns.append(eqn)
      sub_jaxpr = jax.core.Jaxpr(constvars=[], invars=lin_invars,
                                 outvars=lin_outvars, eqns=linear_eqns)

      if (consumed_by[outvar] == 1) and (outvar in consumed_by_linear):
        # If it's going to be consumed by only a linear operation, put it in the
        # to_be_folded dictionary, it will be included with the following linear
        # equation.
        to_be_folded[outvar] = sub_jaxpr
      else:
        # If it's consumed by multiple things, or by a non-linear operation, or
        # is a terminal output, it does not need folding and should be included
        # now.
        agg_lin_eqn = jax.core.new_jaxpr_eqn(
            sub_jaxpr.invars, sub_jaxpr.outvars, linear_p,
            {'jax_verify_subgraph': sub_jaxpr, 'jax_verify_keepjvargs': True},
            jax.core.no_effects)
        new_eqns.append(agg_lin_eqn)
    else:
      # Non linear operation just gets included directly
      new_eqns.append(eqn)

  assert not to_be_folded

  simple_graph = jax.core.Jaxpr(graph.constvars, graph.invars,
                                graph.outvars, new_eqns)
  # There are some cases where this analysis misses combining some linear
  # operations that can be combined, because there is a branching factor that
  # can be resolved.
  # One example: Making the output of a linear layer mean 0 would be this:
  # y=Linear(x) -------------------> u = y - t ---->
  #            \--->t = mean(y)---/
  # It appears that y is consumed by two different operations, but in practice,
  # those two operations can be part of the same operation.
  # Re-applying the simplification will succeed in merging those, so we will
  # recursively simplify, until we have reached a fixed point (counted as a
  # number of eqns in the graph.)
  # In this example, after the first pass we would have identified two linear
  # blocks:
  # y = Linear(x)
  #   and
  # u = y - mean(y).
  # While in the first pass, it looks like y is consumed by two operations, they
  # get agglomerated together which means that y has now only one dependent.

  if len(new_eqns) != len(graph.eqns):
    return group_linear_sequence(simple_graph, var_is_bound)
  else:
    return simple_graph


def group_fused_relu(graph: jax.core.Jaxpr,
                     var_is_bound: VarIsBoundDict
                     ) -> jax.core.Jaxpr:
  """Simplifier identifying FusedRelus (Linear followed by a ReLU).

  The ReLU primitive will be replaced by a FusedRelu primitive, which
  is responsible for implementing the ReLU, but will in appearance
  also take the inputs to the linear as input and have as a special
  parameter the implementation of the linear layer.

  From a graph like this,
  o ----[linear]------> o -----[relu]----------> o
  we will move to graph like this:
    >------------------------------|
    |                              v
  o -----[linear]----> o -----[fused_relu]----> o
                               fused_linear:[linear]

  The goal of this is that we can use the knowledge we have of the
  preceding linear operation to get a tighter relaxation of the ReLU.

  Args:
    graph: Jaxpr to simplify.
    var_is_bound: Dict mapping whether a given variable is a bound or not.
  Returns:
    Simplified Jaxpr, where all the fused ReLU have been identified.
  """
  # Pass through the network to find what variables are eligible to be
  # the intermediate variable of a fused ReLU.
  # The conditions are:
  #   - produced by a linear.
  #   - consumed by a ReLU.
  #   - Not consumed by anything else.
  is_linear_variable = {}
  consumed_by = collections.Counter()
  for eqn in graph.eqns:
    for invar in eqn.invars:
      if not isinstance(invar, jax.core.Literal) and var_is_bound[invar]:
        consumed_by[invar] += 1
    is_linear_variable[eqn.outvars[0]] = eqn.primitive is linear_p
  # Increase consumed_by for graph_input so that we don't fuse a graph output.
  for outvar in graph.outvars:
    if isinstance(outvar, jax.core.Var):
      consumed_by[outvar] += 1

  # Identify exactly which variables are involved in FusedRelus, based on the
  # information collected.
  fused_relu_interm_to_output = {}
  for eqn in graph.eqns:
    if (eqn.primitive is relu_p and
        is_linear_variable[eqn.invars[0]] and
        consumed_by[eqn.invars[0]] == 1):
      fused_relu_interm_to_output[eqn.invars[0]] = eqn.outvars[0]

  # Let's now create the new list of eqns where we replace the eqns involved in
  # the fused ReLU by the fused ReLU.
  new_eqns = []
  for eqn in graph.eqns:
    if eqn.outvars[0] in fused_relu_interm_to_output:
      # This is the linear part of the fused ReLU.
      linear_eqn = eqn
      # Get the corresponding ReLU.
      relu_eqn_idx = _find_eqn(graph.eqns,
                               fused_relu_interm_to_output[eqn.outvars[0]])
      relu_eqn = graph.eqns[relu_eqn_idx]
      # Let's now build the fused ReLU primitive
      non_literal_invars = [invar for invar in eqn.invars
                            if isinstance(invar, jax.core.Var)]
      fused_relu_invars = [eqn.outvars[0], *non_literal_invars]
      fused_relu_jaxpr = jax.core.Jaxpr(
          constvars=[], invars=fused_relu_invars,
          outvars=relu_eqn.outvars, eqns=[relu_eqn])
      # Keep the linear eqn in the jaxpr, so that we can concretize it.
      new_eqns.append(linear_eqn)
      # Insert the relu at that level, with an addition of a copy of the
      # linear operation preceding it so that we can use it for the
      # relaxation.
      new_eqns.append(jax.core.new_jaxpr_eqn(
          fused_relu_jaxpr.invars, fused_relu_jaxpr.outvars, fused_relu_p,
          {'jax_verify_subgraph': fused_relu_jaxpr,
           'jax_verify_keepjvargs': True,
           'jax_verify_fusedlinear': linear_eqn},
          jax.core.no_effects))
    elif (eqn.primitive is relu_p and
          eqn.invars[0] in fused_relu_interm_to_output):
      # This is the relu part of the fused relu. We already included it.
      pass
    else:
      new_eqns.append(eqn)

  return jax.core.Jaxpr(graph.constvars, graph.invars, graph.outvars, new_eqns)


def hoist_constant_computations(graph: jax.core.Jaxpr,
                                var_is_bound: VarIsBoundDict
                                ) -> jax.core.Jaxpr:
  """Rearrange the equations to make for easier to reason about JaxPr.

  All constant computations should be done at the beginning.
  Args:
    graph: Jaxpr to simplify
    var_is_bound: Dict mapping whether a given variable is a bound or not.
  Returns:
    Simplified Jaxpr, where all non-bound computation are at the beginning.
  """
  new_eqns = []
  # Do a pass, including all the constant computations.
  for eqn in graph.eqns:
    if not var_is_bound[eqn.outvars[0]]:
      new_eqns.append(eqn)
  # Do a pass, including all the bound computations
  for eqn in graph.eqns:
    if var_is_bound[eqn.outvars[0]]:
      new_eqns.append(eqn)
  return jax.core.Jaxpr(graph.constvars, graph.invars,
                        graph.outvars, new_eqns)


def _get_count_and_suffix(graph: jax.core.Jaxpr) -> tuple[int, str]:
  # We are going to be creating new variables.
  # Let's find out what level we need to start counting from.
  max_count = 0
  suffix = ''
  for eqn in graph.eqns:
    for outvar in eqn.outvars:
      if outvar.count > max_count:
        max_count = outvar.count
        suffix = outvar.suffix
  return max_count, suffix


def expand_softmax_simplifier(graph: jax.core.Jaxpr,
                              var_is_bound: VarIsBoundDict
                              ) -> jax.core.Jaxpr:
  """Replace the softmax synthetic primitives by its decomposition.

  It might seem like what we would want to do is simply not detect the softmax,
  but simplifying the softmax and then expanding it makes it more verification
  friendly. For example, this allows to remove the primitives that are employed
  for numerical stability (reduce_max, stop_gradient and sub), but don't affect
  output and that we might not handle.

  Note that we as we create new variables, we will modify `var_is_bound` to keep
  it complete.

  Args:
    graph: Jaxpr to simplify
    var_is_bound: Dict mapping whether a given variable is a bound or not.
  Returns:
    Simplified Jaxpr, where all softmax have been expanded into
      Exponential -> sum -> Reciprocal -> multiplication
            |----------------------------/
  """
  max_count, suffix = _get_count_and_suffix(graph)
  new_var_idx = max_count + 1
  new_eqns = []

  find_prim = lambda eqns, prim: [eqn.primitive for eqn in eqns].index(prim)
  for eqn in graph.eqns:
    if eqn.primitive == softmax_p:
      # We will ignore the operations performed for numerical stability , (the
      # removal of the constant) and keep only the operations we need to
      # propagate bound through. We will also replace the division by a
      # reciprocal, and a multiplication.

      # In order to take away some of the guess work, we will spy what the aval
      # are, and the configurations for some of the parameters.
      softmax_subgraph = eqn.params['jax_verify_subgraph']
      if (
          len(softmax_subgraph.eqns) == 1
          and softmax_subgraph.eqns[0].primitive
          == jax.custom_derivatives.custom_jvp_call_p
      ):
        # The softmax is wrapped inside of a custom_jvp. We'll use the custom
        # jvp subgraph as the softmax subgraph.
        softmax_subgraph = softmax_subgraph.eqns[0].params['call_jaxpr']

      exp_index = find_prim(softmax_subgraph.eqns, lax.exp_p)
      full_size_aval = softmax_subgraph.eqns[exp_index].outvars[0].aval

      exp_var = jax.core.Var(new_var_idx, suffix, full_size_aval)
      var_is_bound[exp_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          eqn.invars, [exp_var], lax.exp_p, {}, jax.core.no_effects))

      # Let's find the parameters of the reduce_sum and of the broadcast
      # operation in the original softmax implementation so that we don't have
      # to do guesswork

      # Add the reduce sum eqn
      reduce_sum_index = find_prim(softmax_subgraph.eqns, lax.reduce_sum_p)
      orig_reduce_sum = softmax_subgraph.eqns[reduce_sum_index]
      reduced_size_aval = orig_reduce_sum.outvars[0].aval
      exp_sum_var = jax.core.Var(new_var_idx, suffix, reduced_size_aval)
      var_is_bound[exp_sum_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [exp_var], [exp_sum_var], lax.reduce_sum_p, orig_reduce_sum.params,
          jax.core.no_effects))

      # Add the broadcasting of it.
      broadcast_index = find_prim(softmax_subgraph.eqns, lax.broadcast_in_dim_p)
      orig_broadcast = softmax_subgraph.eqns[broadcast_index]
      broad_size_aval = orig_broadcast.outvars[0].aval
      broad_expsum_var = jax.core.Var(new_var_idx, suffix, broad_size_aval)
      var_is_bound[broad_expsum_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [exp_sum_var], [broad_expsum_var], lax.broadcast_in_dim_p,
          orig_broadcast.params, jax.core.no_effects))

      # Take the inverse of the exp sum
      inv_expsum_var = jax.core.Var(new_var_idx, suffix, broad_size_aval)
      var_is_bound[inv_expsum_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [broad_expsum_var], [inv_expsum_var], posreciprocal_p, {},
          jax.core.no_effects))

      # Multiply the exponential to the (inv exp sum)
      softmax_var = eqn.outvars[0]
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [exp_var, inv_expsum_var], [softmax_var], jax.lax.mul_p, {},
          jax.core.no_effects))
    else:
      new_eqns.append(eqn)

  return jax.core.Jaxpr(graph.constvars, graph.invars,
                        graph.outvars, new_eqns)


def replace_eltwise_minimum(graph: jax.core.Jaxpr,
                            var_is_bound: VarIsBoundDict,
                            ) -> jax.core.Jaxpr:
  """Replace the elementwise min primitive by an equivalent max formulation."""
  max_count, suffix = _get_count_and_suffix(graph)
  new_var_idx = max_count + 1
  new_eqns = []

  for eqn in graph.eqns:
    if (eqn.primitive == lax.min_p) and var_is_bound[eqn.outvars[0]]:
      # Create negation of the first argument.
      neg_inp_0_var = jax.core.Var(new_var_idx, suffix, eqn.invars[0].aval)
      var_is_bound[neg_inp_0_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          eqn.invars[0:1], [neg_inp_0_var], lax.neg_p, {}, jax.core.no_effects))

      # Create negation of the second argument.
      neg_inp_1_var = jax.core.Var(new_var_idx, suffix, eqn.invars[1].aval)
      var_is_bound[neg_inp_1_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          eqn.invars[1:2], [neg_inp_1_var], lax.neg_p, {}, jax.core.no_effects))

      # Create the elementwise maximum
      neg_min = jax.core.Var(new_var_idx, suffix, eqn.outvars[0].aval)
      var_is_bound[neg_min] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [neg_inp_0_var, neg_inp_1_var], [neg_min], lax.max_p, {},
          jax.core.no_effects))

      # Negate to obtain the elementwise minimum
      elt_min_outvar = eqn.outvars[0]
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [neg_min], [elt_min_outvar], lax.neg_p, {}, jax.core.no_effects))
    else:
      new_eqns.append(eqn)

  return jax.core.Jaxpr(graph.constvars, graph.invars, graph.outvars, new_eqns)


def replace_eltwise_maximum(graph: jax.core.Jaxpr,
                            var_is_bound: VarIsBoundDict,
                            ) -> jax.core.Jaxpr:
  """Replace the elementwise max primitive by a ReLU and sum."""
  max_count, suffix = _get_count_and_suffix(graph)
  new_var_idx = max_count + 1
  new_eqns = []

  for eqn in graph.eqns:
    # If this in an elementwise maximum, that is not a ReLU
    if ((eqn.primitive == lax.max_p) and var_is_bound[eqn.outvars[0]] and
        not (isinstance(eqn.invars[1], jax.core.Literal)
             and eqn.invars[1] == 0.)):

      # We know that this is a an elementwise maximum operation max(a, b).
      # We are going to rewrite it as b + ReLU(a - b)

      # Create the difference between the two arguments of the elementwise max.
      diff_var = jax.core.Var(new_var_idx, suffix, eqn.invars[0].aval)
      var_is_bound[diff_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          eqn.invars, [diff_var], lax.sub_p, {}, jax.core.no_effects))

      # Create the ReLU of the difference
      relued_diff_var = jax.core.Var(new_var_idx, suffix, eqn.outvars[0].aval)
      var_is_bound[relued_diff_var] = True
      new_var_idx += 1
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [diff_var], [relued_diff_var], relu_p, {}, jax.core.no_effects))

      # Add the second term back.
      max_outvar = eqn.outvars[0]
      new_eqns.append(jax.core.new_jaxpr_eqn(
          [relued_diff_var, eqn.invars[1]], [max_outvar], lax.add_p, {},
          jax.core.no_effects))
    else:
      new_eqns.append(eqn)
  return jax.core.Jaxpr(graph.constvars, graph.invars, graph.outvars, new_eqns)


class FakePrimitive(jax.core.Primitive):
  """This wraps an implementation of a primitive we want to identify.

  This way our code that assumes that it can go through the primitive by calling
  `bind` will work transparently.
  We don't want to define it properly as a primitive because otherwise
  operations like the jit will assume that it's a real primitive and not have
  definitions for it.
  """

  def __init__(self, name, impl, always_make_node=False):  # pylint: disable=redefined-outer-name
    self._name = name
    self._impl = impl
    self._always_make_node = always_make_node
    self._register_lowering()

  def bind(self, *args, **kwargs):
    params = {k: v for k, v in kwargs.items() if k != 'jax_verify_subgraph'}
    return self._impl(*args, **params)

  @property
  def name(self):
    return self._name

  @property
  def always_make_node(self):
    return self._always_make_node

  def __str__(self):
    return self._name

  def __repr__(self):
    return f'SyntheticPrimitive[{self._name}]'

  def _register_lowering(self):
    """Register the lowering of the primitive.

    In order to be usable inside a pjit, any primitive need to have defined a
    lowering rule. The FakePrimitive are all traceable, so we will use
    lower_fun to produce their lowering rule.
    """
    lowering_rule = mlir.lower_fun(self.bind, multiple_results=False)
    mlir.register_lowering(self, lowering_rule)


def always_make_node(primitive: Primitive):
  return isinstance(primitive, FakePrimitive) and primitive.always_make_node


def simplifier_composition(*graph_simplifiers: GraphSimplifier
                           ) -> GraphSimplifier:
  """Apply each simplifier one after the other."""
  # We leave it as a list of simplifiers, and let `_simplify_graph` handle
  # the aggregation. This will allows the first simplifier to be applied to
  # the entire graph including sub-graphs, before the next simplifier is
  # applied.
  return list(graph_simplifiers)


def _subgraph_bind(*args, **kwargs):
  """Implement the primitive by iterating through the subgraph."""
  jax_verify_subgraph = kwargs['jax_verify_subgraph']
  return jax.core.eval_jaxpr(jax_verify_subgraph, [], *args)[0]


class SubgraphPrimitive(FakePrimitive):
  """Fake primitive delegating to the implementation to jax_verify_subgraph."""

  def __init__(self, name):
    super().__init__(name, _subgraph_bind)

  def bind(self, *args, **kwargs):
    return self._impl(*args, **kwargs)


convert_float32_p = FakePrimitive(
    'ConvertFloat32', functools.partial(lax.convert_element_type,
                                        new_dtype=jnp.float32))
sigmoid_p = FakePrimitive('Sigmoid', jax.nn.sigmoid)
softplus_p = FakePrimitive('Softplus', jax.nn.softplus)
softmax_p = FakePrimitive('Softmax', jax.nn.softmax)
relu_p = FakePrimitive('ReLU', jax.nn.relu)
leaky_relu_p = FakePrimitive('LeakyRelu', jax.nn.leaky_relu)
parametric_leaky_relu_p = FakePrimitive(
    'ParametricLeakyRelu', jax.nn.leaky_relu)
clip_p = FakePrimitive('Clip', jnp.clip)
posreciprocal_p = FakePrimitive('PosReciprocal', jax.lax.reciprocal)
sech_p = FakePrimitive('Sech', lambda x: 1. / jnp.cosh(x))
linear_p = SubgraphPrimitive('Linear')
posbilinear_p = SubgraphPrimitive('PosBilinear')
quadratic_p = SubgraphPrimitive('Quadratic')
fused_relu_p = SubgraphPrimitive('FusedRelu')


def _make_specs(
    fn: TensorFun,
    upstream_simplifier: Optional[SimpleSimplifier],
    primitive: 'FakePrimitive',
    *arg_shapes_dtypes: Union[tuple[Sequence[int], jnp.dtype],
                              Callable[[], Tensor]],
    **params) -> Sequence[SyntheticPrimitiveSpec]:
  """Create specs for the variants of the function that we might encounter."""
  specs = [SyntheticPrimitiveSpec(
      fn, upstream_simplifier, primitive,
      *arg_shapes_dtypes, **params)]
  if hasattr(fn, '__wrapped__'):
    specs.append(SyntheticPrimitiveSpec(
        fn.__wrapped__, upstream_simplifier, primitive,
        *arg_shapes_dtypes, **params))
  return specs


def activation_specs() -> Sequence[SyntheticPrimitiveSpec]:
  """Returns specs of activations to be replaced with synthetic primitives."""
  synthetic_primitive_specs = []
  # # ReLU.
  synthetic_primitive_specs.extend(_make_specs(
      jax.nn.relu, None, relu_p, ([], jnp.float32)))
  # ReLU may also occur as an explicit max with zero.
  synthetic_primitive_specs.extend(_make_specs(
      lambda x: jnp.maximum(x, 0.), None, relu_p, ([], jnp.float32)))
  # Clipped
  synthetic_primitive_specs.extend(_make_specs(
      jnp.clip, None, clip_p, ([], jnp.float32),
      a_min=capture_float32, a_max=capture_float32))
  # Clipped, with tensors inputs.
  synthetic_primitive_specs.extend(_make_specs(
      jnp.clip, None, clip_p,
      ([], jnp.float32), ([], jnp.float32), ([], jnp.float32)))
  # Softplus.
  synthetic_primitive_specs.extend(_make_specs(
      jax.nn.softplus, None, softplus_p, ([], jnp.float32)))
  # Softmax (n-D).
  for rank in range(1, 9):
    synthetic_primitive_specs.extend(_make_specs(
        jax.nn.softmax, None, softmax_p, ([2] * rank, jnp.float32),
        axis=(rank - 1)))
  # LeakyRelu
  synthetic_primitive_specs.extend(_make_specs(
      jax.nn.leaky_relu, None, leaky_relu_p, ([], jnp.float32),
      negative_slope=capture_float32))
  # LeakyRelu with a learnable negative_slope per neuron (input to the function)
  synthetic_primitive_specs.extend(_make_specs(
      jax.nn.leaky_relu, None, parametric_leaky_relu_p,
      ([], jnp.float32), ([], jnp.float32)))
  # Sigmoid
  synthetic_primitive_specs.extend(_make_specs(
      jax.nn.sigmoid, None, sigmoid_p, ([], jnp.float32)))
  # Hyperbolic Secant
  synthetic_primitive_specs.extend(_make_specs(
      lambda x: 1. / jnp.cosh(x), None, sech_p, ([], jnp.float32)))
  # Convert to float32.
  # We mostly assume that all of our bounds are float32 tensors. However, if
  # they are only weakly typed, this will result in convert_element_type
  # primitives. These are fine (and are identity from our point of view).
  # We however want to avoid ignoring all of such primitives (in case for
  # example there is a conversion to int in the network, which we definitely
  # would like to error on.)
  # The plan is to detect the primitive we accept and handle them correctly,
  # while not detecting the ones we don't, and let the code error.
  synthetic_primitive_specs.extend(_make_specs(
      functools.partial(lax.convert_element_type, new_dtype=jnp.float32),
      None, convert_float32_p, lambda: jnp.asarray(0.)))
  return synthetic_primitive_specs


LINEAR_OP: Sequence[Primitive] = [
    lax.neg_p,
    lax.concatenate_p,
    lax.reshape_p,
    lax.squeeze_p,
    lax.transpose_p,
    lax.broadcast_in_dim_p,
    lax.gather_p,
    lax.reduce_sum_p,
    lax.add_p,
    lax.scatter_add_p,
    lax.dynamic_slice_p,
    lax.slice_p,
    lax.sub_p,
    linear_p,
    convert_float32_p,
    *([lax.select_p] if hasattr(lax, 'select_p') else []),
    *([lax.select_n_p] if hasattr(lax, 'select_n_p') else []),
]


BILINEAR_OP: Sequence[Primitive] = [
    lax.dot_general_p,
    lax.conv_general_dilated_p,
    lax.mul_p,
    lax.scatter_mul_p,
    posbilinear_p,
]

# Don't use `functools.partial` here. We need to defer the invocation of
# `activation_specs()`, because it relies on Jax having been initialised.
activation_detector = lambda graph: detect(activation_specs(), graph)
activation_simplifier = lambda graph, _: activation_detector(graph)

default_simplifier = simplifier_composition(
    activation_simplifier,
    replace_eltwise_minimum,
    replace_eltwise_maximum,
    hoist_constant_computations,
    group_linear_sequence,
    group_posbilinear,
)

fused_relu_simplifier = simplifier_composition(default_simplifier,
                                               group_fused_relu)
