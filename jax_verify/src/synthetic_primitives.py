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

"""Provides methods to simplify the JaxPR computation graph.
"""
import jax


def activation_detector(graph):
  """Attempts to simplify the graph by identifying specific activations.

  This is done by recognizing part of the graph and fusing them into synthetic
  primitives. At the moment, only Softplus is detected.

  Args:
    graph: Unprocessed JaxPR graph.
  Returns:
    simple_graph: Potentially modified JaxPR graph.
  """
  new_eqns = []
  eqn_idx = 0
  while eqn_idx < len(graph.eqns):
    if _is_softplus(graph, eqn_idx):
      # In JaxPR, softplus is a call to logaddexp(x, 0) but our restricted
      # primitive only takes x as input.
      softplus_outvars = graph.eqns[eqn_idx + 9].outvars
      softplus_invars = graph.eqns[eqn_idx].invars[:1]
      softplus_eqn = jax.core.new_jaxpr_eqn(softplus_invars, softplus_outvars,
                                            softplus_p, {})
      new_eqns.append(softplus_eqn)
      eqn_idx += 10
    elif graph.eqns[eqn_idx] == jax.lax.convert_element_type_p:
      # Skip this, this should only be consumed by the softplus
      eqn_idx += 1
    else:
      new_eqns.append(graph.eqns[eqn_idx])
      eqn_idx += 1

  simple_graph = jax.core.Jaxpr(graph.constvars, graph.invars,
                                graph.outvars, new_eqns)
  return simple_graph


def _is_softplus(graph, eqn_idx):
  """Check if the the position `eqn_idx` is the start of a softplus sequence."""
  if eqn_idx + 10 > len(graph.eqns):
    # Softplus takes 10 primitive to implement
    return False

  # The last operations should always be the same.
  valid_primitives = (
      (graph.eqns[eqn_idx + 6].primitive == jax.lax.exp_p) and
      (graph.eqns[eqn_idx + 7].primitive == jax.lax.log1p_p) and
      (graph.eqns[eqn_idx + 8].primitive == jax.lax.add_p) and
      (graph.eqns[eqn_idx + 9].primitive == jax.lax.select_p))
  if not valid_primitives:
    return False

  # Check that this is wired correctly to correspond to a softplus.
  # There is a NaN-check
  for pos in range(9):
    if graph.eqns[eqn_idx + pos].primitive == jax.lax.ne_p:
      ne_eqn = graph.eqns[eqn_idx + pos]
      # It's a NaN check if the two inputs compared for equality are the same.
      if ne_eqn.invars[0] != ne_eqn.invars[1]:
        return False
  # Output of the exponential goes to log1p
  if graph.eqns[eqn_idx + 6].outvars[0] != graph.eqns[eqn_idx + 7].invars[0]:
    return False
  # The select is based on the NaN check
  if graph.eqns[eqn_idx + 9].invars[0] != ne_eqn.outvars[0]:
    return False
  # The Non-Nan selected input is the result of the add
  if graph.eqns[eqn_idx + 9].invars[2] != graph.eqns[eqn_idx + 8].outvars[0]:
    return False

  return True


class FakePrimitive:
  """This wraps an implementation of a primitive we want to identify.

  This way our code that assumes that it can go through the primitive by calling
  `bind` will work transparently.
  We don't want to define it properly as a primitive because otherwise
  operations like the jit will assume that it's a real primitive and not have
  definitions for it.
  """

  def __init__(self, name, impl):
    self._name = name
    self._impl = impl

  def bind(self, *args, **kwargs):
    return self._impl(*args, **kwargs)

  @property
  def name(self):
    return self._name

softplus_p = FakePrimitive('Softplus', jax.nn.softplus)
