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

"""Implementation of utils for Branch-and-Bound algorithms.

Contains for example algorithm to evaluate the inputs that concretize the
backward linear bounds.
"""
from typing import Mapping, Tuple

import jax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import bound_utils
from jax_verify.src import graph_traversal
from jax_verify.src import utils
from jax_verify.src.linear import backward_crown
from jax_verify.src.linear import linear_relaxations
from jax_verify.src.types import Index, Nest, Tensor  # pylint: disable=g-multiple-import


class NominalEvaluateConcretizingInputAlgorithm(
    bound_propagation.PropagationAlgorithm[Tensor]):
  """Find the input concretizing a backward linear transform and evaluate it.
  """

  def __init__(self,
               intermediate_bounds: Mapping[Index, Tuple[Tensor, Tensor]],
               backward_transform: backward_crown.LinearBoundBackwardTransform):
    self._forward_bnd_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        bound_utils.FixedBoundApplier(intermediate_bounds))
    self._backward_transform = backward_transform

  def propagate(self, graph: graph_traversal.PropagationGraph,
                *bounds: Nest[graph_traversal.GraphInput]):
    assert len(graph.outputs) == 1
    (out,), bound_env = self._forward_bnd_algorithm.propagate(graph, *bounds)

    max_output = (out.upper == out.upper.max()).astype(jnp.float32)
    max_output = jnp.expand_dims(max_output, 0)

    initial_linear_expression = backward_crown.identity(-max_output)
    flat_inputs, _ = jax.tree_util.tree_flatten(*bounds)
    bound_inputs = [inp for inp in flat_inputs
                    if isinstance(inp, bound_propagation.Bound)]
    input_nodes_indices = [(i,) for i in range(len(bound_inputs))]
    inputs_linfuns, back_env = graph.backward_propagation(
        self._backward_transform, bound_env,
        {graph.outputs[0]: initial_linear_expression},
        input_nodes_indices)

    concretizing_bound_inputs = []
    for input_linfun, inp_bound in zip(inputs_linfuns, bound_inputs):
      if input_linfun is not None:
        conc_inp = minimizing_concretizing_input(input_linfun, inp_bound)
        concretizing_bound_inputs.append(conc_inp[0])

    def eval_model(*graph_inputs):
      # We are going to only pass Tensor.
      # Forward propagation simply evaluate the primitive when there is no
      # bound inputs.
      outvals, _ = graph.forward_propagation(None, graph_inputs)  # pytype: disable=wrong-arg-types
      return outvals
    eval_model_boundinps = utils.bind_nonbound_args(eval_model,
                                                    *flat_inputs)
    nominal_outs = eval_model_boundinps(*concretizing_bound_inputs)

    return nominal_outs, back_env


def minimizing_concretizing_input(
    backward_linexp: linear_relaxations.LinearExpression,
    input_bound: bound_propagation.Bound) -> Tensor:
  """Get the input that concretize the backward bound to its lower bound.

  Args:
    backward_linexp: Coefficients of linear functions. The leading batch
      dimension corresponds to different output neurons that need to be
      concretized.
    input_bound: Bound on the activations of that layer. Its shape should
      match the coefficients of the linear functions to concretize.
  Returns:
    concretizing_inp: The input that correspond to the lower bound of the
      linear function given by backward_linexp.
  """
  return concretizing_input_interval_bounds(backward_linexp, input_bound)


def concretizing_input_interval_bounds(
    backward_linexp: linear_relaxations.LinearExpression,
    input_bound: bound_propagation.Bound) -> Tensor:
  """Compute the inputs that achieves the lower bound of a linear function."""
  act_lower = jnp.expand_dims(input_bound.lower, 0)
  act_upper = jnp.expand_dims(input_bound.upper, 0)
  return jnp.where(backward_linexp.lin_coeffs > 0., act_lower, act_upper)
