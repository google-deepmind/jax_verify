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

"""Backpropagation of sensitivity values.

This computes a measure of dependence of the output objective upon each
intermediate value of a ReLU-based neural network. This sensitivity measure
is constructed by back-propagating through the network, and linearising
each ReLU.

This sensitivity measure is derived in "Provable Defenses via the Convex Outer
Adversarial Polytope", https://arxiv.org/pdf/1711.00851.pdf.
"""

import functools
from typing import Mapping, Optional, Sequence, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src.types import Index, Nest, Primitive, Tensor  # pylint: disable=g-multiple-import


def _sensitivity_linear_op(
    primitive: Primitive,
    outval: Tensor,
    *args: Union[bound_propagation.Bound, Tensor],
    **kwargs) -> Sequence[Optional[Tensor]]:
  """Back-propagates sensitivity through a linear Jax operation.

  For linear ops, sensitivity of the inputs is computed by applying the
  transpose of the op to the sensitivity of the outputs.

  Args:
    primitive: Linear (or affine) primitive op through which to backprop.
    outval: Sensitivity value for the op's output.
    *args: Inputs to the linear op, in the form of interval bounds (for
      variables) and primal values (for constants).
    **kwargs: Additional parameters to the linear op.

  Returns:
    Sensitivitity values for the op's variable inputs. Entries will be `None`
    for constant inputs.
  """
  # Use auto-diff to perform the transpose-product.
  primal_args = [
      jnp.zeros_like(arg.lower)
      if isinstance(arg, bound_propagation.Bound) else arg
      for arg in args]

  _, vjp = jax.vjp(functools.partial(primitive.bind, **kwargs), *primal_args)
  return vjp(outval)


def _sensitivity_relu(
    outval: Tensor,
    inp: bound_propagation.Bound
) -> Tensor:
  """Back-propagates sensitivity through a ReLU.

  For the purposes of back-propagating sensitivity,
  the ReLU uses a linear approximation given by the chord
  from (lower_bound, ReLU(lower_bound)) to (upper_bound, ReLU(upper_bound))

  Args:
    outval: Sensitivity values for the ReLU outputs.
    inp: Interval bounds on the ReLU input

  Returns:
    Sensitivity values for the ReLU input.
  """
  # Arrange for always-blocking and always-passing ReLUs to give a slope
  # of zero and one respectively.
  lower_bound = jnp.minimum(inp.lower, 0.)
  upper_bound = jnp.maximum(inp.upper, 0.)

  chord_slope = upper_bound / jnp.maximum(
      upper_bound - lower_bound, jnp.finfo(jnp.float32).eps)
  return chord_slope * outval,  # pytype: disable=bad-return-type  # jax-ndarray


_LINEAR_PRIMITIVES: Sequence[Primitive] = [
    *bound_propagation.AFFINE_PRIMITIVES,
    *bound_propagation.RESHAPE_PRIMITIVES,
    lax.div_p,
]


def _build_sensitivity_ops() -> Mapping[
    Primitive, graph_traversal.PrimitiveBacktransformFn]:
  """Builds functions to back-prop 'sensitivity' through individual primitives.

  Returns:
    Sensitivity computation functions, in the form suitable to be passed to
    `PropagationGraph.backward_propagation()`.
  """
  sensitivity_primitive_ops = {
      primitive: functools.partial(_sensitivity_linear_op, primitive)
      for primitive in _LINEAR_PRIMITIVES}
  sensitivity_primitive_ops[synthetic_primitives.relu_p] = _sensitivity_relu
  # Through the sign function, we don't really have a sensitivity.
  sensitivity_sign = lambda outval, _: (jnp.zeros_like(outval),)
  sensitivity_primitive_ops[jax.lax.sign_p] = sensitivity_sign

  return sensitivity_primitive_ops


sensitivity_backward_transform = graph_traversal.BackwardOpwiseTransform(
    _build_sensitivity_ops(), sum)


class SensitivityAlgorithm(bound_propagation.PropagationAlgorithm[Tensor]):
  """Propagation algorithm computing output sensitivity to intermediate nodes."""

  def __init__(
      self,
      forward_bound_transform: bound_propagation.BoundTransform,
      sensitivity_targets: Sequence[Index],
      output_sensitivity: Optional[Tensor] = None):
    """Define the sensitivity that needs to be computed.

    Args:
      forward_bound_transform: Transformation to use to compute intermediate
        bounds.
      sensitivity_targets: Index of the nodes for which we want to obtain
        sensitivities.
      output_sensitivity: (Optional) Linear coefficients for which we want the
        sensitivity, defined over the output.
    """
    self._forward_bnd_algorithm = bound_propagation.ForwardPropagationAlgorithm(
        forward_bound_transform)
    self._output_sensitivity = output_sensitivity
    self._sensitivity_targets = sensitivity_targets
    self.target_sensitivities = []

  def propagate(self, graph: graph_traversal.PropagationGraph,
                *bounds: Nest[graph_traversal.GraphInput]):
    assert len(graph.outputs) == 1
    out, bound_env = self._forward_bnd_algorithm.propagate(graph, bounds)

    if self._output_sensitivity is not None:
      output_sensitivity = self._output_sensitivity
    else:
      output_sensitivity = -jnp.ones(out[0].shape)
    sensitivities, backward_env = graph.backward_propagation(
        sensitivity_backward_transform, bound_env,
        {graph.outputs[0]: output_sensitivity},
        self._sensitivity_targets)

    self.target_sensitivities = sensitivities

    return out, backward_env
