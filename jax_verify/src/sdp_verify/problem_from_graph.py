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

# Lint as: python3
"""Facilities to construct SDP Verification problem instances."""

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
from jax_verify.src.sdp_verify import utils
import numpy as np


Bound = bound_propagation.Bound
Tensor = bound_propagation.Tensor
Index = bound_propagation.Index
Primitive = bound_propagation.Primitive
TransformContext = bound_propagation.TransformContext
SdpDualVerifInstance = utils.SdpDualVerifInstance


class SdpReluProblem:
  """SDP problem to optimise over a ReLU-based network."""

  def __init__(
      self,
      boundprop_transform: bound_propagation.BoundTransform,
      spec_fn: Callable[..., Tensor],
      *input_bounds: Bound,
  ):
    """Initialises a ReLU-based network SDP problem.

    Args:
      boundprop_transform: Transform to supply concrete bounds.
      spec_fn: Network to verify.
      *input_bounds: Concrete bounds on the network inputs.
    """
    self._output_node, self._env = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(
            _SdpTransform(boundprop_transform)),
        spec_fn, *input_bounds)

  def build_sdp_verification_instance(self) -> SdpDualVerifInstance:
    dual_shapes, dual_types = self._dual_shapes_and_types()
    return SdpDualVerifInstance(
        make_inner_lagrangian=self._build_lagrangian_fn,
        bounds=self._bounds(), dual_shapes=dual_shapes, dual_types=dual_types)

  def _dual_shapes_and_types(self) -> Tuple[
      Sequence[Union[Mapping[str, np.ndarray], np.ndarray]],
      Sequence[Union[Mapping[str, utils.DualVarTypes], utils.DualVarTypes]]]:
    """Returns shapes and types of dual vars."""
    dual_shapes = []
    dual_types = []
    num_kappa = 1
    for node in self._env.values():
      if isinstance(node, Bound) and not node.is_affine:
        node_dual_shapes, node_dual_types = node.dual_shapes_and_types()
        dual_shapes.append(node_dual_shapes)
        dual_types.append(node_dual_types)
        num_kappa += np.prod(node.shape[1:], dtype=np.int32)
    dual_shapes.append(np.array([1, num_kappa]))
    dual_types.append(utils.DualVarTypes.INEQUALITY)
    return dual_shapes, dual_types

  def _bounds(self) -> Sequence[utils.IntBound]:
    return [
        utils.IntBound(lb=node.lower, ub=node.upper, lb_pre=None, ub_pre=None)
        for node in self._env.values()
        if isinstance(node, Bound) and not node.is_affine]

  def _build_lagrangian_fn(
      self,
      dual_vars: Sequence[Union[Mapping[str, Tensor], Tensor]],
  ) -> Callable[[Tensor], Tensor]:
    """Returns a function that computes the Lagrangian for a ReLU network.

    This function assumes `spec_fn` represents a feedforward ReLU network i.e.
    x_{i+1} = relu(W_i x_i + b_i), with a final linear objective. The network
    may be branched (e.g. ResNets or DenseNets), including the objective part
    which may, for example, depend linearly on any of the intermediate
    ReLU activations.

    It defines the Lagrangian by applying the linear/affine functions to the
    inputs and all intermediate activations, and encoding the Lagrangian
    terms for each of the constraints defining the ReLU network. It then returns
    this function.

    Args:
      dual_vars: Dual variables for each ReLU node.

    Returns:
      Function that computes Lagrangian L(x) with fixed `dual_vars`.
    """
    nodes = [node for node in self._env.values()
             if isinstance(node, Bound) and not node.is_affine]
    assert len(dual_vars) == len(nodes) + 1

    def lagrangian(xs):
      """Computes Lagrangian L(x) with fixed `dual_vars`."""
      assert all([x.shape[0] == 1 for x in xs]), 'no batch mode support'
      assert len(xs) == len(nodes)

      ys = {node.index: x for node, x in zip(nodes, xs)}
      for node in self._env.values():
        if isinstance(node, SdpNode):
          node.forward_propagate(ys)

      lag = jnp.reshape(ys[self._output_node.index], ())
      for node, node_dual_vars, x in zip(nodes, dual_vars[:-1], xs):
        lag += node.lagrangian_contrib(node_dual_vars, x, ys)
      return lag
    return lagrangian


class SdpNode(Bound):
  """Node in the ReLU network to be optimised using SDP."""

  def __init__(
      self,
      index: Index,
      base_bound: Bound,
      is_input: bool,
      preact_node: Optional['SdpNode'],
      forward_propagate_fn: Callable[[Dict[Index, Tensor]], None]):
    self._index = index
    self._base_bound = base_bound
    self._is_input = is_input
    self._preact_node = preact_node
    self._forward_propagate_fn = forward_propagate_fn

  @property
  def index(self) -> Index:
    return self._index

  @property
  def base_bound(self) -> Bound:
    return self._base_bound

  @property
  def lower(self) -> Tensor:
    """Concrete lower bound."""
    return self._base_bound.lower

  @property
  def upper(self) -> Tensor:
    """Concrete upper bound."""
    return self._base_bound.upper

  @property
  def is_affine(self) -> bool:
    return not self._is_input and self._preact_node is None

  def dual_shapes_and_types(self) -> Tuple[
      Mapping[str, np.ndarray], Mapping[str, utils.DualVarTypes]]:
    """Returns dual shapes and types for this ReLU (or input) layer."""
    if self.is_affine:
      raise ValueError('No duals for affine layer')
    shape = np.array([1] + list(self.shape[1:]))
    dual_shapes = {'nu': shape}
    dual_types = {'nu': utils.DualVarTypes.INEQUALITY}
    if not self._is_input:
      dual_shapes.update({
          'lam': shape,
          'muminus': shape,
          'muplus': shape,
      })
      dual_types.update({
          'lam': utils.DualVarTypes.EQUALITY,
          'muminus': utils.DualVarTypes.INEQUALITY,
          'muplus': utils.DualVarTypes.INEQUALITY,
      })
    return dual_shapes, dual_types

  def lagrangian_contrib(
      self,
      dual_vars: Mapping[str, Tensor],
      x: Tensor,
      ys: Dict[Index, Tensor],
  ) -> Tensor:
    """Returns contribution of this ReLU (or input) layer to the Lagrangian.

    Args:
      dual_vars: Dual variables for this node.
      x: Primal value for this activation (or input).
      ys: Primal values for all pre-activations.

    Dual variables correspond to:
    lam: ReLU quadratic constraint: z^2 = z*(Wx)
    nu: IBP quadratic constraint: x^2 <= (l+u)*x - l*u
    muminus: x'>=0
    muplus: x'>=Wx+b
    """
    if self.is_affine:
      raise ValueError('No Lagrangian contribution for affine layer')
    lag = 0.
    if not self._is_input:
      y = ys[self._preact_node.index]
      # Lagrangian for constraint x' * x' = x' * (Wx+b) where x'=ReLU(Wx+b)
      lag += jnp.sum(dual_vars['lam'] * x * (y - x))
      # Lagrangian for the constraint x'>=Wx+b
      lag += jnp.sum(dual_vars['muplus'] * (x - y))

      # Lagrangian for the constraint x'>=0
      lag += jnp.sum(dual_vars['muminus'] * x)

    # Lagrangian for IBP constraint (x-l)(x-u) <= 0
    if 'nu' in dual_vars:
      lag += -jnp.sum(dual_vars['nu'] *
                      (x - self.lower) * (x - self.upper))
    return lag

  def forward_propagate(self, xs: Dict[Index, Tensor]):
    self._forward_propagate_fn(xs)


class _SdpTransform(bound_propagation.GraphTransform[SdpNode]):
  """Converts a specification function into an SDP problem."""

  def __init__(self, boundprop_transform: bound_propagation.BoundTransform):
    super().__init__()
    self._boundprop_transform = boundprop_transform

  def input_transform(
      self,
      context: TransformContext,
      lower_bound: Tensor,
      upper_bound: Tensor,
  ) -> SdpNode:
    bound = self._boundprop_transform.input_transform(
        context, lower_bound, upper_bound)
    return SdpNode(context.index, bound, True, None, lambda ys: None)

  def primitive_transform(
      self,
      context: TransformContext,
      primitive: Primitive,
      *args: Union[SdpNode, Tensor],
      **params,
  ) -> SdpNode:
    arg_bounds = [arg.base_bound if isinstance(arg, SdpNode) else arg
                  for arg in args]
    bound = self._boundprop_transform.equation_transform(
        context, primitive, *arg_bounds, **params)
    if primitive == synthetic_primitives.relu_p:
      preact, = args
      return SdpNode(context.index, bound, False, preact, lambda ys: None)
    else:
      def forward_propagate(ys: Dict[Index, Tensor]):
        xs = [ys[arg.index] if isinstance(arg, Bound) else arg for arg in args]
        ys[context.index] = primitive.bind(*xs, **params)
      return SdpNode(context.index, bound, False, None, forward_propagate)

