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

"""Implementation of Interval Bound Propagation.
"""
from typing import Union, Tuple

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils

Tensor = jnp.ndarray
PrimitiveInput = Union[Tensor, bound_propagation.Bound]


class IntervalBound(bound_propagation.Bound):
  """Represent an interval where some activations might be valid."""

  def __init__(self, lower_bound: Tensor, upper_bound: Tensor):
    self.shape = lower_bound.shape
    self._lower_bound = lower_bound
    self._upper_bound = upper_bound

  @property
  def lower(self) -> Tensor:
    return self._lower_bound

  @property
  def upper(self) -> Tensor:
    return self._upper_bound

  def __add__(self, other: PrimitiveInput) -> 'IntervalBound':
    if isinstance(other, IntervalBound):
      return IntervalBound(self.lower + other.lower,
                           self.upper + other.upper)
    else:
      return IntervalBound(self.lower + other,
                           self.upper + other)

  def __radd__(self, other: PrimitiveInput) -> 'IntervalBound':
    return self.__add__(other)

  def __sub__(self, other: PrimitiveInput) -> 'IntervalBound':
    if isinstance(other, IntervalBound):
      return IntervalBound(self.lower - other.upper,
                           self.upper - other.lower)
    else:
      return IntervalBound(self.lower - other,
                           self.upper - other)

  def __rsub__(self, other: PrimitiveInput) -> 'IntervalBound':
    if isinstance(other, IntervalBound):
      return IntervalBound(other.lower - self.upper,
                           other.upper - self.lower)
    else:
      return IntervalBound(other - self.upper,
                           other - self.lower)


def _make_ibp_passthrough_primitive(primitive: jax.core.Primitive):
  """Generate a function that simply apply the primitive to the bounds.

  Args:
    primitive: jax primitive
  Returns:
    ibp_primitive: Function applying transparently the primitive to both
      upper and lower bounds.
  """
  def ibp_primitive(*args: PrimitiveInput, **kwargs) -> IntervalBound:
    # We assume that one version should be called with all the 'lower' bound and
    # one with the upper bound. If there is some argument that is not a bound,
    # we assumed it's just simple parameters and pass them through.
    lower_args = [arg.lower if isinstance(arg, bound_propagation.Bound) else arg
                  for arg in args]
    upper_args = [arg.upper if isinstance(arg, bound_propagation.Bound) else arg
                  for arg in args]

    out_lb = primitive.bind(*lower_args, **kwargs)
    out_ub = primitive.bind(*upper_args, **kwargs)
    return IntervalBound(out_lb, out_ub)
  return ibp_primitive


def _decompose_affine_argument(hs: PrimitiveInput) -> Tuple[Tensor, Tensor]:
  """Decompose an argument for a (bound, parameter) IBP propagation.

  We do not need to know which argument is the bound and which one is the
  parameter because we can simply decompose the operation.

  To propagate W * x, we can simply do
    out_diff = abs(W) * (x.upper - x.lower) / 2
    out_mean = W * (x.upper + x.lower) / 2
    out_ub = out_mean + out_diff
    out_lb = out_mean - out_diff

  This function will separate W and x in the two components we need so that we
  do not have to worry about argument order.

  Args:
    hs: Either an IntervalBound or a jnp.array
  Returns:
    part1, part2: jnp.arrays
  """
  if isinstance(hs, bound_propagation.Bound):
    return (hs.upper - hs.lower) / 2, (hs.upper + hs.lower) / 2
  else:
    return jnp.abs(hs), hs


def _ibp_conv_general_dilated(lhs: PrimitiveInput,
                              rhs: PrimitiveInput,
                              **kwargs
                              ) -> PrimitiveInput:
  """Propagation of IBP bounds through a convolution.

  Args:
    lhs: Input to the convolution.
    rhs: Kernel weights.
    **kwargs: Dict with the parameters of the convolution.
  Returns:
    out_bounds: IntervalBound on the output of the convolution.
  """
  req_arguments = ['window_strides', 'padding', 'lhs_dilation',
                   'rhs_dilation', 'dimension_numbers', 'feature_group_count',
                   'precision']
  lax_conv_params = utils.collect_required_arguments(req_arguments, kwargs)

  if (isinstance(lhs, bound_propagation.Bound) !=
      isinstance(rhs, bound_propagation.Bound)):
    lhses = _decompose_affine_argument(lhs)
    rhses = _decompose_affine_argument(rhs)

    forward_mean = lax.conv_general_dilated(lhses[1], rhses[1],
                                            **lax_conv_params)
    forward_range = lax.conv_general_dilated(lhses[0], rhses[0],
                                             **lax_conv_params)

    out_lb = forward_mean - forward_range
    out_ub = forward_mean + forward_range

    return IntervalBound(out_lb, out_ub)
  elif ((not isinstance(lhs, bound_propagation.Bound)) and
        (not isinstance(rhs, bound_propagation.Bound))):
    # Both are arrays, so can simply go through
    return lax.conv_general_dilated(lhs, rhs, **lax_conv_params)
  else:
    raise ValueError('BoundPropagation through general convolution '
                     'is not supported when both inputs are bounds.')


def _ibp_dot_general(lhs: PrimitiveInput,
                     rhs: PrimitiveInput,
                     **kwargs
                     ) -> PrimitiveInput:
  """Propagation of IBP bounds through a general dot product.

  We don't know if the bound is on the left or right hand side, but we expect
  that one hand is a bound and the other is a constant/parameter.

  Args:
    lhs: First input to the dot primitive.
    rhs: Second input to the dot primitive.
    **kwargs: Dict with the parameters of the general dot product.
  Returns:
    out_bounds: IntervalBound on the output of the dot product.
  """
  if (isinstance(lhs, bound_propagation.Bound) !=
      isinstance(rhs, bound_propagation.Bound)):
    lhses = _decompose_affine_argument(lhs)
    rhses = _decompose_affine_argument(rhs)

    forward_mean = lax.dot_general(lhses[1], rhses[1], **kwargs)
    forward_range = lax.dot_general(lhses[0], rhses[0], **kwargs)

    out_lb = forward_mean - forward_range
    out_ub = forward_mean + forward_range

    return IntervalBound(out_lb, out_ub)

  elif ((not isinstance(lhs, bound_propagation.Bound)) and
        (not isinstance(rhs, bound_propagation.Bound))):
    # Both are arrays, so can simply go through
    return lax.dot_general(lhs, rhs, **kwargs)
  else:
    raise ValueError('BoundPropagation through general dot product '
                     'is not supported when both inputs are bounds.')


def _ibp_mul(lhs: PrimitiveInput, rhs: PrimitiveInput) -> PrimitiveInput:
  """Propagation of IBP bounds through Elementwise multiplication.

  Args:
    lhs: Lefthand side of multiplication.
    rhs: Righthand side of multiplication.
  Returns:
    out_bounds: IntervalBound.
  """
  if (isinstance(lhs, bound_propagation.Bound) !=
      isinstance(rhs, bound_propagation.Bound)):
    # This is the case where one is a Bound and the other is not.
    lhses = _decompose_affine_argument(lhs)
    rhses = _decompose_affine_argument(rhs)

    forward_mean = lhses[1] * rhses[1]
    forward_range = lhses[0] * rhses[0]

    out_lb = forward_mean - forward_range
    out_ub = forward_mean + forward_range

    return IntervalBound(out_lb, out_ub)
  elif ((not isinstance(lhs, bound_propagation.Bound)) and
        (not isinstance(rhs, bound_propagation.Bound))):
    # Both are arrays, so can simply go through
    return lhs * rhs
  else:
    raise ValueError('BoundPropagation through multiply is not supported when '
                     'both inputs are bounds.')


def _ibp_div(lhs: PrimitiveInput, rhs: Tensor) -> PrimitiveInput:
  """Propagation of IBP bounds through Elementwise division.

  We don't support the propagation of bounds through the denominator.

  Args:
    lhs: Numerator of the division.
    rhs: Denominator of the division.
  Returns:
    out_bounds: Bound on the output of the division.
  """
  if isinstance(rhs, bound_propagation.Bound):
    raise ValueError('Bound propagation through the denominator unsupported.')
  return _ibp_mul(lhs, 1. / rhs)


@bound_propagation.unwrapping
def _ibp_add(lhs, rhs):
  """Propagation of IBP bounds through an addition.

  Args:
    lhs: Lefthand side of addition.
    rhs: Righthand side of addition.
  Returns:
    out_bounds: IntervalBound.
  """
  return lhs + rhs


@bound_propagation.unwrapping
def _ibp_sub(lhs, rhs):
  """Propagation of IBP bounds through a substraction.

  Args:
    lhs: Lefthand side of substraction.
    rhs: Righthand side of substraction.
  Returns:
    out_bounds: IntervalBound.
  """
  return lhs - rhs

# Define the mapping from jaxpr primitive to the IBP version.
_input_transform = lambda _, lower, upper: IntervalBound(lower, upper)
_primitives_to_passthrough = [lax.broadcast_in_dim_p,
                              lax.reduce_sum_p,
                              lax.max_p, lax.reshape_p,
                              synthetic_primitives.softplus_p]
_primitive_transform = {primitive: _make_ibp_passthrough_primitive(primitive)
                        for primitive in _primitives_to_passthrough}
_primitive_transform.update({
    lax.conv_general_dilated_p: _ibp_conv_general_dilated,
    lax.dot_general_p: _ibp_dot_general,
    lax.sub_p: _ibp_sub,
    lax.add_p: _ibp_add,
    lax.mul_p: _ibp_mul,
    lax.div_p: _ibp_div,
})
_primitive_transform = {key: utils.simple_propagation(prop_fun)
                        for key, prop_fun in _primitive_transform.items()}
bound_transform = bound_propagation.OpwiseBoundTransform(
    _input_transform, _primitive_transform)


def interval_bound_propagation(function, *bounds):
  """Performs IBP as described in https://arxiv.org/abs/1810.12715.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBounds, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by IBP
  """
  output_bound, _ = bound_propagation.bound_propagation(
      bound_transform, function, *bounds)
  return output_bound
