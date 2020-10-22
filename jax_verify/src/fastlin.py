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

"""Implementation of Fastlin.
"""
import functools
from typing import Union, Tuple, Optional

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src import intersection
from jax_verify.src import utils
import numpy as np

Tensor = jnp.ndarray


class LinearExpression:
  """Describes a set of linear expressions."""

  def __init__(self, lin_coeffs, offset):
    """Creates a LinearExpression object.

    Args:
      lin_coeffs: batch x (nb_coeffs) x ... array
      offset: batch x ... array
    """
    self.lin_coeffs = lin_coeffs
    self.offset = offset

  def coeff_dims(self):
    return self.lin_coeffs.shape[:2]

  @property
  def shape(self):
    return self.offset.shape

  def __add__(self, other):
    if isinstance(other, LinearExpression):
      return LinearExpression(self.lin_coeffs + other.lin_coeffs,
                              self.offset + other.offset)
    else:
      return LinearExpression(self.lin_coeffs, self.offset + other)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, LinearExpression):
      return LinearExpression(self.lin_coeffs - other.lin_coeffs,
                              self.offset - other.offset)
    else:
      return LinearExpression(self.lin_coeffs, self.offset - other)

  def __rsub__(self, other):
    if isinstance(other, LinearExpression):
      return LinearExpression(other.lin_coeffs - self.lin_coeffs,
                              other.offset - self.offset)
    else:
      return LinearExpression(-self.lin_coeffs, other - self.offset,)

  def __truediv__(self, other):
    return LinearExpression(self.lin_coeffs / other, self.offset / other)

  def pack(self) -> Tensor:
    """Return an array suitable for forwarding through a layer.

    This is required because we often have to forward all the lin_coeffs (even
    though they have an extra dimension), and the offset. This function packs
    all elements together in dimension 1 so that we only have to go through the
    module once.

    Use the inverse function `unpack` to recover a LinearExpression

    Returns:
      packed: A tensor with of the shape of the activation at the layer,
        with an additional dimension where all the linear coefficients and the
        offset are concatenated.
    """
    packed = jnp.concatenate([self.lin_coeffs, jnp.expand_dims(self.offset, 1)],
                             axis=1)
    return packed

  @staticmethod
  def unpack(packed_array: Tensor,
             reference: 'LinearExpression') -> 'LinearExpression':
    """Construct a linear expression based on a packed array.

    Packed array might have been obtained from the `pack` function or through
    forwarding a packed array through a layer.

    Args:
      packed_array: Tensor containing a packed array
      reference: Linear expression to copy the coeff dimensions from
    Returns:
      lin_expression: Linear expression
    """
    _, nb_coeffs = reference.coeff_dims()
    lin_coeffs, offset = jnp.split(packed_array, [nb_coeffs], axis=1)
    offset = offset.squeeze(1)
    return LinearExpression(lin_coeffs, offset)


class LinearBound(bound_propagation.Bound):
  """Represent a pair of linear functions that encompass feasible activations.

  We store the linear functions as LinearExpressions objects in `lower_lin` and
  `upper_lin`, and also maintain a reference to the initial bounds on the input
  to be able to concretize the bounds when needed.
  """

  def __init__(self, lower_bound: LinearExpression,
               upper_bound: LinearExpression,
               reference: Optional['LinearBound']):
    self.lower_lin = lower_bound
    self.upper_lin = upper_bound
    self.reference = reference or self
    self._concretized = None

  @property
  def shape(self):
    return self.lower_lin.shape

  @property
  def lower(self):
    if self._concretized is None:
      self.concretize()
    return self._concretized.lower

  @property
  def upper(self):
    if self._concretized is None:
      self.concretize()
    return self._concretized.upper

  def set_concretized(self, interval_bound):
    self._concretized = interval_bound

  @staticmethod
  def initial_linear_bound(lower_bound, upper_bound):
    batch_size = lower_bound.shape[0]
    act_shape = lower_bound.shape[1:]
    input_dim = np.prod(lower_bound.shape[1:])

    sp_lin = jnp.reshape(jnp.eye(input_dim), (input_dim, *act_shape))
    batch_lin = jnp.repeat(jnp.expand_dims(sp_lin, 0), batch_size, axis=0)

    identity_lin = LinearExpression(batch_lin,
                                    jnp.zeros((batch_size, *act_shape)))
    lin_bound = LinearBound(identity_lin, identity_lin, None)
    lin_bound.set_concretized(ibp.IntervalBound(lower_bound, upper_bound))
    return lin_bound

  def concretize(self):
    if self._concretized is not None:
      return self._concretized

    batch_size = self.shape[0]
    nb_act = len(self.shape) - 1
    broad_shape = (batch_size, -1) + (1,)*nb_act
    flat_ref_lb = jnp.reshape(self.reference.lower, broad_shape)
    flat_ref_ub = jnp.reshape(self.reference.upper, broad_shape)

    concrete_lb = (
        (jnp.maximum(self.lower_lin.lin_coeffs, 0.) * flat_ref_lb).sum(axis=1) +
        (jnp.minimum(self.lower_lin.lin_coeffs, 0.) * flat_ref_ub).sum(axis=1) +
        self.lower_lin.offset
    )
    concrete_ub = (
        (jnp.maximum(self.upper_lin.lin_coeffs, 0.) * flat_ref_ub).sum(axis=1) +
        (jnp.minimum(self.upper_lin.lin_coeffs, 0.) * flat_ref_lb).sum(axis=1) +
        self.upper_lin.offset
    )

    self._concretized = ibp.IntervalBound(concrete_lb, concrete_ub)
    return self._concretized

  def __add__(self, other):
    if isinstance(other, LinearBound):
      if self.reference is not other.reference:
        raise ValueError('Adding bounds referring to different inputs.')
      return LinearBound(self.lower_lin + other.lower_lin,
                         self.upper_lin + other.upper_lin,
                         self.reference)
    else:
      return LinearBound(self.lower_lin + other,
                         self.upper_lin + other,
                         self.reference)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, LinearBound):
      if self.reference is not other.reference:
        raise ValueError('Adding bounds refering to different inputs.')
      return LinearBound(self.lower_lin - other.upper_lin,
                         self.upper_lin - other.lower_lin,
                         self.reference)
    else:
      return LinearBound(self.lower_lin - other,
                         self.upper_lin - other,
                         self.reference)

  def __rsub__(self, other):
    if isinstance(other, LinearBound):
      if self.reference is not other.reference:
        raise ValueError('Adding bounds refering to different inputs.')
      return LinearBound(other.lower_lin - self.upper_lin,
                         other.upper_lin - self.lower_lin,
                         self.reference)
    else:
      return LinearBound(other - self.upper_lin,
                         other - self.lower_lin,
                         self.reference)


def _decompose_affine_argument(hs: Union[LinearBound, Tensor]
                               ) -> Tuple[Tensor, Tensor,
                                          Optional[LinearBound]]:
  """Decompose an argument for a (bound, parameter) Fastlin propagation.

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
    hs: Either a LinearBound or a tensor, input argument to an affine function.
  Returns:
    part1: Packed coefficients for the bound "range" if hs is a bound, or
      absolute value of the tensor if hs is a tensor.
    part2: Packed coefficients for the bound "mean" if hs is a bound, or
      value of the tensor if hs is a tensor.
    part3: Reference for the bound if hs is a bound, otherwise None.
  """
  if isinstance(hs, LinearBound):
    return (((hs.upper_lin - hs.lower_lin) / 2).pack(),
            ((hs.upper_lin + hs.lower_lin) / 2).pack(),
            hs.reference)
  else:
    return jnp.abs(hs), hs, None


@bound_propagation.unwrapping
def _fastlin_linear_op(lin_fun, lhs, rhs, **kwargs):
  """Propagation of Linear bounds through the linear operation `lin_fun`.

  Args:
    lin_fun: Linear function to pass through.
    lhs: Either parameters or LinearBound.
    rhs: Either parameters or LinearBound.
    **kwargs: Dict with the parameters of the linear operation.
  Returns:
    out_bounds: LinearBound
  """
  if isinstance(lhs, LinearBound) != isinstance(rhs, LinearBound):

    lhs_0, lhs_1, ref_l = _decompose_affine_argument(lhs)
    rhs_0, rhs_1, ref_r = _decompose_affine_argument(rhs)
    # One of the reference is None, so we keep only the one defined.
    ref = ref_l or ref_r

    vmap_axis = tuple(1 if isinstance(arg, LinearBound) else None
                      for arg in [lhs, rhs])
    unkwarged_linfun = functools.partial(lin_fun, **kwargs)
    vmap_linfun = jax.vmap(unkwarged_linfun, in_axes=vmap_axis, out_axes=1)

    unp_forward_mean = vmap_linfun(lhs_1, rhs_1)
    unp_forward_range = vmap_linfun(lhs_0, rhs_0)

    # pytype: disable=attribute-error
    # ref is necessarily a Bound, because either lhs or rhs is a bound, so
    # it's not possible for `ref_l` and `ref_r` to both be None, but pytype
    # can not deduce that and will raise an attribute-error for lower_lin.
    forward_mean = LinearExpression.unpack(unp_forward_mean, ref.lower_lin)
    forward_range = LinearExpression.unpack(unp_forward_range, ref.lower_lin)
    # pytype: enable=attribute-error
    out_lb = forward_mean - forward_range
    out_ub = forward_mean + forward_range

    return LinearBound(out_lb, out_ub, ref)
  elif ((not isinstance(lhs, bound_propagation.Bound)) and
        (not isinstance(rhs, bound_propagation.Bound))):
    # Both are arrays, so can simply go through
    return lin_fun(lhs, rhs, **kwargs)
  else:
    raise ValueError('BoundPropagation through Linear operation '
                     'is not supported when both inputs are bounds.')


@bound_propagation.unwrapping
def _fastlin_add(lhs, rhs):
  return lhs + rhs


@bound_propagation.unwrapping
def _fastlin_sub(lhs, rhs):
  return lhs - rhs


def _fastlin_div(lhs, rhs):
  """Propagation of Linear bounds through Elementwise division.

  We don't support the propagation of bounds through the denominator.

  Args:
    lhs: Numerator of the division.
    rhs: Denominator of the division.
  Returns:
    out_bounds: Bound on the output of the division.
  """
  if isinstance(rhs, bound_propagation.Bound):
    raise ValueError('Bound propagation through the denominator unsupported.')
  return _fastlin_linear_op(lambda x, y: x/y, lhs, rhs)


def _fastlin_max(lhs, rhs):
  """Propagation of Fastlin bounds through a max.

  At the moment, only the ReLU is supported.

  This relaxes the ReLU with the parallel bounds of slope (ub) / (ub - lb)

  Args:
    lhs: First input to the max, assumed to be a ReLU input
    rhs: Second input to the max, assumed to be 0
  Returns:
    out_bounds: FastlinBound.
  """
  if not (isinstance(lhs, bound_propagation.Bound) and rhs == 0.):
    raise NotImplementedError('Only ReLU is implemented for now.')

  # Incorporate applied concrete bounds from the wrapper, if any.
  lhs_lower, lhs_upper = lhs.lower, lhs.upper
  lhs = lhs.unwrap()

  relu_on = (lhs_lower >= 0.)
  relu_amb = jnp.logical_and(lhs_lower < 0., lhs_upper >= 0.)
  slope = relu_on.astype(jnp.float32)
  slope += jnp.where(relu_amb,
                     lhs_upper / jnp.maximum(lhs_upper - lhs_lower, 1e-12),
                     jnp.zeros_like(lhs_lower))
  ub_offset = jnp.where(relu_amb, - slope * lhs_lower,
                        jnp.zeros_like(lhs_lower))

  broad_slope = jnp.expand_dims(slope, 1)
  ub_lin_coeffs = broad_slope * lhs.upper_lin.lin_coeffs
  lb_lin_coeffs = broad_slope * lhs.lower_lin.lin_coeffs
  ub_lin_offset = slope * lhs.upper_lin.offset + ub_offset
  lb_lin_offset = slope * lhs.lower_lin.offset

  return LinearBound(LinearExpression(lb_lin_coeffs, lb_lin_offset),
                     LinearExpression(ub_lin_coeffs, ub_lin_offset),
                     lhs.reference)


def _crown_max(lhs, rhs):
  """Propagation of Fastlin bounds through a max.

  This relaxes the ReLU with the adaptive choice of lower bounds as described
  for CROWN-ada in https://arxiv.org/abs/1811.00866.

  Args:
    lhs: First input to the max, assumed to be a ReLU input
    rhs: Second input to the max, assumed to be 0
  Returns:
    out_bounds: FastlinBound.
  """
  if not (isinstance(lhs, bound_propagation.Bound) and rhs == 0.):
    raise NotImplementedError('Only ReLU is implemented for now.')

  # Incorporate applied concrete bounds from the wrapper, if any.
  lhs_lower, lhs_upper = lhs.lower, lhs.upper
  lhs = lhs.unwrap()

  relu_on = (lhs_lower >= 0.)
  relu_amb = jnp.logical_and(lhs_lower < 0., lhs_upper >= 0.)
  ub_slope = relu_on.astype(jnp.float32)
  ub_slope += jnp.where(relu_amb,
                        lhs_upper / jnp.maximum(lhs_upper - lhs_lower, 1e-12),
                        jnp.zeros_like(lhs_lower))
  ub_offset = jnp.where(relu_amb, - ub_slope * lhs_lower,
                        jnp.zeros_like(lhs_lower))
  lb_slope = (ub_slope >= 0.5).astype(jnp.float32)

  broad_ub_slope = jnp.expand_dims(ub_slope, 1)
  broad_lb_slope = jnp.expand_dims(lb_slope, 1)
  ub_lin_coeffs = broad_ub_slope * lhs.upper_lin.lin_coeffs
  lb_lin_coeffs = broad_lb_slope * lhs.lower_lin.lin_coeffs
  ub_lin_offset = ub_slope * lhs.upper_lin.offset + ub_offset
  lb_lin_offset = lb_slope * lhs.lower_lin.offset

  return LinearBound(LinearExpression(lb_lin_coeffs, lb_lin_offset),
                     LinearExpression(ub_lin_coeffs, ub_lin_offset),
                     lhs.reference)


@bound_propagation.unwrapping
def _fastlin_coeffwise(
    primitive: jax.core.Primitive,
    *args: Union[LinearBound, jnp.ndarray],
    **params
    ) -> LinearBound:
  """Propagation of Fastlin bounds through a coefficient-wise op like reshape.

  This vectorises the given primitive to apply it to each coefficient of the
  bounds' linear expressions.

  Args:
    primitive: Operation to apply to the Fastlin bounds.
    *args: Arguments of the primitive.
    **params: Keyword arguments of the primitive.
  Returns:
    out_bounds: FastlinBound, reshaped.
  """
  if any(isinstance(arg, LinearBound) for arg in args):
    bound_arg = [arg for arg in args if isinstance(arg, LinearBound)][0]

    vmap_axis = tuple(1 if isinstance(arg, LinearBound) else None
                      for arg in args)
    op = functools.partial(primitive.bind, **params)
    vmap_op = jax.vmap(op, in_axes=vmap_axis, out_axes=1)

    return LinearBound(
        LinearExpression(
            vmap_op(*[arg.lower_lin.lin_coeffs if isinstance(arg, LinearBound)
                      else arg for arg in args]),
            op(*[arg.lower_lin.offset if isinstance(arg, LinearBound)
                 else arg for arg in args])),
        LinearExpression(
            vmap_op(*[arg.upper_lin.lin_coeffs if isinstance(arg, LinearBound)
                      else arg for arg in args]),
            op(*[arg.upper_lin.offset if isinstance(arg, LinearBound)
                 else arg for arg in args])),
        reference=bound_arg.reference)

  else:
    # No interval dependence. Compute the output value directly.
    return primitive.bind(*args, **params)


fastlin_input_transform = (
    lambda _, lower, upper: LinearBound.initial_linear_bound(lower, upper))
fastlin_primitive_transform = {
    lax.add_p: _fastlin_add,
    lax.sub_p: _fastlin_sub,
    lax.max_p: _fastlin_max,
    lax.reshape_p: functools.partial(_fastlin_coeffwise, lax.reshape_p),
    lax.reduce_sum_p: functools.partial(_fastlin_coeffwise, lax.reduce_sum_p),
    lax.dot_general_p: functools.partial(_fastlin_linear_op, lax.dot_general),
    lax.conv_general_dilated_p: functools.partial(
        _fastlin_linear_op, utils.wrapped_general_conv),
    lax.mul_p: functools.partial(_fastlin_linear_op, lambda x, y: x*y),
    lax.div_p: _fastlin_div,
}
fastlin_primitive_transform = {
    key: utils.simple_propagation(fastlin_primitive_transform[key])
    for key in fastlin_primitive_transform
}
fastlin_transform = bound_propagation.OpwiseBoundTransform(
    fastlin_input_transform, fastlin_primitive_transform)


def fastlin_bound_propagation(function, *bounds):
  """Performs FastLin as described in https://arxiv.org/abs/1804.09699.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  output_bound, _ = bound_propagation.bound_propagation(
      fastlin_transform, function, *bounds)
  return output_bound


_crown_input_transform = fastlin_input_transform
_crown_primitive_transform = fastlin_primitive_transform.copy()
_crown_primitive_transform.update({
    lax.max_p: utils.simple_propagation(_crown_max),
})
_crown_transform = bound_propagation.OpwiseBoundTransform(
    _crown_input_transform, _crown_primitive_transform)


def crown_bound_propagation(function, *bounds):
  """Performs CROWN as described in https://arxiv.org/abs/1811.00866.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  output_bound, _ = bound_propagation.bound_propagation(
      _crown_transform, function, *bounds)
  return output_bound


def ibpfastlin_bound_propagation(function, *bounds):
  """Obtains the best of IBP and Fastlin bounds.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  output_bound, _ = bound_propagation.bound_propagation(
      intersection.IntersectionBoundTransform(
          ibp.bound_transform, fastlin_transform),
      function, *bounds)
  return output_bound
