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

"""Implementation of CrownIBP.

This corresponds to doing a forward pass based on IBP, followed by a bounding of
the output based on Crown bounds built using the IBP intermediate bounds.
"""
import functools
import jax

from jax import lax
from jax.interpreters import ad
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src import utils
from jax_verify.src.fastlin import LinearExpression
import numpy as np


class CrownBackwardBound(bound_propagation.Bound):
  """Represent a pair of linear functions that bound the output of the network.

  The linear functions are stored as LinearExpressions. The linear coefficients
  correspond to activation at a given layer of the network and the offset are
  constant terms, of the shape of the network's output.

  Linear Coefficients will have the shape:
    batch_size x (output_shape) x (activation_shape)
  The offset will have the shape:
    batch_size x (output_shape)

  where output_shape and activation_shape can consist of multiple dimensions.
  """

  def __init__(self, lower_lin, upper_lin):
    self.lower_lin = lower_lin
    self.upper_lin = upper_lin

  @property
  def lower(self):
    raise NotImplementedError()

  @property
  def upper(self):
    raise NotImplementedError()

  def __add__(self, other):
    return self if other == 0 else CrownBackwardBound(
        self.lower_lin + other.lower_lin,
        self.upper_lin + other.upper_lin)

  def __radd__(self, other):
    return self if other == 0 else CrownBackwardBound(
        other.lower_lin + self.lower_lin,
        other.upper_lin + self.upper_lin)


def _broadcast_alltargets(arr, lincoeff):
  """Add unit dimensions to a Tensor to match the multiple targets.

  Args:
    arr: Tensor of shape batch_size x (activation_shape)
    lincoeff: Tensor of shape batch_size x (output_shape) x (activation_shape)
  Returns:
    reshaped_arr: `arr` input reshaped to have additional (output_shape) dims.
  """
  nb_dim_to_expand = lincoeff.ndim - arr.ndim
  target_shape = (arr.shape[0],) + (1,)*nb_dim_to_expand + arr.shape[1:]
  return jnp.reshape(arr, target_shape)


def concretize_backward_bound(backward_bound, act_bound):
  """Compute the value of a backward bound.

  Args:
    backward_bound: a CrownBackwardBound, representing linear functions of
     activations lower and upper bounding the output of the network.
    act_bound: Bound on the activation that the backward_bound is a function of.
  Returns:
    bound: A concretized bound
  """
  act_lower = _broadcast_alltargets(act_bound.lower,
                                    backward_bound.lower_lin.lin_coeffs)
  act_upper = _broadcast_alltargets(act_bound.upper,
                                    backward_bound.lower_lin.lin_coeffs)

  nb_dims_to_reduce = act_lower.ndim - backward_bound.lower_lin.offset.ndim
  dims_to_reduce = tuple(range(-nb_dims_to_reduce, 0))

  lower_lin = backward_bound.lower_lin
  upper_lin = backward_bound.upper_lin
  lower_bound = (lower_lin.offset
                 + jnp.sum(jnp.minimum(lower_lin.lin_coeffs, 0.) * act_upper
                           + jnp.maximum(lower_lin.lin_coeffs, 0.) * act_lower,
                           dims_to_reduce))
  upper_bound = (upper_lin.offset
                 + jnp.sum(jnp.maximum(upper_lin.lin_coeffs, 0.) * act_upper
                           + jnp.minimum(upper_lin.lin_coeffs, 0.) * act_lower,
                           dims_to_reduce))

  return ibp.IntervalBound(lower_bound, upper_bound)


def _crown_linear_op(lin_primitive, out_bound, *invals, **kwargs):
  """Backward propagation of LinearBounds through the primitive `lin_primitive`.

  This is achieved by piggybacking on the auto-differentiation code and relying
  on the fact that the backward propagation through linear layer is the
  transpose of the linear operation, which is the same as the operation done
  for gradient backpropagation.

  Args:
    lin_primitive: Jax primitive representing a bilinear operation, to go
      backward through.
    out_bound: CrownBackwardBound, linear function of the network outputs with
      regards to the output activation of this layer.
    *invals: input of the bound propagation in the forward pass
    **kwargs: Dict with the parameters of the linear operation.
  Returns:
    new_in_args: List of CrownBackwardBound
  """
  backward_primitive = ad.get_primitive_transpose(lin_primitive)

  to_backprop = jnp.concatenate([out_bound.lower_lin.lin_coeffs,
                                 out_bound.upper_lin.lin_coeffs], axis=1)
  nb_coeff_dim = (out_bound.lower_lin.lin_coeffs.ndim
                  - out_bound.lower_lin.offset.ndim)
  nb_output_dim = out_bound.lower_lin.offset.ndim - 1
  cts_in = to_backprop

  unwrapped_invals = []
  for inval in invals:
    if isinstance(inval, ibp.IntervalBound):
      # Create a fake input that would have matched the artificial construct
      # we defined as cts_in
      shape = cts_in.shape[:-nb_coeff_dim] + inval.shape[1:]
      unwrapped_invals.append(ad.UndefinedPrimal(jnp.zeros(shape)))
    elif isinstance(inval, jnp.ndarray):
      unwrapped_invals.append(inval)
    else:
      raise ValueError('Unexpected input for the crown-ibp'
                       f'primitive for {lin_primitive}.')

  vmap_outaxes = tuple(1 if isinstance(arg, ibp.IntervalBound) else None
                       for arg in invals)
  vmap_inaxes = (1,) + vmap_outaxes
  backward_op = functools.partial(backward_primitive, **kwargs)
  vmap_backward_op = backward_op
  for _ in range(nb_output_dim):
    # Vmap over all the dimensions that we need to pass through.
    vmap_backward_op = jax.vmap(vmap_backward_op, in_axes=vmap_inaxes,
                                out_axes=vmap_outaxes)
  cts_out = vmap_backward_op(cts_in, *unwrapped_invals)

  new_in_args = []
  for arg in cts_out:
    if arg is None:
      # This correspond to the input that was a constant, we don't want
      # to propagate anything there.
      new_in_args.append(arg)
    else:
      lower_lin_coeffs, upper_lin_coeffs = jnp.split(arg, 2, axis=1)
      new_in_args.append(CrownBackwardBound(
          LinearExpression(lower_lin_coeffs, out_bound.lower_lin.offset),
          LinearExpression(upper_lin_coeffs, out_bound.upper_lin.offset)))
  return new_in_args


def _crown_div(out_bound, lhs, rhs):
  """Backward propagation of LinearBounds through an addition.

  This is a linear operation only in the case where this is a division by a
  constant.

  Args:
    out_bound: CrownBackwardBound, linear function of network outputs bounds
      with regards to the results of the division.
    lhs: Numerator of the division.
    rhs: Denominator of the division.
  Returns:
    new_in_args: List of CrownBackwardBounds or Nones
  """
  if isinstance(rhs, bound_propagation.Bound):
    raise ValueError('Bound propagation through the denominator unsupported.')
  return _crown_linear_op(lax.div_p, out_bound, lhs, rhs)


def _crown_add(out_bound, lhs, rhs):
  """Backward propagation of LinearBounds through an addition.

  Args:
    out_bound: CrownBackwardBound, linear function of network outputs bounds
      with regards to the results of the addition.
    lhs: left input to the addition
    rhs: right input to the addition.
  Returns:
    new_in_args: List of CrownBackwardBounds or Nones
  """
  new_in_args = []
  for arg, other_arg in ((lhs, rhs), (rhs, lhs)):
    if isinstance(arg, ibp.IntervalBound):
      if isinstance(other_arg, ibp.IntervalBound):
        # If the two arguments are bounds, we should pass backward the output in
        # each of the branch.
        new_in_args.append(CrownBackwardBound(out_bound.lower_lin,
                                              out_bound.upper_lin))
      else:
        # If one arguments is a bound and the other is a constant, we should
        # pass backward the bound with a modified offset to the bound, and don't
        # need to propagate towards the constant.

        # Broadcast the `other` argument so that we can compute it's impact on
        # the final bound
        other_arg = _broadcast_alltargets(other_arg,
                                          out_bound.lower_lin.lin_coeffs)
        nb_dims_to_reduce = other_arg.ndim - out_bound.lower_lin.offset.ndim
        dims_to_reduce = tuple(range(-nb_dims_to_reduce, 0))

        lower_offset = jnp.sum(out_bound.lower_lin.lin_coeffs * other_arg,
                               dims_to_reduce)
        upper_offset = jnp.sum(out_bound.upper_lin.lin_coeffs * other_arg,
                               dims_to_reduce)

        new_in_arg = CrownBackwardBound(out_bound.lower_lin + lower_offset,
                                        out_bound.upper_lin + upper_offset)
        new_in_args.append(new_in_arg)
    elif isinstance(arg, jnp.ndarray):
      # This was simply a constant. We won't need to propagate further backwards
      new_in_args.append(None)
    else:
      raise ValueError('Unexpected input for _crown_add.')
  return new_in_args


def _crown_sub(out_bound, lhs, rhs):
  """Backward propagation of LinearBounds through a substraction.

  Args:
    out_bound: CrownBackwardBound, linear function of network outputs bounds
      with regards to the results of the addition.
    lhs: left input to the substraction
    rhs: right input to the substraction.
  Returns:
    new_in_args: List of CrownBackwardBounds or Nones
  """

  # We'll decompose this in the form of a multiplication of the right argument
  # by -1, followed by an addition.
  new_bound = _crown_add(out_bound, lhs, rhs)

  if isinstance(new_bound[1], CrownBackwardBound):
    to_neg = new_bound[1]
    new_bound[1] = CrownBackwardBound(
        LinearExpression(-to_neg.lower_lin.lin_coeffs, to_neg.lower_lin.offset),
        LinearExpression(-to_neg.upper_lin.lin_coeffs, to_neg.upper_lin.offset)
    )
  return new_bound


def _crown_max(out_bound, lhs, rhs):
  """Backward propagation of Linear Bounds through a ReLU.

  Args:
    out_bound: CrownBackwardBound, linear function of network outputs bounds
      with regards to the output of the ReLU
    lhs: left input to the max, inputs to the ReLU
    rhs: right input to the max, we assume this to be 0
  Returns:
    lhs_backbound: CrownBackwardBound, linear function of network outputs bounds
      with regards to the inputs of the ReLU.
    rhs_backbound: None, because we assume the second argument to be 0.
  """
  if not (isinstance(lhs, ibp.IntervalBound) and rhs == 0.):
    raise NotImplementedError('Only ReLU implemented for now.')

  relu_on = (lhs.lower >= 0.)
  relu_amb = jnp.logical_and(lhs.lower < 0., lhs.upper >= 0.)
  ub_slope = relu_on.astype(jnp.float32)
  ub_slope += jnp.where(relu_amb,
                        lhs.upper / jnp.maximum(lhs.upper - lhs.lower, 1e-12),
                        jnp.zeros_like(lhs.lower))
  ub_bias = jnp.where(relu_amb, - ub_slope * lhs.lower,
                      jnp.zeros_like(lhs.lower))
  # Crown Relu propagation.
  lb_slope = (ub_slope >= 0.5).astype(jnp.float32)
  lb_bias = jnp.zeros_like(ub_bias)

  lower_lin_coeffs = out_bound.lower_lin.lin_coeffs
  upper_lin_coeffs = out_bound.upper_lin.lin_coeffs

  ub_slope = _broadcast_alltargets(ub_slope, lower_lin_coeffs)
  lb_slope = _broadcast_alltargets(lb_slope, lower_lin_coeffs)

  new_lower_lin_coeffs = (jnp.minimum(lower_lin_coeffs, 0.) * ub_slope
                          + jnp.maximum(lower_lin_coeffs, 0.) * lb_slope)
  new_upper_lin_coeffs = (jnp.maximum(upper_lin_coeffs, 0.) * ub_slope
                          + jnp.minimum(upper_lin_coeffs, 0.) * lb_slope)

  bias_to_conc = ibp.IntervalBound(lb_bias, ub_bias)
  new_offset = concretize_backward_bound(out_bound, bias_to_conc)

  lhs_backbound = CrownBackwardBound(
      LinearExpression(new_lower_lin_coeffs, new_offset.lower),
      LinearExpression(new_upper_lin_coeffs, new_offset.upper))
  rhs_backbound = None

  return lhs_backbound, rhs_backbound


def _crown_reshape(out_bound, inp, new_sizes, dimensions):
  """Backward propagation of Linear Bounds through a reshape."""
  # Rather than parsing these, we will observe the shape of the inputs to deduce
  # what our target shape should be.
  del new_sizes, dimensions

  non_batch_inpshape = inp.shape[1:]

  # How many (output_shape) dimension is there.
  nb_target_dim = (out_bound.lower_lin.lin_coeffs.ndim
                   - out_bound.lower_lin.offset.ndim)
  batch_tgt_dims = nb_target_dim + 1

  lin_coeff_shape = out_bound.lower_lin.lin_coeffs.shape
  tgt_lin_coeff_shape = lin_coeff_shape[:batch_tgt_dims] + non_batch_inpshape
  new_lower_lin_coeffs = jnp.reshape(out_bound.lower_lin.lin_coeffs,
                                     tgt_lin_coeff_shape)
  new_upper_lin_coeffs = jnp.reshape(out_bound.upper_lin.lin_coeffs,
                                     tgt_lin_coeff_shape)
  inp_backbound = CrownBackwardBound(
      LinearExpression(new_lower_lin_coeffs, out_bound.lower_lin.offset),
      LinearExpression(new_upper_lin_coeffs, out_bound.upper_lin.offset)
  )
  new_sizes_backbound = None
  return inp_backbound, new_sizes_backbound


def crownibp_bound_propagation(function, bounds):
  """Performs Crown-IBP as described in https://arxiv.org/abs/1906.06316.

  We first perform IBP to obtain intermediate bounds and then propagate linear
  bounds backwards.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    bounds: jax_verify.IntervalBounds, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by Crown-IBP
  """
  ibp_bound, graph = bound_propagation.bound_propagation(
      ibp.bound_transform, function, bounds)

  # Define the initial bound to propagate backward
  assert hasattr(ibp_bound, 'shape'), (
      'crownibp_bound_propagation requires `function` to output a single array '
      'as opposed to an arbitrary pytree')
  batch_size = ibp_bound.shape[0]
  act_shape = ibp_bound.shape[1:]

  nb_act = np.prod(act_shape)
  identity_lin_coeffs = jnp.reshape(jnp.eye(nb_act), act_shape + act_shape)
  initial_lin_coeffs = jnp.repeat(jnp.expand_dims(identity_lin_coeffs, 0),
                                  batch_size, axis=0)
  initial_offsets = jnp.zeros_like(ibp_bound.lower)

  initial_backward_bound = CrownBackwardBound(
      LinearExpression(initial_lin_coeffs, initial_offsets),
      LinearExpression(initial_lin_coeffs, initial_offsets))

  input_fun, = graph.backward_propagation(
      _primitive_transform, sum, initial_backward_bound)

  return concretize_backward_bound(input_fun, bounds)


_primitive_transform = {
    lax.add_p: _crown_add,
    lax.sub_p: _crown_sub,
    lax.reshape_p: _crown_reshape,
    lax.dot_general_p: functools.partial(_crown_linear_op, lax.dot_general_p),
    lax.mul_p: functools.partial(_crown_linear_op, lax.mul_p),
    lax.div_p: _crown_div,
    lax.conv_general_dilated_p: functools.partial(
        _crown_linear_op, lax.conv_general_dilated_p),
    lax.max_p: _crown_max
}

_primitive_transform = {key: utils.simple_propagation(prop_fun)
                        for key, prop_fun in _primitive_transform.items()}
