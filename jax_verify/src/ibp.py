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

"""Implementation of Interval Bound Propagation.
"""
import functools
from typing import Callable, Mapping, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils
from jax_verify.src.linear import linear_relaxations
from jax_verify.src.types import ArgsKwargsCallable, Primitive, Tensor  # pylint: disable=g-multiple-import


IntervalBound = bound_propagation.IntervalBound


def _make_ibp_passthrough_primitive(
    primitive: Primitive,
) -> ArgsKwargsCallable[
    graph_traversal.LayerInput[IntervalBound], IntervalBound]:
  """Generate a function that simply apply the primitive to the bounds.

  Args:
    primitive: jax primitive
  Returns:
    ibp_primitive: Function applying transparently the primitive to both
      upper and lower bounds.
  """
  def ibp_primitive(
      *args: bound_propagation.LayerInput, **kwargs) -> IntervalBound:
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


def _decompose_affine_argument(
    hs: bound_propagation.LayerInput) -> Tuple[Tensor, Tensor]:
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
    part1, part2: jnp.ndarrays
  """
  if isinstance(hs, bound_propagation.Bound):
    return (hs.upper - hs.lower) / 2, (hs.upper + hs.lower) / 2
  else:
    return jnp.abs(hs), hs


def _ibp_bilinear(
    primitive: Primitive,
    lhs: bound_propagation.LayerInput,
    rhs: bound_propagation.LayerInput,
    **kwargs) -> bound_propagation.LayerInput:
  """Propagation of IBP bounds through a bilinear primitive (product, conv).

  We don't know if the bound is on the left or right hand side, but we expect
  that one hand is a bound and the other is a constant/parameter.

  Args:
    primitive: Bilinear primitive.
    lhs: First input to the primitive.
    rhs: Second input to the primitive.
    **kwargs: Parameters of the primitive.
  Returns:
    out_bounds: IntervalBound on the output of the primitive.
  """
  if (isinstance(lhs, bound_propagation.Bound) !=
      isinstance(rhs, bound_propagation.Bound)):
    # This is the case where one is a Bound and the other is not.

    lhses = _decompose_affine_argument(lhs)
    rhses = _decompose_affine_argument(rhs)

    forward_mean = primitive.bind(lhses[1], rhses[1], **kwargs)
    forward_range = primitive.bind(lhses[0], rhses[0], **kwargs)

    out_lb = forward_mean - forward_range
    out_ub = forward_mean + forward_range

    return IntervalBound(out_lb, out_ub)

  elif ((not isinstance(lhs, bound_propagation.Bound)) and
        (not isinstance(rhs, bound_propagation.Bound))):
    # Both are arrays, so can simply go through
    return primitive.bind(lhs, rhs, **kwargs)
  elif primitive == lax.dot_general_p:
    # If both inputs are bounds and this is a `dot_general_p` primitive, we have
    # a special implementation for it with the tighter bounds of using the exact
    # bounds instead of relying on McCormick based bounds.
    return _ibp_dotgeneral_bilinear(lhs, rhs, **kwargs)
  else:
    # If both inputs are bounds but this is not the special case of the
    # dotgeneral primitive, we can use McCormick inequalities for bilinear
    # functions to compute the interval bounds.

    # If x in [x_l, x_u] and y in [y_l, y_u], then the following hold:
    # xy >= y_l*x + x_l*y - x_l*y_l
    # xy >= y_u*x + x_u*y - x_u*y_u
    # xy <= y_u*x + x_l*y - x_l*y_u
    # xy <= y_l*x + x_u*y - x_u*y_l
    # These bounds are also used in the fastlin approach proposed in
    # https://arxiv.org/pdf/2002.06622.pdf

    # TODO: Tighten the bounds further to min/max of
    # x_l*y_l, x_l*y_u, x_u*y_l, and x_u*y_u for more primitives.

    # Use the first McCormick lower bound
    out_lb1 = _ibp_bilinear(primitive, lhs, rhs.lower, **kwargs).lower
    out_lb1 += _ibp_bilinear(primitive, lhs.lower, rhs, **kwargs).lower
    out_lb1 -= _ibp_bilinear(primitive, lhs.lower, rhs.lower, **kwargs)

    # Use the second McCormick lower bound
    out_lb2 = _ibp_bilinear(primitive, lhs, rhs.upper, **kwargs).lower
    out_lb2 += _ibp_bilinear(primitive, lhs.upper, rhs, **kwargs).lower
    out_lb2 -= _ibp_bilinear(primitive, lhs.upper, rhs.upper, **kwargs)

    # Choose the best lower bound out of the two
    out_lb = jnp.maximum(out_lb1, out_lb2)

    # Use the first McCormick upper bound
    out_ub1 = _ibp_bilinear(primitive, lhs, rhs.upper, **kwargs).upper
    out_ub1 += _ibp_bilinear(primitive, lhs.lower, rhs, **kwargs).upper
    out_ub1 -= _ibp_bilinear(primitive, lhs.lower, rhs.upper, **kwargs)

    # Use the second McCormick upper bound
    out_ub2 = _ibp_bilinear(primitive, lhs, rhs.lower, **kwargs).upper
    out_ub2 += _ibp_bilinear(primitive, lhs.upper, rhs, **kwargs).upper
    out_ub2 -= _ibp_bilinear(primitive, lhs.upper, rhs.lower, **kwargs)

    # Choose the best upper bound out of the two
    out_ub = jnp.minimum(out_ub1, out_ub2)

    return IntervalBound(out_lb, out_ub)


def _move_axes(
    bound: bound_propagation.Bound,
    cdims: Tuple[int, ...],
    bdims: Tuple[int, ...],
    orig_axis: int,
    new_axis: int,
) -> Tuple[IntervalBound, Tuple[int, ...], Tuple[int, ...]]:
  """Reorganise the axis of a bound, and the dimension_numbers indexing it.

  The axis in position `orig_axis` gets moved to position `new_axis`.

  Args:
    bound: Bound whose axis needs to be re-organised.
    cdims: Contracting dimensions, pointing at some axis of bound.
    bdims: Batch dimensions, pointing at some axis of bound.
    orig_axis: Index of the axis to move.
    new_axis: New position for this axis.
  Returns:
    new_bound: Re-organised bound.
    new_cdims: Re-organised cdims.
    new_bdims: Re-organised bdims.
  """
  def new_axis_fn(old_axis):
    if old_axis == orig_axis:
      # This is the axis being moved. Return its new position.
      return new_axis
    elif (old_axis < orig_axis) and (old_axis >= new_axis):
      # The original axis being moved was after, but it has now moved to before
      # (or at this space). This means that this axis gets shifted back
      return old_axis + 1
    elif (old_axis > orig_axis) and (old_axis <= new_axis):
      # The original axis being moved was before this one, but it has now moved
      # to after. This means that this axis gets shifted forward.
      return old_axis - 1
    else:
      # Nothing should be changing.
      return old_axis

  mapping = {old_axis: new_axis_fn(old_axis)
             for old_axis in range(len(bound.lower.shape))}
  permutation = sorted(range(len(bound.lower.shape)), key=lambda x: mapping[x])
  new_bound = IntervalBound(jax.lax.transpose(bound.lower, permutation),
                            jax.lax.transpose(bound.upper, permutation))
  new_cdims = tuple(new_axis_fn(old_axis) for old_axis in cdims)
  new_bdims = tuple(new_axis_fn(old_axis) for old_axis in bdims)
  return new_bound, new_cdims, new_bdims


def _ibp_dotgeneral_bilinear(lhs: bound_propagation.Bound,
                             rhs: bound_propagation.Bound,
                             **kwargs
                             ) -> IntervalBound:
  """IBP propagation through a dotgeneral primitive with two bound input.

  Args:
    lhs: First input to the dotgeneral
    rhs: Second input to the primitive.
    **kwargs: Parameters of the primitive.
  Returns:
    out_bounds: Bound on the output of the general dot product.
  """

  (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims) = kwargs['dimension_numbers']

  # Move the contracting dimensions to the front.
  for cdim_index in range(len(lhs_cdims)):
    lhs, lhs_cdims, lhs_bdims = _move_axes(lhs, lhs_cdims, lhs_bdims,
                                           lhs_cdims[cdim_index], cdim_index)
    rhs, rhs_cdims, rhs_bdims = _move_axes(rhs, rhs_cdims, rhs_bdims,
                                           rhs_cdims[cdim_index], cdim_index)

  # Because we're going to scan over the contracting dimensions, the
  # batch dimensions are appearing len(cdims) earlier.
  new_lhs_bdims = tuple(bdim - len(lhs_cdims) for bdim in lhs_bdims)
  new_rhs_bdims = tuple(bdim - len(rhs_cdims) for bdim in rhs_bdims)

  merge_cdims = lambda x: x.reshape((-1,) + x.shape[len(lhs_cdims):])
  operands = ((merge_cdims(lhs.lower), merge_cdims(lhs.upper)),
              (merge_cdims(rhs.lower), merge_cdims(rhs.upper)))
  batch_shape = tuple(lhs.lower.shape[axis] for axis in lhs_bdims)
  lhs_contr_shape = tuple(dim for axis, dim in enumerate(lhs.lower.shape)
                          if axis not in lhs_cdims + lhs_bdims)
  rhs_contr_shape = tuple(dim for axis, dim in enumerate(rhs.lower.shape)
                          if axis not in rhs_cdims + rhs_bdims)
  out_shape = batch_shape + lhs_contr_shape + rhs_contr_shape
  init_carry = (jnp.zeros(out_shape), jnp.zeros(out_shape))

  new_dim_numbers = (((), ()), (new_lhs_bdims, new_rhs_bdims))
  unreduced_dotgeneral = functools.partial(jax.lax.dot_general,
                                           dimension_numbers=new_dim_numbers)

  def scan_fun(carry: Tuple[Tensor, Tensor],
               inp: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
               ) -> Tuple[Tuple[Tensor, Tensor], None]:
    """Accumulates the minimum and maximum as inp traverse the first dimension.

    (The first dimension is where we have merged all contracting dimensions.)

    Args:
      carry: Current running sum of the lower bound and upper bound
      inp: Slice of the input tensors.
    Returns:
      updated_carry: New version of the running sum including these elements.
      None
    """

    (lhs_low, lhs_up), (rhs_low, rhs_up) = inp
    carry_min, carry_max = carry
    opt_1 = unreduced_dotgeneral(lhs_low, rhs_low)
    opt_2 = unreduced_dotgeneral(lhs_low, rhs_up)
    opt_3 = unreduced_dotgeneral(lhs_up, rhs_low)
    opt_4 = unreduced_dotgeneral(lhs_up, rhs_up)
    elt_min = jnp.minimum(jnp.minimum(jnp.minimum(opt_1, opt_2), opt_3), opt_4)
    elt_max = jnp.maximum(jnp.maximum(jnp.maximum(opt_1, opt_2), opt_3), opt_4)
    return (carry_min + elt_min, carry_max + elt_max), None

  (lower, upper), _ = jax.lax.scan(scan_fun, init_carry, operands)

  return IntervalBound(lower, upper)


def _ibp_div(
    lhs: bound_propagation.LayerInput,
    rhs: Tensor,
) -> bound_propagation.LayerInput:
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
  return _ibp_bilinear(lax.mul_p, lhs, 1. / rhs)


def _ibp_add(
    lhs: bound_propagation.LayerInput,
    rhs: bound_propagation.LayerInput,
) -> IntervalBound:
  """Propagation of IBP bounds through an addition.

  Args:
    lhs: Lefthand side of addition.
    rhs: Righthand side of addition.
  Returns:
    out_bounds: IntervalBound.
  """
  if isinstance(lhs, bound_propagation.Bound):
    if isinstance(rhs, bound_propagation.Bound):
      new_lower = lhs.lower + rhs.lower
      new_upper = lhs.upper + rhs.upper
    else:
      new_lower = lhs.lower + rhs
      new_upper = lhs.upper + rhs
  else:
    # At least one of the inputs is a bound
    new_lower = rhs.lower + lhs
    new_upper = rhs.upper + lhs
  return IntervalBound(new_lower, new_upper)


def _ibp_neg(inp: IntervalBound) -> IntervalBound:
  """Propagation of IBP bounds through a negation.

  Args:
    inp: Bounds that need to be negated.
  Returns:
    out_bounds: IntervalBound
  """
  return IntervalBound(-inp.upper, -inp.lower)


def _ibp_sub(
    lhs: bound_propagation.LayerInput,
    rhs: bound_propagation.LayerInput,
) -> IntervalBound:
  """Propagation of IBP bounds through a substraction.

  Args:
    lhs: Lefthand side of substraction.
    rhs: Righthand side of substraction.
  Returns:
    out_bounds: IntervalBound.
  """
  if isinstance(lhs, bound_propagation.Bound):
    if isinstance(rhs, bound_propagation.Bound):
      return IntervalBound(lhs.lower - rhs.upper, lhs.upper - rhs.lower)
    else:
      return IntervalBound(lhs.lower - rhs, lhs.upper - rhs)
  else:
    # At least one of the inputs is a bound
    return IntervalBound(lhs - rhs.upper, lhs - rhs.lower)


def _ibp_softmax(logits: bound_propagation.LayerInput, axis) -> IntervalBound:
  """Propagation of IBP bounds through softmax.

  Args:
    logits: logits, or their bounds
    axis: the axis or axes along which the softmax should be computed
  Returns:
    out_bounds: softmax output or its bounds
  """
  if isinstance(logits, bound_propagation.Bound):
    # TODO: This bound of the softmax is not as tight as it could be
    # because in the normalization term it uses all upper rather than all except
    # 1 upper.
    log_z_ub = jax.nn.logsumexp(logits.upper, keepdims=True, axis=axis)
    out_lb = jnp.exp(logits.lower - log_z_ub)

    log_z_lb = jax.nn.logsumexp(logits.lower, keepdims=True, axis=axis)
    out_ub = jnp.exp(jnp.minimum(logits.upper - log_z_lb, 0))

    return IntervalBound(out_lb, out_ub)
  else:
    # If the inputs are not bounds, return softmax of logits
    return jax.nn.softmax(logits, axis=axis)


def _ibp_unimodal_0min(
    fun: Callable[[Tensor], Tensor],
    x: IntervalBound
) -> IntervalBound:
  """Propagation of IBP bounds through unimodal function whose min is 0.

  Args:
    fun: Elementwise function which is decreasing before 0 and increasing after,
      achieving its minimum value is at 0.
    x: Bounds on the inputs to the function.
  Returns:
    Bounds on the output of the function.
  """
  lower_out = fun(x.lower)
  upper_out = fun(x.upper)

  out_lb = jnp.where(jnp.logical_and(x.upper >= 0., x.lower <= 0.),
                     fun(jnp.zeros_like(x.lower)),
                     jnp.minimum(lower_out, upper_out))
  out_ub = jnp.maximum(lower_out, upper_out)
  return IntervalBound(out_lb, out_ub)


def _ibp_unimodal_0max(
    fun: Callable[[Tensor], Tensor],
    x: IntervalBound
) -> IntervalBound:
  """Propagation of IBP bounds through unimodal function whose max is 0.

  Args:
    fun: Elementwise function which is increasing before 0 and decreasing after,
      achieving its maximum value is at 0.
    x: Bounds on the inputs to the function.
  Returns:
    Bounds on the output of the function.
  """
  lower_out = fun(x.lower)
  upper_out = fun(x.upper)

  out_lb = jnp.minimum(lower_out, upper_out)
  out_ub = jnp.where(jnp.logical_and(x.upper >= 0., x.lower <= 0.),
                     fun(jnp.zeros_like(x.lower)),
                     jnp.maximum(lower_out, upper_out))
  return IntervalBound(out_lb, out_ub)


def _ibp_leaky_relu(x: IntervalBound,
                    negative_slope: Union[float, Tensor]) -> IntervalBound:
  """Propagation of IBP bounds through leaky Relu.

  Considers the case where the negative slope is negative.

  Args:
    x: Bounds on the inputs to the leaky ReLU.
    negative_slope: Slope for negative inputs.
  Returns:
    out_bounds: Bounds on the output of the leaky ReLU.
  """
  sigma_l = jax.nn.leaky_relu(x.lower, negative_slope)
  sigma_u = jax.nn.leaky_relu(x.upper, negative_slope)
  l_sigma = jnp.where(
      jnp.logical_and(x.lower < 0., x.upper > 0.),
      jnp.minimum(jnp.minimum(sigma_l, sigma_u), 0.),
      jnp.minimum(sigma_l, sigma_u))
  u_sigma = jnp.maximum(sigma_l, sigma_u)
  return IntervalBound(l_sigma, u_sigma)


def _ibp_integer_pow(x: bound_propagation.LayerInput, y: int) -> IntervalBound:
  """Propagation of IBP bounds through integer_pow.

  Args:
    x: Argument be raised to a power, element-wise
    y: fixed integer exponent

  Returns:
    out_bounds: integer_pow output or its bounds.
  """
  if y < 0:
    raise NotImplementedError
  l_pow = lax.integer_pow(x.lower, y)
  u_pow = lax.integer_pow(x.upper, y)

  if y % 2 == 0:
    # Even powers
    contains_zero = jnp.logical_and(
        jnp.less_equal(x.lower, 0), jnp.greater_equal(x.upper, 0))
    lower = jnp.where(contains_zero, jnp.zeros_like(x.lower),
                      jnp.minimum(l_pow, u_pow))
    upper = jnp.maximum(l_pow, u_pow)
    return IntervalBound(lower, upper)
  else:
    # Odd powers
    return IntervalBound(l_pow, u_pow)


def _ibp_linear(*args, **kwargs) -> IntervalBound:
  """Propagation of IBP bounds through a linear primitive treated as a blackbox.

  Args:
    *args: All inputs to the linear operation, which can be either Tensor or
      bounds.
    **kwargs: Parameters of the primitive, which would include the subgraph that
      explains how to implement them.
  Returns:
    Bound on the output of the linear primitive.
  """
  primitive = synthetic_primitives.linear_p
  apply_fun = utils.bind_nonbound_args(primitive.bind, *args, **kwargs)
  bound_args = [arg for arg in args if isinstance(arg, bound_propagation.Bound)]

  all_args = tuple(range(len(bound_args)))
  jac_fn = jax.jacfwd(apply_fun, argnums=all_args)

  zero_inp_args = [jnp.zeros(b_arg.shape) for b_arg in bound_args]
  offset = apply_fun(*zero_inp_args)
  jacobians = jac_fn(*zero_inp_args)

  pos_jacs = jax.tree_map(lambda x: jnp.maximum(x, 0.), jacobians)
  neg_jacs = jax.tree_map(lambda x: jnp.minimum(x, 0.), jacobians)

  new_lower = offset
  new_upper = offset
  for p_jac, n_jac, b_arg in zip(pos_jacs, neg_jacs, bound_args):
    sum_dims = tuple(range(-len(b_arg.shape), 0))
    new_lower += (p_jac * b_arg.lower + n_jac * b_arg.upper).sum(sum_dims)
    new_upper += (p_jac * b_arg.upper + n_jac * b_arg.lower).sum(sum_dims)

  return IntervalBound(new_lower, new_upper)


def _ibp_reciprocal(x: bound_propagation.LayerInput) -> IntervalBound:
  """Propagation of IBP bounds through reciprocal, assuming positive input.

  Args:
    x: Argument to get the inverse of.
  Returns:
    out_bounds: Reciprocal of the bounds.
  """
  return IntervalBound(1. / jax.nn.relu(x.upper), 1. / jax.nn.relu(x.lower))


_input_transform = lambda x: IntervalBound(x.lower, x.upper)

# Define the mapping from jaxpr primitive to the IBP version.
_primitives_to_pass_through = [
    *bound_propagation.RESHAPE_PRIMITIVES,
    lax.reduce_sum_p,
    lax.max_p,
    lax.scatter_add_p,
    lax.exp_p,
    lax.sinh_p,
    lax.log_p,
    lax.tanh_p,
    synthetic_primitives.softplus_p,
    synthetic_primitives.relu_p,
    synthetic_primitives.sigmoid_p,
    lax.sqrt_p,
    lax.sign_p,
]
_primitive_transform: Mapping[
    Primitive,
    ArgsKwargsCallable[
        graph_traversal.LayerInput[IntervalBound], IntervalBound],
] = {
    **{primitive: _make_ibp_passthrough_primitive(primitive)
       for primitive in _primitives_to_pass_through},
    **{primitive: functools.partial(_ibp_bilinear, primitive)
       for primitive in bound_propagation.BILINEAR_PRIMITIVES},
    lax.abs_p: functools.partial(_ibp_unimodal_0min, lax.abs),
    lax.add_p: _ibp_add,
    lax.cosh_p: functools.partial(_ibp_unimodal_0min, lax.cosh),
    lax.sub_p: _ibp_sub,
    lax.neg_p: _ibp_neg,
    lax.div_p: _ibp_div,
    lax.integer_pow_p: _ibp_integer_pow,
    synthetic_primitives.sech_p: functools.partial(
        _ibp_unimodal_0max, synthetic_primitives.sech_p.bind),
    synthetic_primitives.leaky_relu_p: _ibp_leaky_relu,
    synthetic_primitives.parametric_leaky_relu_p: _ibp_leaky_relu,
    synthetic_primitives.softmax_p: _ibp_softmax,
    synthetic_primitives.posreciprocal_p: _ibp_reciprocal,
}
bound_transform = graph_traversal.OpwiseGraphTransform(
    _input_transform, _primitive_transform)

fused_linear_ibp_transform = graph_traversal.OpwiseGraphTransform(
    _input_transform,
    _primitive_transform | {synthetic_primitives.linear_p: _ibp_linear})


def interval_bound_propagation(function, *bounds, fused_linear=False):
  """Performs IBP as described in https://arxiv.org/abs/1810.12715.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBounds, bounds on the inputs of the function.
    fused_linear: Boolean indicating whether to treat sequence of linear
      operations as a single operation, by materializing the equivalent weights.
      This has the potential to be tighter, but may be less efficient.
  Returns:
    output_bound: Bounds on the output of the function obtained by IBP
  """
  transform = fused_linear_ibp_transform if fused_linear else bound_transform
  output_bound, _ = bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(transform),
      function, *bounds)
  return output_bound
