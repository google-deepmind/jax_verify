# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

"""Define convex relaxations for primitives."""
import dataclasses
import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import mccormick
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils


Tensor = bound_propagation.Tensor
TensorFunction = Callable[..., Tensor]
Bound = bound_propagation.Bound
Primitive = bound_propagation.Primitive

def intersection_relaxation(
    piecewise_linear_relaxation_fn: Callable[..., Tuple[
        Sequence[TensorFunction], Sequence[TensorFunction]]],
    *inputs: Bound,
    **params,
) -> Tuple[TensorFunction, TensorFunction]:
  """Relaxation based on intersection of piecewise linear components.

  Args:
    piecewise_linear_relaxation_fn: Accepts input bounds and params;
      returns lists of lower and upper bounding linear functions.
    *inputs: Bound on the inputs to the function.
    **params: Keyword parameters of the primitive.
  Returns:
    lb_fun, ub_fun
  """
  lb_funs, ub_funs = piecewise_linear_relaxation_fn(*inputs, **params)

  def lower_bound(*x):
    return functools.reduce(jnp.maximum, [lb_fun(*x) for lb_fun in lb_funs])

  def upper_bound(*x):
    return functools.reduce(jnp.minimum, [ub_fun(*x) for ub_fun in ub_funs])

  return lower_bound, upper_bound


def posbilinear_piecewise_linear_relaxation(
    inp_x: Bound, inp_y: Bound, **params):
  """Piecewise linear relaxation of a Positive Bilinear primitive.

  This uses pairwise interpolations of the McCormick inequalities.
  For x in [x_l, x_u] and y in [y_l, y_u], the bound imposed are:
    x·y >= x·y_l + x_l·y - x_l·y_l
    x·y >= x·y_u + x_h·y - x_h·y_u
    x·y <= x·y_u + x_l·y - x_l·y_u
    x·y <= x·y_l + x_u·y - x_l·y_u

  Args:
    inp_x: Bounds on the first input.
    inp_y: Bounds on the second input.
    **params: Keywords parameters, notably containing the `jax_verify_subgraph`
      that defines the bilinear primitive.
  Returns:
    lb_funs: Pair of linear lower-bounding functions.
    ub_funs: Pair of linear upper-bounding functions.
  """
  lb_fun0, lb_fun1, ub_fun0, ub_fun1 = (
      mccormick.posbilinear_mccormick_relaxations(
          functools.partial(synthetic_primitives.posbilinear_p.bind, **params),
          inp_x.lower, inp_x.upper, inp_y.lower, inp_y.upper))

  return (lb_fun0, lb_fun1), (ub_fun0, ub_fun1)


def fused_relu_relaxation(linear_out: Bound,
                          *linear_inps: Union[Bound, Tensor],
                          **params):
  """Performs the relaxation of a Fused ReLU primitive.

  This is based on the algorithm described in:
  " The Convex Relaxation Barrier, Revisited: Tightened Single-Neuron
    Relaxations for Neural Network Verification "

  Args:
    linear_out: Output of a linear primitive, input to the ReLU.
    *linear_inps: Inputs to the linear layer that produced linear_out.
    **params: Params of the Fused Relu operation, mainly the jaxpr defining it.
  Returns:
    lb_fun, ub_fun
  """
  del linear_out
  # Check that we can handle the primitive that we have been given.
  subgraph = params['jax_verify_subgraph']
  lin_eqn = params['jax_verify_fusedlinear']
  # Ensure that we have the expected structure.
  assert len(subgraph.eqns) == 1
  relu_eqn = subgraph.eqns[0]
  assert relu_eqn.primitive is synthetic_primitives.relu_p
  assert relu_eqn.invars[0] is lin_eqn.outvars[0]

  # Get the linear part isolated
  bound_args = [(i, arg) for i, arg in enumerate(linear_inps)
                if isinstance(arg, Bound)]
  # Ensure that we have a single bound input.
  assert len(bound_args) == 1
  bound_arg_index, bound_arg = bound_args[0]

  # The input to the ReLU is always going to be an intermediate value, by
  # construction, so the inputs to the lin_eqn are the same as the input
  # to the fused_relu.
  bound_linear_fun = utils.bind_nonbound_args(lin_eqn.primitive.bind,
                                              *linear_inps, **lin_eqn.params)
  def flat_bound_linear_fun(flat_inp):
    inp = jnp.reshape(flat_inp, bound_arg.shape)
    out = bound_linear_fun(inp)
    return jnp.ravel(out)

  zero_bound_inp = jnp.zeros(bound_arg.shape)
  flat_zero_bound_inp = jnp.ravel(zero_bound_inp)
  flat_lbs = jnp.ravel(bound_arg.lower)
  flat_ubs = jnp.ravel(bound_arg.upper)
  flat_lin_weight = jax.jacrev(flat_bound_linear_fun)(flat_zero_bound_inp)
  lin_bias = bound_linear_fun(zero_bound_inp)
  flat_lin_bias = jnp.ravel(lin_bias)

  def flat_concave_upper_bound(x):
    # Let's define l_cup and u_cup.
    # l_cup contains the bound value that each neuron should be at when we want
    # to achieve the lowest possible value after the linear layer.
    # u_cup contains the value when we want to achieve the highest values.
    l_cup = jnp.where(flat_lin_weight >= 0,
                      jnp.expand_dims(flat_lbs, 0),
                      jnp.expand_dims(flat_ubs, 0))
    u_cup = jnp.expand_dims(flat_lbs + flat_ubs, 0) - l_cup

    brd_x = jnp.expand_dims(x, 0)

    # We define the value L(I) to be
    # sum_{i in I} w_i l_cup(i) + sum_{i not in I} w_i u_cup(i) + b
    # This corresponds to the value at a specific vertex of the input domain to
    # the fused relu.
    # Computing an upper bound requires to find a set I such that:
    #      L(I) >= 0
    # and  L(I + {h}) < 0

    # The corresponding bound is given by:
    # (a)  sum_{i in I} w_i (x_i - l_cup(i))
    # (b)  + L(I) * (x_h - l_cup(h)) / (u_cup(h) - l_cup(h))

    # We compute scores that indicate to us in which order we are going to add
    # elements to the set I. This is based on (x - l_cup) / (u_cup - l_cup), but
    # we are also putting all the elements with weights = 0 at the end.
    # We also need to be careful for the cases where u_cup is equal to l_cup, in
    # which case the input has no impact on L(I), on score(h) or on the bound.
    # Proposition 1-2 in the paper explain why this is the right order to get
    # the tightest bound.
    tied_bounds = jnp.abs(u_cup - l_cup) < 1e-6
    safe_denom = jnp.where(tied_bounds, 1., u_cup - l_cup)
    scores = (brd_x - l_cup) / safe_denom
    no_zw_scores = jnp.where((flat_lin_weight == 0.) | tied_bounds,
                             jnp.inf, scores)
    # We will sort according to (x - l_cup) / (u_cup - l_cup)
    score_order = jnp.argsort(no_zw_scores, axis=1)

    # We are going to compute the L(I) as a function of the number of elements
    # that we have added into the I set so far, based on the order given by
    # `score_order`.
    # ini_li is equivalent to computing the upper bound with IBP. This
    # corresponds to an empty set I.
    ini_li = jnp.sum(u_cup * flat_lin_weight, axis=1) + flat_lin_bias
    # li_inc is how much you change l_i by adding a given variable to the I set.
    # It is necessarily negative, (because we switch from a positive
    # contribution to a negative contribution.)
    li_inc = flat_lin_weight * (l_cup - u_cup)

    li_inc_sorted = jnp.take_along_axis(li_inc, score_order, axis=1)
    li_end = jnp.expand_dims(ini_li, axis=1) + jnp.cumsum(li_inc_sorted, axis=1)
    # This is L(I) from which we are progressively going to take. We need to
    # remove the contribution so that index=0 actually corresponds to the set I
    # with 0 elements.
    li = li_end - li_inc_sorted

    # h represents the index of the first element for which we have L(I) that is
    # negative.
    h = (li > 0.).sum(axis=1, keepdims=True)
    # iopt is the index just before h, the last index for which L(I) is > 0.
    i_opt = h - 1

    # Let's now compute the corresponding bound, starting with part (a)
    w_xml = flat_lin_weight * (brd_x - l_cup)
    sorted_wxml = jnp.take_along_axis(w_xml, score_order, axis=1)
    # Similarly as for the L(I) computation, we remove the increment from the
    # cumsum so that we represent correctly the case where I is empty
    acced_sorted_wxml = jnp.cumsum(sorted_wxml, axis=1) - sorted_wxml
    a_part = jnp.squeeze(jnp.take_along_axis(acced_sorted_wxml, i_opt, axis=1),
                         1)

    li_opt = jnp.squeeze(jnp.take_along_axis(li, i_opt, axis=1), 1)
    scores_sorted = jnp.take_along_axis(no_zw_scores, score_order, axis=1)
    score_h = jnp.squeeze(jnp.take_along_axis(scores_sorted, i_opt, axis=1), 1)
    b_part = li_opt * score_h

    relaxation_upper_bound = a_part + b_part

    # The relaxation computed so far is only valid if the ReLU is ambiguous.
    # In the other cases, we now exactly the values of the output bound.
    ibp_ub = ini_li
    ibp_lb = jnp.sum(l_cup * flat_lin_weight, axis=1) + flat_lin_bias

    upper_bound = jnp.where(ibp_lb > 0., flat_bound_linear_fun(x),
                            jnp.where(ibp_ub < 0., 0., relaxation_upper_bound))

    return upper_bound

  def ub_fun(linear_out, *linear_inps):
    del linear_out
    # Find the input corresponding to the bound input.
    bound_inp = linear_inps[bound_arg_index]
    flat_bound_inp = jnp.ravel(bound_inp)
    flat_upper_bound = flat_concave_upper_bound(flat_bound_inp)
    return jnp.reshape(flat_upper_bound, lin_bias.shape)

  lb_fun = functools.partial(synthetic_primitives.fused_relu_p.bind, **params)

  return lb_fun, ub_fun


def convex_fn_relaxation(
    primitive: bound_propagation.Primitive,
    inp: Bound,
    **params) -> Tuple[TensorFunction, TensorFunction]:
  """Relaxation of an element-wise convex primitive.

  Args:
    primitive: Convex primitive to relax.
    inp: Bounds on the input.
    **params: Params of the quadratic operation, mainly the jaxpr defining it.
  Returns:
    lb_fun, ub_fun
  """
  prim_fun = functools.partial(primitive.bind, **params)
  x_lb, x_ub = inp.lower, inp.upper
  y_lb, y_ub = prim_fun(x_lb), prim_fun(x_ub)

  has_interval = x_ub != x_lb
  denom = jnp.where(has_interval, x_ub - x_lb, jnp.ones_like(x_lb))
  chord_slope = jnp.where(
      has_interval, (y_ub - y_lb) / denom, jnp.zeros_like(x_lb))
  chord_intercept = jnp.where(
      has_interval, (y_lb * x_ub - y_ub * x_lb) / denom, y_lb)
  chord_fun = lambda x: chord_slope * x + chord_intercept
  return prim_fun, chord_fun


def relu_piecewise_linear_relaxation(inp: Bound) -> Tuple[
    Tuple[TensorFunction, TensorFunction],
    Tuple[TensorFunction]]:
  """Piecewise linear relaxation of the ReLU function.

  Args:
    inp: Bound on the inputs to the ReLU.
  Returns:
    lb_funs: Pair of linear lower-bounding functions.
    ub_funs: Linear upper-bounding function.
  """
  lb_fun0 = jnp.zeros_like
  lb_fun1 = lambda x: x
  _, chord_fun = convex_fn_relaxation(synthetic_primitives.relu_p, inp)
  return (lb_fun0, lb_fun1), (chord_fun,)


def leaky_relu_piecewise_linear_relaxation(
    inp: Bound, *, negative_slope: float,
) -> Tuple[Sequence[TensorFunction], Sequence[TensorFunction]]:
  """Piecewise linear relaxation of the leaky ReLU.

  Depending on how negative_slope compares to 1.0, the leaky_relu function is
  going to either be convex or concave. The other function is going to be given
  by its chord.

  Args:
    inp: Bounds on the leaky ReLU input.
    negative_slope: Slope for negative inputs.
  Returns:
    lb_funs: Pair of linear lower-bounding functions (or single function if
      negative_slope > 1).
    ub_funs: Linear upper-bounding function (or pair if negative_slope > 1).
  """
  lr_fun0 = lambda x: negative_slope * x
  lr_fun1 = lambda x: x
  _, chord_fun = convex_fn_relaxation(
      synthetic_primitives.leaky_relu_p, inp, negative_slope=negative_slope)

  if negative_slope > 1.:
    # The leaky ReLu is a concave function
    return (chord_fun,), (lr_fun0, lr_fun1)
  else:
    # The leaky Relu is a convex function
    return (lr_fun0, lr_fun1), (chord_fun,)


def abs_piecewise_linear_relaxation(inp: Bound) -> Tuple[
    Tuple[TensorFunction, TensorFunction],
    Tuple[TensorFunction]]:
  """Piecewise linear relaxation of the abs function.

  Args:
    inp: Bound on the inputs to the Abs.
  Returns:
    lb_funs: Pair of linear lower-bounding functions.
    ub_funs: Linear upper-bounding function.
  """
  lb_fun0 = lambda x: -x
  lb_fun1 = lambda x: x
  _, chord_fun = convex_fn_relaxation(lax.abs_p, inp)
  return (lb_fun0, lb_fun1), (chord_fun,)


def _find_sigmoid_upper_bound_tangent(range_lb: Tensor, range_ub: Tensor,
                                      tol: float) -> Tensor:
  """Search the point where the concave hull of the sigmoid stops being linear.

  The concave upper bound of the sigmoid can be several things:
    - It can be the sigmoid itself (if the interval considered is in R+)
    - It can be linear (If the upper bound is small enough. This is a bit more
      general that just if the interval is in R-)
    - It can start linear and at some tangent point become sigmoid.

  This functions searches for the tangent point.
  For the other cases, another function would have narrowed the search range
  such that range_lb = range_ub and we early exit from the loop.

  This is a combination of a binary search and of the Newton method.

  Args:
    range_lb: Lower bound of the domain on which to define the convex hull.
    range_ub: Upper bound of the domain on which to define the convex hull.
    tol: Tolerance criterion for convergence
  Returns:
    final_t: Tangent point at which the concave upper bound of the sigmoid
      should go from linear to sigmoid. If range_lb == range_ub, that number
      should be returned.
  """
  flat_range_lb = jnp.reshape(range_lb, (-1,))
  flat_range_ub = jnp.reshape(range_ub, (-1,))

  fun = jax.nn.sigmoid
  dfun = lambda x: fun(x) * (1 - fun(x))

  # The point that we are looking for is the point where:
  #  dfun(x) = (fun(x) - fun(lb)) / (x - lb)
  to_root_fun = lambda x, lb: dfun(x) - (fun(x)-fun(lb))/jnp.maximum(x-lb, tol)
  to_root_val_and_grad = jax.vmap(jax.value_and_grad(to_root_fun))

  search_lb = jnp.maximum(flat_range_lb, 0.)
  # In the case where l is very large (in the negative), we can have an
  # approximate solution. We can use this to shrink the search space, making the
  # binary search converge significantly faster.
  # If lb<-1e3, fun(lb)=0, so we can just solve:
  # dfun(x) = fun(x) / (x - lb).
  # <=> fun(X) (1 - fun(x)) = fun(x) / (x - lb)
  # <=> 1 - fun(x) = 1 / (x - lb)
  # <=> exp(-x) / (1 + exp(-x)) = 1 / (x - lb)
  # <=> exp(-x) * (x - lb - 1) = 1
  # <=> exp(x) - x = -lb -1
  # And we can assume that for large value, exp(x)-x ~ exp(x)
  # So we know that the optimal t is going to be close to log(-lb-1)
  # We add some padding (+1) around that value to make sure we are not excluding
  # a valid solution from the search space.
  upper_bound_for_large_l = jnp.where(
      flat_range_lb < -1e3,
      jnp.log(jnp.maximum(-flat_range_lb -1, 1.)) + 1,
      float('inf'))
  search_ub = jnp.minimum(flat_range_ub, upper_bound_for_large_l)
  t_k = 0.5 * (search_lb + search_ub)
  it = jnp.array(0)

  def body_fun(loop_args):
    it, t, lb, ub = loop_args
    new_it = it + 1
    f, df = to_root_val_and_grad(t, flat_range_lb)
    new_lb = jnp.where(f >= 0., jnp.maximum(lb, t), lb)
    new_ub = jnp.where(f <= 0., jnp.minimum(ub, t), ub)
    newton_t = t - f / df
    out_of_bounds_t = (newton_t <= new_lb) | (newton_t >= new_ub)
    new_t = jnp.where((jnp.abs(df) <= tol) | out_of_bounds_t,
                      0.5 * (new_lb + new_ub),
                      newton_t)
    return new_it, new_t, new_lb, new_ub

  def continue_search(loop_args):
    it, t, lb, ub = loop_args

    # Points that have not converged have both
    #   - high value on the difference between average slope and sig derivative
    #   - high value on the gap between upper bound and lower bound
    # If any one of this criterion is not satisfied, the point has converged.
    not_converged = ((jnp.abs(to_root_fun(t, flat_range_lb)) >= tol) &
                     (jnp.abs(ub - lb) >= tol))
    # We keep searching as long as:
    #   - we don't exceed 100 iterations
    #   - There is at least 1 point that has not converged.
    return jnp.logical_and(it <= 100, jnp.any(not_converged))

  _, final_t, _, _ = jax.lax.while_loop(
      continue_search, body_fun, (it, t_k, search_lb, search_ub))
  final_t = jnp.reshape(final_t, range_lb.shape)
  return final_t


def _find_upperbound_sigmoid_linear_cutoff(lbs: Tensor, ubs: Tensor, tol: float
                                           ) -> Tensor:
  """Find the point where the sigmoid concave upper bound stops being linear.

  This function restricts the search space to a single point for the cases where
  the concave upper bound is simply the sigmoid or is fully linear. It then
  calls the binary search function.

  Args:
    lbs: Lower bound of the domain on which to define the convex hull.
    ubs: Upper bound of the domain on which to define the convex hull.
    tol: Tolerance for numerical operations
  Returns:
    tang_point: Tangent point at which the concave upper bound of the sigmoid
      should go from linear to sigmoid.
  """
  sigmoid = jax.nn.sigmoid
  dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

  dsig_ub = dsigmoid(ubs)
  avg_slope = (sigmoid(ubs) - sigmoid(lbs)) / jnp.maximum(ubs - lbs, tol)

  t_lb = jnp.where((lbs <= 0.) & (dsig_ub >= avg_slope), ubs, lbs)
  t_ub = jnp.where(lbs >= 0, lbs, ubs)
  return _find_sigmoid_upper_bound_tangent(t_lb, t_ub, tol)


def sigmoid_relaxation(inp: Bound,
                       tol: float = 1e-6
                       )->Tuple[TensorFunction, TensorFunction]:
  """Perform the relaxation of the sigmoid.

  See the supplementary materials of https://arxiv.org/pdf/2002.10410.pdf for
  the derivation.

  Args:
    inp: Bound on the inputs to the Sigmoid
    tol: Tolerance criterion
  Returns:
    lb_fun, ub_fun
  """
  sigmoid = jax.nn.sigmoid
  lbs = inp.lower
  ubs = inp.upper
  # Derive the concave upper bound, find where the cutoff is between the linear
  # part and the sigmoid part
  up_tangent_point = _find_upperbound_sigmoid_linear_cutoff(lbs, ubs, tol)
  up_lin_slope = (sigmoid(up_tangent_point) - sigmoid(lbs)) / (
      jnp.maximum(up_tangent_point - lbs, tol))
  up_lin_offset = sigmoid(up_tangent_point) - up_lin_slope * up_tangent_point

  def sigmoid_upper_concave(x):
    return jnp.where(jnp.greater_equal(x, up_tangent_point),
                     sigmoid(x), up_lin_slope * x + up_lin_offset)

  # Derive the convex lower bound, find there the cutoff is between the sigmoid
  # part and the linear part. By symmetry, we can reuse the upper bound code.
  neg_low_tangent_point = _find_upperbound_sigmoid_linear_cutoff(
      -ubs, -lbs, tol)
  low_tang_point = -neg_low_tangent_point
  low_lin_slope = (sigmoid(ubs) - sigmoid(low_tang_point)) / (
      jnp.maximum(ubs - low_tang_point, tol))
  low_lin_offset = sigmoid(low_tang_point) - low_lin_slope * low_tang_point

  def sigmoid_lower_convex(x):
    return jnp.where(jnp.less_equal(x, low_tang_point),
                     sigmoid(x), low_lin_slope * x + low_lin_offset)
  return sigmoid_lower_convex, sigmoid_upper_concave


def posreciprocal_relaxation(
    inp: Bound,
) -> Tuple[TensorFunction, TensorFunction]:
  """Relaxation of reciprocal, on strictly positive inputs.

  The (unchecked) assumption is that inputs are always positive, and 1/x is
  therefore convex.

  Args:
    inp: Bounds on the input.
  Returns:
    lb_fun, ub_fun
  """
  safe_inp = bound_propagation.IntervalBound(
      utils.safe_pos(inp.lower), utils.safe_pos(inp.upper))
  return convex_fn_relaxation(synthetic_primitives.posreciprocal_p, safe_inp)


@dataclasses.dataclass
class ActivationRelaxation:
  """Activation function traits, including convex relaxation.

  Attributes:
    relaxation_fn: Accepts input bounds and params; returns convex lower bound
      and concave upper bound.
    piecewise_linear_relaxation_fn: Optional; accepts input bounds and params;
      returns lists of lower and upper bounding linear functions.
    pos_neg_linear: Whether this positive and negative parts of this
      1D activation function are both affine, e.g. ReLU, sign.
    convex: Whether the activation function is convex.
    eltwise_increasing: Whether the activation function is known to be
      element-wise monotonically increasing.
  """

  relaxation_fn: Callable[..., Tuple[TensorFunction, TensorFunction]]
  piecewise_linear_relaxation_fn: Optional[Callable[..., Tuple[
      Sequence[TensorFunction], Sequence[TensorFunction]]]] = None
  pos_neg_linear: bool = False
  convex: bool = False
  eltwise_increasing: bool = False


relaxation_fns = {
    synthetic_primitives.relu_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation, relu_piecewise_linear_relaxation),
        piecewise_linear_relaxation_fn=relu_piecewise_linear_relaxation,
        pos_neg_linear=True, convex=True, eltwise_increasing=True),
    synthetic_primitives.leaky_relu_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation, leaky_relu_piecewise_linear_relaxation),
        piecewise_linear_relaxation_fn=leaky_relu_piecewise_linear_relaxation,
        pos_neg_linear=True),
    lax.abs_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation, abs_piecewise_linear_relaxation),
        piecewise_linear_relaxation_fn=abs_piecewise_linear_relaxation,
        pos_neg_linear=True, convex=True),
    synthetic_primitives.softplus_p: ActivationRelaxation(
        functools.partial(
            convex_fn_relaxation, synthetic_primitives.softplus_p),
        convex=True, eltwise_increasing=True),
    lax.exp_p: ActivationRelaxation(
        functools.partial(convex_fn_relaxation, lax.exp_p),
        convex=True, eltwise_increasing=True),
    synthetic_primitives.posreciprocal_p: ActivationRelaxation(
        posreciprocal_relaxation, convex=True),
    synthetic_primitives.sigmoid_p: ActivationRelaxation(
        sigmoid_relaxation, eltwise_increasing=True),
    synthetic_primitives.posbilinear_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation, posbilinear_piecewise_linear_relaxation),
        piecewise_linear_relaxation_fn=posbilinear_piecewise_linear_relaxation),
    synthetic_primitives.fused_relu_p: ActivationRelaxation(
        fused_relu_relaxation),
}
