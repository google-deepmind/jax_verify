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

"""Define convex relaxations for primitives."""
import dataclasses
import functools
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import mccormick
from jax_verify.src import opt_utils
from jax_verify.src import simplex_bound
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils
from jax_verify.src.types import Primitive, Tensor, TensorFun  # pylint: disable=g-multiple-import
import typing_extensions


def posbilinear_piecewise_linear_relaxation(
    inp_x: bound_propagation.Bound, inp_y: bound_propagation.Bound, **params):
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


def fused_relu_relaxation(linear_out: bound_propagation.Bound,
                          *linear_inps: bound_propagation.LayerInput,
                          **params):
  """Performs the relaxation of a Fused ReLU primitive.

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
                if isinstance(arg, bound_propagation.Bound)]
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

  lb_fun = functools.partial(synthetic_primitives.fused_relu_p.bind, **params)

  if True:
    # Define the relaxation of the Fused ReLU over a hypercube - interval bound.
    single_neuron_ub_fun = alt_fused_relu_hypercube_upper_bound(
        flat_lbs, flat_ubs
    )

  flat_param_ub_fun = jax.vmap(single_neuron_ub_fun, in_axes=(0, 0, None))
  flat_ub_fun = functools.partial(flat_param_ub_fun, flat_lin_weight,
                                  flat_lin_bias)

  def ub_fun(linear_out, *linear_inps):
    del linear_out
    # Find the input corresponding to the bound input to the linear.
    bound_inp = linear_inps[bound_arg_index]
    flat_bound_inp = jnp.ravel(bound_inp)
    flat_upper_bound = flat_ub_fun(flat_bound_inp)
    return jnp.reshape(flat_upper_bound, lin_bias.shape)

  return lb_fun, ub_fun


def fused_relu_ubfun(
    bound: graph_traversal.InputBound
) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
  """Get the function upper bounding a ReLU still parameterized by z."""
  if True:
    _, solve_given_z = fused_relu_hypercube_upper_bound(
        bound.lower.flatten(), bound.upper.flatten())
  return solve_given_z


def alt_fused_relu_hypercube_upper_bound(
    flat_lbs: Tensor, flat_ubs: Tensor
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
  """Performs the relaxation of the FusedReLU over an hypercube.

  This is based on the algorithm described in:
  " The Convex Relaxation Barrier, Revisited: Tightened Single-Neuron
    Relaxations for Neural Network Verification "

  Args:
    flat_lbs: Tensor of flattened lower bounds.
    flat_ubs: Tensor of flattened upper bounds.
  Returns:
    ub_fun: Function computing the concave upper bound of a fused relu when
      given the coefficient of a linear function, the bias, and the point at
      which to evaluate the upper bound.
  """
  def ub_fun(coeffs: Tensor, bias: float, x: Tensor) -> float:
    # Let's define l_cup and u_cup.
    # l_cup contains the bound value that each neuron should be at when we want
    # to achieve the lowest possible value after the linear layer.
    # u_cup contains the value when we want to achieve the highest values.
    l_cup = jnp.where(coeffs >= 0, flat_lbs, flat_ubs)
    u_cup = flat_lbs + flat_ubs - l_cup

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
    scores = (x - l_cup) / safe_denom
    no_zw_scores = jnp.where((coeffs == 0.) | tied_bounds, jnp.inf, scores)

    # We are going to compute the L(I) as a function of the number of elements
    # that we have added into the I set so far, based on the order given by
    # `score_order`.
    # ini_li is equivalent to computing the upper bound with IBP. This
    # corresponds to an empty set I.
    ini_li = jnp.sum(u_cup * coeffs) + bias

    # li_inc is how much you change l_i by adding a given variable to the I set.
    # It is necessarily negative, (because we switch from a positive
    # contribution to a negative contribution.)
    li_inc = coeffs * (l_cup - u_cup)
    # w_xml is the contribution of a given variable to the evaluated bound if it
    # is in the I set.
    w_xml = coeffs * (x - l_cup)

    # Compute all the reorganized arrays together to avoid having to do costly
    # take_along_axis operations.
    scores_sorted, li_inc_sorted, wxml_sorted = jax.lax.sort(
        [no_zw_scores, li_inc, w_xml], num_keys=1)
    li_end = ini_li + jnp.cumsum(li_inc_sorted)

    # This is L(I) from which we are progressively going to take. We need to
    # remove the contribution so that index=0 actually corresponds to the set I
    # with 0 elements.
    li = li_end - li_inc_sorted

    # iopt is the index just before h, the last index for which L(I) is > 0.
    # As L(I) is strictly decreasing, this will be a one-hot vector.
    i_opt_mask = jnp.diff(li <= 0., append=jnp.array([True]))

    # Similarly as for the L(I) computation, we remove the increment from the
    # cumsum so that we represent correctly the case where I is empty
    acced_sorted_wxml = jnp.cumsum(wxml_sorted) - wxml_sorted

    # Let's now compute the corresponding bound, as described above. We will
    # use the i_opt_mask to select the correct values.
    a_part = acced_sorted_wxml
    b_part = li * scores_sorted
    relaxation_upper_bound = ((a_part + b_part) * i_opt_mask).sum()

    # The relaxation computed so far is only valid if the ReLU is ambiguous.
    # In the other cases, we now exactly the values of the output bound.
    ibp_ub = ini_li
    ibp_lb = jnp.sum(l_cup * coeffs) + bias

    pass_through = jnp.sum(x * coeffs) + bias
    upper_bound = jnp.where(ibp_lb > 0., pass_through,
                            jnp.where(ibp_ub < 0., 0., relaxation_upper_bound))
    return upper_bound
  return ub_fun  # pytype: disable=bad-return-type  # jax-ndarray


def fused_relu_hypercube_upper_bound(
    flat_lbs: Tensor, flat_ubs: Tensor
) -> Tuple[Callable[[Tensor, Tensor, Tensor], Tensor],
           Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]]:
  """Performs the relaxation of the FusedReLU over an hypercube.

  This corresponds to the upper bound obtained based on the Sharp formulation,
  as described in Equation (1) of the "Bounds for Verified Compressed Sensing"
  document.

  We return two upper bounds because there exists two different settings:
   - If the fused ReLU is defined only over a hypercube, we can use the first
     bound as we know how to efficiently maximize over z.
   - If the fused ReLU is defined over a product of domains, the problem is
     more complex. We need to optimize z based on the maximization produced by
     *all* the domains, so we need to return the un-optimized (over z) bound,
     so that we can construct the overall bound function.

  Args:
    flat_lbs: Tensor of flattened lower bounds.
    flat_ubs: Tensor of flattened upper bounds.

  Returns:
    ub_fun: Concave upper bound operating on the flattened input to the
      FusedReLU.
    solve_given_z: Upper bound over the output of the ReLU which is still
      dependent on z.
  """
  bound_gap = flat_ubs - flat_lbs
  def solve_given_z(coeffs: Tensor, bias: float, x: Tensor, z: float):
    alpha_equal_w_case = (coeffs * x
                          + (1 - z) * (jnp.maximum(-coeffs, 0) * bound_gap
                                       - coeffs * flat_lbs))
    alpha_equal_0_case = z * (jax.nn.relu(coeffs) * bound_gap
                              + coeffs * flat_lbs)
    return jnp.minimum(alpha_equal_w_case, alpha_equal_0_case).sum() + bias * z

  eval_all_z = jax.vmap(solve_given_z, in_axes=(None, None, None, 0))

  def single_neuron_ub_fun(coeffs, bias, x):
    ibp_upper = (jnp.maximum(coeffs, 0) * flat_ubs
                 + jnp.minimum(coeffs, 0) * flat_lbs)
    ibp_lower = (jnp.maximum(coeffs, 0) * flat_lbs
                 + jnp.minimum(coeffs, 0) * flat_ubs)
    ibp_bound_gap = ibp_upper - ibp_lower
    safe_gap = jnp.where(ibp_bound_gap != 0., ibp_bound_gap, 1.)
    possible_z = (coeffs * x - ibp_lower) / safe_gap
    possible_z = jnp.concatenate((possible_z, jnp.array([0., 1.])))
    vals_all_z = eval_all_z(coeffs, bias, x, possible_z)
    return jnp.max(vals_all_z)

  return single_neuron_ub_fun, solve_given_z  # pytype: disable=bad-return-type  # jax-ndarray


def convex_fn_relaxation(
    primitive: Primitive,
    *args: Union[bound_propagation.Bound, Tensor],
    **params) -> Tuple[TensorFun, TensorFun]:
  """Relaxation of an element-wise convex primitive.

  Args:
    primitive: Convex primitive to relax.
    *args: Inputs to the convex function: bounds on the input, unnamed params.
    **params: Params of the convex operation, mainly the jaxpr defining it.
  Returns:
    lb_fun, ub_fun
  """
  prim_fun = utils.bind_nonbound_args(primitive.bind, *args, **params)
  inp, = [arg for arg in args if isinstance(arg, bound_propagation.Bound)]
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


def relu_piecewise_linear_relaxation(inp: bound_propagation.Bound) -> Tuple[
    Tuple[TensorFun, TensorFun],
    Tuple[TensorFun]]:
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
    inp: bound_propagation.Bound, *, negative_slope: float,
) -> Tuple[Sequence[TensorFun], Sequence[TensorFun]]:
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


def cvx_parametric_leaky_relu_pl_relaxation(
    inp: bound_propagation.Bound, negative_slope: Tensor,
) -> Tuple[Sequence[TensorFun], Sequence[TensorFun]]:
  """Piecewise linear relaxation of convex (slope < 1) parametric leaky ReLU.

  NOTE: the relaxation is valid only if the function is convex:
    jnp.all(negative_slope < 1.)

  TODO: could be adapted to work for all prelus by returning two lower
  and upper bounding functions in any case. This would require handling in other
  places where piecewise_linear_relaxation_fn is employed
  (e.g., mip_solver/relaxation.py)

  Args:
    inp: Bounds on the leaky ReLU input.
    negative_slope: Slope for negative inputs.
  Returns:
    lb_funs: Pair of linear lower-bounding functions.
    ub_funs: Linear upper-bounding function.
  """
  # All returned functions accept two inputs to comply with the primitive.
  # The second input (negative_slope) is not used as it's passed in this scope.
  lr_fun0 = lambda x, y: negative_slope * x
  lr_fun1 = lambda x, y: x
  _, chord_fun = convex_fn_relaxation(
      synthetic_primitives.parametric_leaky_relu_p, inp, negative_slope)
  # NOTE: assumes the leaky Relu is a convex function (negative_slope < 1)
  return (lr_fun0, lr_fun1), (lambda x, y: chord_fun(x),)


def clip_piecewise_linear_relaxation(
    inp: bound_propagation.Bound,
    a_min: float, a_max: float,
) -> Tuple[Sequence[TensorFun], Sequence[TensorFun]]:
  """Piecewise linear relaxation of the Clipping function.

  Args:
    inp: Bounds on the clipped input.
    a_min: Minimum value imposed on the output.
    a_max: Maximum value imposed on the output.
  Returns:
    lb_funs: Pair of linear lower-bounding functions.
    ub_funs: Pair of linear upper-bounding functions.
  """
  clip_fun = functools.partial(jnp.clip, a_min=a_min, a_max=a_max)

  x_lb, x_ub = inp.lower, inp.upper
  y_lb, y_ub = clip_fun(inp.lower), clip_fun(inp.upper)
  passing_has_interval = y_ub != y_lb

  passing_ub_denom = jnp.where(passing_has_interval,
                               y_ub - x_lb, jnp.ones_like(x_lb))
  passing_ub_slope = jnp.where(passing_has_interval,
                               (y_ub - y_lb) / passing_ub_denom,
                               jnp.zeros_like(x_lb))
  passing_ub_intercept = jnp.where(passing_has_interval,
                                   y_lb - passing_ub_slope * x_lb, y_ub)
  passing_ub = lambda x, *_: passing_ub_slope * x + passing_ub_intercept
  clipped_ub = lambda x, *_: a_max * jnp.ones_like(x_lb)

  passing_lb_denom = jnp.where(passing_has_interval,
                               x_ub - y_lb, jnp.ones_like(x_lb))
  passing_lb_slope = jnp.where(passing_has_interval,
                               (y_ub - y_lb) / passing_lb_denom,
                               jnp.zeros_like(x_lb))
  passing_lb_intercept = jnp.where(passing_has_interval,
                                   y_lb * (1 - passing_lb_slope), y_lb)
  passing_lb = lambda x, *_: passing_lb_slope * x + passing_lb_intercept
  clipped_lb = lambda x, *_: a_min * jnp.ones_like(x_lb)

  return (clipped_lb, passing_lb), (passing_ub, clipped_ub)


def abs_piecewise_linear_relaxation(inp: bound_propagation.Bound) -> Tuple[
    Tuple[TensorFun, TensorFun],
    Tuple[TensorFun]]:
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


def _find_s_shape_upper_bound_tangent(
    fun: TensorFun, dfun: TensorFun, approx_tang_pt: TensorFun,
    range_lb: Tensor, range_ub: Tensor,
    tol: float) -> Tensor:
  """Search the point where the concave hull of s-shape fun stops being linear.

  The concave upper bound of an s-shape function can be several things:
    - It can be the function itself (if the interval considered is in R+)
    - It can be linear (If the upper bound is small enough. This is a bit more
      general that just if the interval is in R-)
    - It can start linear and at some tangent point become the function.

  This functions searches for the tangent point.
  For the other cases, another function would have narrowed the search range
  such that range_lb = range_ub and we early exit from the loop.

  This is a combination of a binary search and of the Newton method.

  Args:
    fun: Function for which we are trying to find the cutoff.
    dfun: Derivative of the function for which we are trying to find the cutoff.
    approx_tang_pt: Approximate solution of the cutoff when the lower bound is
      large.
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

  # The point that we are looking for is the point where:
  #  dfun(x) = (fun(x) - fun(lb)) / (x - lb)
  to_root_fun = lambda x, lb: dfun(x) - (fun(x)-fun(lb))/jnp.maximum(x-lb, tol)
  to_root_val_and_grad = jax.vmap(jax.value_and_grad(to_root_fun))

  search_lb = jnp.maximum(flat_range_lb, 0.)
  search_ub = jnp.maximum(flat_range_ub, 0.)

  upper_bound_for_large_l = jnp.where(flat_range_lb < -1e3,
                                      approx_tang_pt(flat_range_lb),
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
                     ((ub - lb) >= tol))
    # We keep searching as long as:
    #   - we don't exceed 100 iterations
    #   - There is at least 1 point that has not converged.
    return jnp.logical_and(it <= 100, jnp.any(not_converged))

  _, final_t, _, _ = jax.lax.while_loop(
      continue_search, body_fun, (it, t_k, search_lb, search_ub))
  final_t = jax.lax.stop_gradient(final_t)
  # The search that we implemented is only valid when we are looking for the
  # tangent point in R^+. In case we have called the search function with
  # negative values, we should recover those.
  final_t = jnp.clip(final_t, flat_range_lb, flat_range_ub)
  final_t = jnp.reshape(final_t, range_lb.shape)
  return final_t


def _find_upperbound_s_shape_linear_cutoff(
    fun: TensorFun, dfun: TensorFun, approx_tang_pt: TensorFun,
    lbs: Tensor, ubs: Tensor, tol: float
) -> Tensor:
  """Find the point where the s-shape concave upper bound stops being linear.

  This function restricts the search space to a single point for the cases where
  the concave upper bound is simply the function or is fully linear. It then
  calls the binary search function.

  Args:
    fun: Function for which we are trying to find the cutoff.
    dfun: Derivative of the function for which we are trying to find the cutoff.
    approx_tang_pt: Approximate solution of the cutoff when the lower bound is
    lbs: Lower bound of the domain on which to define the convex hull.
    ubs: Upper bound of the domain on which to define the convex hull.
    tol: Tolerance for numerical operations
  Returns:
    linear_cutoff_pt: Tangent point at which the concave upper bound of the
      s-shaped function should go from linear to s-shaped function.
  """
  dfun_ub = dfun(ubs)
  avg_slope = (fun(ubs) - fun(lbs)) / jnp.maximum(ubs - lbs, tol)

  t_lb = jnp.where((lbs <= 0.) & (dfun_ub >= avg_slope), ubs, lbs)
  t_ub = jnp.where(lbs >= 0, lbs, ubs)

  binary_search_fun = functools.partial(_find_s_shape_upper_bound_tangent,
                                        fun, dfun, approx_tang_pt, tol=tol)
  short_circuit_fun = lambda l, u: l
  linear_cutoff_pt = jax.lax.cond(jnp.all(t_lb == t_ub),
                                  short_circuit_fun, binary_search_fun,
                                  t_lb, t_ub)
  return linear_cutoff_pt


def s_shape_relaxation(
    fun: TensorFun,
    dfun: TensorFun,
    approx_tang_pt: TensorFun,
    inp: bound_propagation.Bound,
    tol: float = 1e-6,
) -> Tuple[TensorFun, TensorFun]:
  """Perform the relaxation of an S-shape function.

  See the supplementary materials of https://arxiv.org/pdf/2002.10410.pdf for
  the derivation.
  The tricky part is to determine when does the function concave upper bound (
  respectively convex lower bound) switch from being linear to being the
  function itself. We solve the problem of finding where the cutoff point is
  between those two parts.

  Args:
    fun: Function to get the convex hull of, assumed to be s-shape.
      What this means is that the function is:
        - Odd (f(-x) = - f(x))
        - Monotonically increasing.
        - The derivative is 0. at -inf and +inf.
        - The function is convex over R^- and concave over R^+
    dfun: Derivative of the function.
    approx_tang_pt: Upper bound on the position of the tangent point when the
      lower bound of the domain is very large.
    inp: Bound on the inputs to the S-shape function.
    tol: Tolerance criterion
  Returns:
    lb_fun, ub_fun
  """

  lbs = inp.lower
  ubs = inp.upper
  # Derive the concave upper bound, find where the cutoff is between the linear
  # part and the s-shaped part
  up_tangent_point = _find_upperbound_s_shape_linear_cutoff(
      fun, dfun, approx_tang_pt, lbs, ubs, tol)
  up_lin_slope = (fun(up_tangent_point) - fun(lbs)) / (
      jnp.maximum(up_tangent_point - lbs, tol))
  up_lin_offset = fun(up_tangent_point) - up_lin_slope * up_tangent_point

  def s_shape_upper_concave(x):
    return jnp.where(jnp.greater_equal(x, up_tangent_point),
                     fun(x), up_lin_slope * x + up_lin_offset)

  # Derive the convex lower bound, find there the cutoff is between the s-shaped
  # part and the linear part. By symmetry, we can reuse the upper bound code.
  neg_low_tangent_point = _find_upperbound_s_shape_linear_cutoff(
      fun, dfun, approx_tang_pt, -ubs, -lbs, tol)
  low_tang_point = -neg_low_tangent_point
  low_lin_slope = (fun(ubs) - fun(low_tang_point)) / (
      jnp.maximum(ubs - low_tang_point, tol))
  low_lin_offset = fun(low_tang_point) - low_lin_slope * low_tang_point

  def s_shape_lower_convex(x):
    return jnp.where(jnp.less_equal(x, low_tang_point),
                     fun(x), low_lin_slope * x + low_lin_offset)
  return s_shape_lower_convex, s_shape_upper_concave


def sigmoid_relaxation(inp: bound_propagation.Bound,
                       tol: float = 1e-6
                       )->Tuple[TensorFun, TensorFun]:
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
  dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

  # In the case where l is very large (in the negative), we can have an
  # approximate solution of the tangent point. We can use this to shrink the
  # search space, making the binary search converge significantly faster.
  # If lb<-1e3, fun(lb)=0, so we can just solve:
  # dfun(x) = fun(x) / (x - lb).
  # <=> fun(X) (1 - fun(x)) = fun(x) / (x - lb)
  # <=> 1 - fun(x) = 1 / (x - lb)
  # <=> exp(-x) / (1 + exp(-x)) = 1 / (x - lb)
  # <=> exp(-x) * (x - lb - 1) = 1
  # <=> exp(x) - x = -lb -1
  # Given that we know x >=0 for a tangent point, we have exp(x) < - lb - 1
  # Therefore the solution is upper bounded by log(-lb-1)
  # We add some padding (+1) around that value to make sure we are not excluding
  # a valid solution from the search space.
  approx_sig_tang = lambda lb: jnp.log(jnp.maximum(-lb - 1., 1.)) + 1.
  return s_shape_relaxation(sigmoid, dsigmoid, approx_sig_tang, inp, tol)


def tanh_relaxation(
    inp: bound_propagation.Bound,
    tol: float = 1e-6,
) -> Tuple[TensorFun, TensorFun]:
  """Perform the relaxation of the hyperbolic tangent.

  This is an implementation modeled on the convex hull of the sigmoid.

  Args:
    inp: Bound on the inputs to the hyperbolic tangent.
    tol: Tolerance criterion
  Returns:
    lb_fun, ub_fun
  """
  tanh = jax.nn.tanh
  dtanh = lambda x: 1 - tanh(x)**2

  # In the case where l is very large, in the negative, we can have an
  # approximate solution.
  # If lb <-1e3, fun(lb) = -1
  # dfun(x) = (fun(x) + 1) / (x - lb)
  # <=> 1 - fun(x)**2 = (fun(x) + 1) / (x - lb)
  # <=> (1 - fun(x)) * (1 + fun(x)) = (fun(x) + 1) / (x - lb)
  # <=> (1 - fun(x)) = 1 / (x - lb)
  # <=> 2 / (exp(2x) + 1) = 1 / (x - lb)
  # <=> exp(2x) - 2x = -2*lb - 1
  # We know that for the tangent point we are searching, x >= 0,
  # so we know that for our solution exp(2x) <= -2 * lb - 1
  # Therefore, we know that the solution is upperbounded by 0.5 * log(-2*lb -1.)
  # We'll add some padding to ensure we are not excluding any valid solution.
  approx_tanh_tang = lambda lb: 0.5*jnp.log(jnp.maximum(-2*lb-1., 1.)) + 1.
  return s_shape_relaxation(tanh, dtanh, approx_tanh_tang, inp, tol)


def posreciprocal_relaxation(
    inp: bound_propagation.Bound,
) -> Tuple[TensorFun, TensorFun]:
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


class RelaxationFn(typing_extensions.Protocol):

  def __call__(
      self,
      *inputs: bound_propagation.Bound,
      **params,
  ) -> Tuple[TensorFun, TensorFun]:
    """Convex relation.

    Args:
      *inputs: Bound on the inputs to the function.
      **params: Keyword parameters of the primitive.

    Returns:
      Convex lower bound and concave upper bound.
    """


class PiecewiseLinearRelaxationFn(typing_extensions.Protocol):

  def __call__(
      self,
      *inputs: bound_propagation.Bound,
      **params,
  ) -> Tuple[Sequence[TensorFun], Sequence[TensorFun]]:
    """Piecewise linear convex relation.

    Args:
      *inputs: Bound on the inputs to the function.
      **params: Keyword parameters of the primitive.

    Returns:
      Lists of lower and upper bounding linear functions.
    """


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

  relaxation_fn: RelaxationFn
  piecewise_linear_relaxation_fn: Optional[PiecewiseLinearRelaxationFn] = None
  pos_neg_linear: bool = False
  convex: bool = False
  eltwise_increasing: bool = False


def intersection_relaxation(
    piecewise_linear_relaxation_fn: PiecewiseLinearRelaxationFn,
    *inputs: Union[bound_propagation.Bound, Tensor],
    **params,
) -> Tuple[TensorFun, TensorFun]:
  """Relaxation based on intersection of piecewise linear components.

  Args:
    piecewise_linear_relaxation_fn: Accepts input bounds and params;
      returns lists of lower and upper bounding linear functions.
    *inputs: Inputs to the convex function: bounds on the input, unnamed params.
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


relaxation_fns: Mapping[Primitive, ActivationRelaxation] = {
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
    # NOTE: only convex (negative_slope < 1.) parametric ReLUs are supported
    synthetic_primitives.parametric_leaky_relu_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation,
            cvx_parametric_leaky_relu_pl_relaxation),
        piecewise_linear_relaxation_fn=cvx_parametric_leaky_relu_pl_relaxation,
        pos_neg_linear=True),
    synthetic_primitives.clip_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation, clip_piecewise_linear_relaxation),
        piecewise_linear_relaxation_fn=clip_piecewise_linear_relaxation,
        eltwise_increasing=True),
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
    lax.tanh_p: ActivationRelaxation(
        tanh_relaxation, eltwise_increasing=True),
    synthetic_primitives.posbilinear_p: ActivationRelaxation(
        functools.partial(
            intersection_relaxation, posbilinear_piecewise_linear_relaxation),
        piecewise_linear_relaxation_fn=posbilinear_piecewise_linear_relaxation),
    synthetic_primitives.fused_relu_p: ActivationRelaxation(
        fused_relu_relaxation),
}
