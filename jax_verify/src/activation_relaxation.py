# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
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
import functools
from typing import Callable, Tuple

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import mccormick
from jax_verify.src import synthetic_primitives


Tensor = bound_propagation.Tensor
TensorFunction = Callable[..., Tensor]
Bound = bound_propagation.Bound

def posbilinear_relaxation(inp_a: Bound, inp_b: Bound, **params):
  """Perform the relaxation of a Positive Bilinear primitive.

  Args:
    inp_a: Bounds on the first input.
    inp_b: Bounds on the second input.
    **params: Keywords arguments, notably containing the `jax_verify_subgraph`
      that defines the bilinear primitive.
  Returns:
    lb_fun, ub_fun
  """
  inp_a_lb, inp_a_ub = inp_a.lower, inp_a.upper
  inp_b_lb, inp_b_ub = inp_b.lower, inp_b.upper

  def posbilinear_fun(tensor_a, tensor_b):
    return synthetic_primitives.posbilinear_p.bind(tensor_a, tensor_b, **params)

  def lb_fun(inp_a, inp_b):
    low_a = (posbilinear_fun(inp_a, inp_b_lb) +
             posbilinear_fun(inp_a_lb, inp_b) -
             posbilinear_fun(inp_a_lb, inp_b_lb))
    low_b = (posbilinear_fun(inp_a, inp_b_ub) +
             posbilinear_fun(inp_a_ub, inp_b) -
             posbilinear_fun(inp_a_ub, inp_b_ub))
    return jnp.maximum(low_a, low_b)

  def ub_fun(inp_a, inp_b):
    up_a = (posbilinear_fun(inp_a, inp_b_lb) +
            posbilinear_fun(inp_a_ub, inp_b) -
            posbilinear_fun(inp_a_ub, inp_b_lb))
    up_b = (posbilinear_fun(inp_a, inp_b_ub) +
            posbilinear_fun(inp_a_lb, inp_b) -
            posbilinear_fun(inp_a_lb, inp_b_ub))
    return jnp.minimum(up_a, up_b)

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

  chord_slope_safe_denom = jnp.maximum(x_ub - x_lb, 1e-12)
  chord_slope = (y_ub - y_lb) / chord_slope_safe_denom
  chord_intercept = y_lb - chord_slope * x_lb
  chord_fun = lambda x: chord_slope * x + chord_intercept
  return prim_fun, chord_fun


def leaky_relu_relaxation(
    inp: Bound, *, negative_slope: float,
) -> Tuple[TensorFunction, TensorFunction]:
  """Perform the relaxation of the leaky ReLU.

  Depending on how negative_slope compares to 1.0, the leaky_relu function is
  going to either be convex or concave. The other function is going to be given
  by its chord.

  Args:
    inp: Bounds on the leaky ReLU input.
    negative_slope: Slope for negative inputs.
  Returns:
    lb_fun, ub_fun.
  """
  lr, chord_fun = convex_fn_relaxation(
      synthetic_primitives.leaky_relu_p, inp, negative_slope=negative_slope)

  if negative_slope > 1.:
    # The leaky ReLu is a concave function
    return chord_fun, lr
  else:
    # The leaky Relu is a convex function
    return lr, chord_fun


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
