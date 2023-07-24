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

"""Routines to solve basic optimization problems."""
from typing import Callable, Tuple
import jax
from jax import numpy as jnp

from jax_verify.src import utils
from jax_verify.src.types import Tensor


def greedy_assign(upper: Tensor, total_sum: float):
  """Perform a greedy assignment respecting an upper bound constraint.

  Args:
    upper: Maximum value to assign to each coordinate.
    total_sum: Total value to assign to the coordinates.
  Returns:
    A tensor of greedy assignment.
  """
  return jnp.clip(total_sum - (jnp.cumsum(upper) - upper),
                  a_min=0., a_max=upper)


def sorted_knapsack(lower: Tensor, upper: Tensor, total_sum: float,
                    backward: bool = False):
  """Perform a fractional knapsack assuming coordinates sorted by weights.

  Args:
    lower: Smallest allowable value on each of the coordinates.
    upper: Largest allowable value on each of the coordinates.
    total_sum: Total sum that need to be achieved.
    backward: Whether to perform the greedy assignment from the start of the
      tensor or from the end of it.
  Returns:
    Assignment to the variable of the knapsack problem.
  """
  assignment_budget = upper - lower
  if backward:
    # If we want to do the assignment backward, we will follow the strategy of
    # initially putting all the values at their upper bound, and then greeedily
    # removing from the beginning of the Tensor until the budget is met, which
    # would give the same result as if we were assigning in order from the back.
    return upper - greedy_assign(assignment_budget, upper.sum() - total_sum)  # pytype: disable=wrong-arg-types  # jax-ndarray
  else:
    return lower + greedy_assign(assignment_budget, total_sum - lower.sum())  # pytype: disable=wrong-arg-types  # jax-ndarray


def fractional_exact_knapsack(weights: Tensor, simplex_sum: float,
                              lower_bound: Tensor, upper_bound: Tensor
                              ) -> float:
  """Solve the fractional knapsack problem.

      max          sum_i w_i x_i
      such that    l_i <= x_i <= u_i   forall i
                   sum_i x_i = simplex_sum

  Note that this is not exactly the classic fractional knapsack problem which
  would have had the constraint sum_i x_i <= simplex_sum. This causes a
  difference in the case where negative weights might be present, in which case
  the results would differ.

  The problem is solved using a greedy algorithm based on sorting the weights
  rather than using a weighted-median type algorithm.

  @TODO Ensure some error checking. What if the simplex sum is too high
  (greater than the sum of upper bounds) or too low (smaller than sum of lower
  bounds?).

  Args:
    weights: w_i, the linear coefficients on the coordinates.
    simplex_sum: The total value that the sum of all constraints need to sum to.
    lower_bound: Lower bound on the coordinates.
    upper_bound: Upper bound on the coordinates.
  Returns:
    val: the value of the optimum solution of the optimization problem.
  """
  flat_lower = jnp.reshape(lower_bound, (-1,))
  flat_upper = jnp.reshape(upper_bound, (-1,))
  flat_weights = jnp.reshape(weights, (-1,))

  sorted_weights, sorted_lower, sorted_upper = jax.lax.sort(
      (flat_weights, flat_lower, flat_upper), num_keys=1)

  sorted_assignment = sorted_knapsack(sorted_lower, sorted_upper, simplex_sum,
                                      backward=True)
  return (sorted_weights * sorted_assignment).sum()  # pytype: disable=bad-return-type  # jax-types


def concave_1d_max(
    obj_fn: Callable[[Tensor], Tensor],
    x_lb: Tensor,
    x_ub: Tensor,
    *,
    num_steps: int = 30,
) -> Tuple[Tensor, Tensor]:
  """Maximises the given element-wise function using golden ratio search.

  Args:
    obj_fn: Function to be maximised. Its inputs and outputs must have the
      same shape, and its computation is assumed to be independent across
      its elements.
    x_lb: Lower bounds on the inputs to `obj_fn`.
    x_ub: Upper bounds on the inputs to `obj_fn`.
    num_steps: Number of optimisation steps. Every five steps give approximately
      one additional decimal digit of accuracy of `y`.
  Returns:
    x: Tensor of same shape as inputs to `obj_fn` containing the argmax.
    y: Tensor of same shape as inputs/outputs of `obj_fn` containing its max.
  """
  phi = 0.61803398875  # inverse golden ratio

  def loop_init():
    # Initialise with four abcissae spaced at intervals  phi^2 : phi^3 : phi^2 .
    xs = [
        x_lb,
        phi * x_lb + (1.-phi) * x_ub,
        (1.-phi) * x_lb + phi * x_ub,
        x_ub,
    ]
    ys = [
        None,
        obj_fn(xs[1]),
        obj_fn(xs[2]),
        None,
    ]
    return xs, ys

  def loop_step(_, val):
    xs, ys = val
    # Insert two new abcissae, so we have
    # xs[0] ... left_x ... xs[1] ... xs[2] ... right_x ... xs[3]
    # spaced at intervals  phi^3 : phi^4 : phi^3 : phi^4 : phi^3 .
    left_x = phi * xs[0] + (1.-phi) * xs[2]
    right_x = (1.-phi) * xs[1] + phi * xs[3]
    # Select either leftmost or rightmost four abcissae, whichever contains
    # the maximum.
    select = ys[1] > ys[2]
    xs = [
        jnp.where(select, xs[0], xs[1]),
        jnp.where(select, left_x, xs[2]),
        jnp.where(select, xs[1], right_x),
        jnp.where(select, xs[2], xs[3]),
    ]
    ys = [
        None,
        jnp.where(select, obj_fn(left_x), ys[2]),
        jnp.where(select, ys[1], obj_fn(right_x)),
        None,
    ]
    return xs, ys

  xs, _ = utils.fori_loop_no_backprop(0, num_steps, loop_step, loop_init())

  # Extract optimal (argmax) x without gradients.
  x = jax.lax.stop_gradient((xs[1] + xs[2]) / 2.)

  # Re-evaluate at max with gradients.
  return x, obj_fn(x)


def project_onto_interval_simplex(lower: Tensor, upper: Tensor,
                                  simplex_sum: float, point: Tensor) -> Tensor:
  """Solve the projection on the intersection of simplex and interval domains.

  The problem being solved is:

    x_opt = argmin_x  0.5 || x - x_0 ||^2
              s.t    sum(x) = s
                     l <= x <= u

  By dualising the simplex constraint and looking at the lagrangian, we can show
  that x_opt = clip(x_0 - mu, l, u)
  where mu is the lagrangian variable associated with the simplex constraint.

  We are looking for which mu results in the constraint sum(x) = s to be
  satisfied.

  sum(x_opt) is going to be a 1D piecewise linear function. We are going to find
  its root by identifying which of the linear pieces cross zero, and then using
  linear interpolation to find where the crossing happen.

  Args:
    lower: Lower bound on the admissible values. (l in equations)
    upper: Upper bound on the admissible values. (u in equations)
    simplex_sum: Value that all coordinates need to sum to (s in equations)
    point: Point that we are trying to project (x_0 in equations)

  Returns:
    x_opt: Projection of `point` onto the set of constraints.
  """

  flat_lb = jnp.reshape(lower, (-1,))
  flat_ub = jnp.reshape(upper, (-1,))
  flat_pt = jnp.reshape(point, (-1,))
  # We're also considering fake breakpoints outside of the actual ones, to avoid
  # some numerical errors in the case where the breakpoints would be exactly at
  # some upper bounds / lower bounds.
  out_lower = flat_pt.min() - flat_ub.max() - 1.0
  out_upper = flat_pt.max() - flat_lb.min() + 1.0
  all_breakpoints = jax.lax.concatenate(
      (flat_pt - flat_lb,
       flat_pt - flat_ub,
       jnp.array([out_lower, out_upper])), dimension=0)
  sorted_breakpoints = jax.lax.sort(all_breakpoints)

  brd_break = jnp.expand_dims(sorted_breakpoints, 1)
  brd_lb = jnp.expand_dims(flat_lb, 0)
  brd_ub = jnp.expand_dims(flat_ub, 0)
  brd_point = jnp.expand_dims(flat_pt, 0)

  terms = jnp.clip(brd_point - brd_break, a_min=brd_lb, a_max=brd_ub)
  fun_val = terms.sum(axis=1) - simplex_sum

  dy = jnp.diff(fun_val)
  dx = jnp.diff(sorted_breakpoints)

  safe_dy = jnp.where(dy != 0, dy, 1.)
  # We need to use a safe version of dy to avoid creating NaNs when dy==0.
  # The replacement of the values by 1. does not cause any problems for the
  # computation of the opt_mu coefficient later.
  interp = sorted_breakpoints[:-1] - fun_val[:-1] * dx / safe_dy
  # interp gets multiplied by a mask of jnp.diff(fun_val >= 0.), so for
  # incorrect value of interp, if dy == 0, then it means that fun_val was
  # unchanged, so the jnp.diff would be set to False and mask out the
  # incorrect value.
  opt_mu = (interp * jnp.diff(fun_val >= 0.)).sum()

  flat_x_opt = jnp.clip(flat_pt - opt_mu, a_min=flat_lb, a_max=flat_ub)
  return jnp.reshape(flat_x_opt, lower.shape)
