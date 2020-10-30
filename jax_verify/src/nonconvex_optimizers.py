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

"""Provides optimizers to use with the NonConvexBound in `nonconvex.py`.
"""

import abc
import functools
import math
from typing import Tuple

import jax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src import nonconvex
from jax_verify.src import nonconvex_duals
import numpy as np
import optax

Tensor = jnp.ndarray


class BoundOptimizer(metaclass=abc.ABCMeta):
  """Abstract Class to define the API of optimizer.

  Each subclass defines an optimization algorithm to solve the optimization
  problem defined by a NonConvexBound. This is done by overloading the
  `optimize` function.
  """

  @abc.abstractmethod
  def optimize(self, non_convex_bound: nonconvex.NonConvexBound):
    pass


def _chunked_optimization(bound_shape, max_parallel_nodes, optimize_chunk):
  """Perform optimization of the target in chunks.

  Args:
    bound_shape: Shape of the bound to compute
    max_parallel_nodes: How many activations to optimize at once. If =0, perform
      optimize all the nodes simultaneously.
    optimize_chunk: Function to optimize a chunk and return updated bounds.
  Returns:
    bounds: Optimized bounds.
  """
  ini_lbs = jnp.zeros(bound_shape, jnp.float32)
  ini_ubs = jnp.zeros(bound_shape, jnp.float32)
  if max_parallel_nodes == 0:
    lbs, ubs = optimize_chunk(0, (ini_lbs, ini_ubs))
  else:
    nb_opt_chunk = math.ceil(np.prod(bound_shape[1:]) / max_parallel_nodes)
    lbs, ubs = jax.lax.fori_loop(0, nb_opt_chunk, optimize_chunk,
                                 (ini_lbs, ini_ubs))
  bounds = ibp.IntervalBound(lbs, ubs)
  return bounds


def _create_opt_problems(non_convex_bound: nonconvex.NonConvexBound,
                         batch_index: int,
                         nb_parallel_nodes: int):
  """Define the objective function and the necessary variables shape.

  Iteratively yields the objectives to minimize in order to limit memory usage.

  Args:
    non_convex_bound: Bound for which to create the optimization problems.
    batch_index: Index of the optimization batch to generate.
    nb_parallel_nodes: How large should the optimization batches be. If 0,
      optimize all problems at once.
  Returns:
    var_to_opt: shapes of the variables to optimize to compute the bounds.
    obj: objectives to minimize.
  """
  # Create the objective matrix
  output_shape = non_convex_bound.shape
  total_nb_nodes_to_opt = int(np.prod(output_shape[1:]))

  start_node = batch_index * nb_parallel_nodes
  if nb_parallel_nodes == 0:
    nb_nodes_to_opt = total_nb_nodes_to_opt
  else:
    nb_nodes_to_opt = nb_parallel_nodes

  # In order to be able to use the function in the while loop, we have to have
  # all tensors remain the same size so we're going to always create a tensor
  # of the same size, but will not necessarily fill all the rows.
  flat_lb_obj = jnp.zeros((nb_nodes_to_opt, total_nb_nodes_to_opt))
  opt_idx = jnp.arange(nb_nodes_to_opt)
  node_idx = jnp.minimum(start_node + opt_idx, total_nb_nodes_to_opt-1)
  to_add = ((start_node + opt_idx) < total_nb_nodes_to_opt).astype(jnp.float32)
  flat_lb_obj = jax.ops.index_add(flat_lb_obj, (opt_idx, node_idx), to_add,
                                  indices_are_sorted=True,
                                  unique_indices=False)
  lb_obj = jnp.reshape(flat_lb_obj, (nb_nodes_to_opt, *output_shape[1:]))
  # Get the objective for the upper bounds.
  ub_obj = -lb_obj
  per_sp_obj = jnp.concatenate([lb_obj, ub_obj], axis=0)
  # Make the objective for all the samples in the batch
  obj = jnp.repeat(jnp.expand_dims(per_sp_obj, 0), output_shape[0], axis=0)

  # Generate the shape of the variables necessary to solve the problem
  var_to_opt = {}
  for pos, var_shape in non_convex_bound.variables.items():
    var_to_opt[pos] = (var_shape[0], 2 * nb_nodes_to_opt) + var_shape[1:]

  return var_to_opt, obj


def _unpack_opt_problem(dual_vals: Tensor, obj: Tensor)->Tuple[Tensor, Tensor]:
  """Extract the lower bounds and upper bounds from the result of optmization.

  Assumes that the tensor obj is in two part along axis=1 with the first part
  computing upper bounds and the second part computing lower bounds. We also
  assume that the value given by dual_vals match the objective given in `obj`.

  Args:
    dual_vals: Value of the dual returned by the optimization process.
    obj: Linear functions that were optimized for.
  Returns:
    lb: Tensor containing lower bounds were they were computed and 0 elsewhere.
    ub: Tensor containing upper bounds were they were computed and 0 elsewhere.
  """
  lb_obj, ub_obj = jnp.split(obj, 2, axis=1)
  lb_duals, ub_duals = jnp.split(dual_vals, 2, axis=1)

  lb = jnp.einsum('bo,bo...->b...', lb_duals, lb_obj)
  ub = jnp.einsum('bo,bo...->b...', ub_duals, ub_obj)

  return lb, ub


def _pgd_step(current, grad, step_size):
  """Do a projected gradient step with the given step size."""
  new_varset = {}
  for key, var in current.items():
    var_grad = grad[key]
    nb_act_dims = len(var.shape) - len(step_size.shape)
    broad_step_size = jnp.reshape(step_size,
                                  step_size.shape + (1,)*nb_act_dims)
    new_varset[key] = jnp.clip(var + broad_step_size * var_grad, 0., 1.)
  return new_varset


class LinesearchFistaOptimizer(BoundOptimizer):
  """FISTA with line search.

  As done in the "An efficient nonconvex reformulation of stagewise convex
  optimization problems" NeurIPS2020 submission. This is a reimplementation
  of the code at:
  l/d/r/r_v/verification/ibp/verification/nonconvex_optimizable_bounds.py

  The difference between the two versions of the code is that this
  implementation performs minimization while the other one performed
  maximization. This difference is visible in the following places:
    - Changes the formula of the quadratic approximation (sign before
      the L2 norm term).
    - Direction of the comparison for the line search.
    - Direction of the step.
    - Direction of the dual gap (when minimizing, it is primal - dual, while
      it is dual - primal when maximizing) to check for convergence.
  """

  def __init__(self,
               num_steps: int,
               max_step_size: float = 100.0,
               min_step_size: float = 1e-5,
               beta_l: float = 0.5,
               beta_h: float = 1.5,
               check_convergence_every: int = 1,
               check_relative_dual_gap: bool = False,
               termination_dual_gap: float = 1e-2,
               max_parallel_nodes: int = 0):
    self._num_steps = num_steps
    self._max_step_size = max_step_size
    self._min_step_size = min_step_size
    self._beta_l = beta_l
    self._beta_h = beta_h
    self._check_convergence_every = check_convergence_every
    self._check_relative_dual_gap = check_relative_dual_gap
    self._termination_dual_gap = termination_dual_gap
    self._max_parallel_nodes = max_parallel_nodes

  def optimize(self, non_convex_bound: nonconvex.NonConvexBound
               )->bound_propagation.Bound:
    ## Define the functions for the backtracking line search
    # We have a separate optimization per optimization target (so one per batch
    # element times per neuron).
    # This will be performed in jax.lax.while_loop, with the following arguments
    # ls_loop_args:
    #   need_lower: Boolean array indicating for each step_size if we still
    #     needs to lower the step size.
    #   step_size: Array of step size being used.
    #   y_stats: Tuple with y, f(y) and grad(y), so that we don't have to keep
    #     recomputing it.
    #   objectives: Coefficients of the objective functions.
    def quad_approx(x, y, grad_y, step_size):
      quad_approx = 0
      for key, x_var in x.items():
        y_var = y[key]
        grady_var = grad_y[key]
        dims_to_reduce = tuple(range(2, y_var.ndim))
        quad_approx = quad_approx + (
            ((x_var - y_var)*grady_var).sum(axis=dims_to_reduce)
            + 0.5 / step_size * ((x_var - y_var)**2).sum(axis=dims_to_reduce))
      return quad_approx

    def should_decrease(step_size, y_stats, objectives):
      y, f_y, grad_y = y_stats
      new_x = _pgd_step(y, grad_y, -step_size)
      val_newx, _ = non_convex_bound.primal_fn(new_x, objectives)
      val_qapprox = f_y + quad_approx(new_x, y, grad_y, step_size)
      per_sp_insufficient_progress = (val_newx >= val_qapprox)
      step_size_not_min = step_size > self._min_step_size
      return jnp.logical_and(step_size_not_min, per_sp_insufficient_progress)

    def lower_stepsize_if_needed(ls_loop_args):
      """Reduce the step size for all the optimization target that need it.

      Update the check to see if it needs to be reduced further.

      Args:
        ls_loop_args: Line search loop arguments
      Returns:
        new_ls_loop_args: Updated line search loop arguments
      """
      need_lower, step_size, y_stats, objectives = ls_loop_args
      new_step_size = jnp.where(need_lower,
                                self._beta_l * step_size, step_size)
      new_need_lower = should_decrease(new_step_size, y_stats, objectives)
      return (new_need_lower, new_step_size, y_stats, objectives)

    any_need_lower_stepsize = lambda ls_loop_args: ls_loop_args[0].any()

    ## Define the function for the optimization loop
    # Perform the Fista with backtracking line search algorithm, as described
    # in "A Fast Iterative Shrinkage-Thresholding Algorithm", Beck and Teboulle
    # The only change is that we increase the step size by a factor of
    # self._beta_h for step size that didn't need to be reduced at all during
    # the linesearch.
    # This is performed in a jax.lax.while_loop, with the following arguments:
    # opt_loop_args:
    #   it: Iteration counter
    #   x, y: variable set
    #   gamma: float, coefficient used for the momentum (t_k in the paper)
    #   step_size: Array containing the current values of the step size.
    #   objectives: Coefficients of the objective functions.
    # We stop either based on a maximum number of iterations, or based on the
    # convergence between the primal objective and the dual objective, which is
    # checked every self._check_convergence_every iterations.
    def fista_with_linesearch_step(opt_loop_args):
      it, x, y, gamma, step_size, objectives = opt_loop_args
      # Compute f_y and d(f_y)/d(y)
      value_and_gradofsum_fn = jax.value_and_grad(non_convex_bound.primal_sumfn,
                                                  has_aux=True)
      (_, (f_y, _)), grad_y = value_and_gradofsum_fn(y, objectives)

      # Compute the step size to use with a line search
      y_stats = (y, f_y, grad_y)
      ini_need_lower = should_decrease(step_size, y_stats, objectives)
      _, new_step_size, _, _ = jax.lax.while_loop(
          any_need_lower_stepsize,
          lower_stepsize_if_needed,
          (ini_need_lower, step_size, y_stats, objectives))

      # Perform the updates
      new_x = _pgd_step(y, grad_y, -new_step_size)
      new_gamma = 1 + jnp.sqrt(1 + gamma ** 2) / 2
      coeff = (gamma - 1) / new_gamma

      new_y = {}
      for key, new_x_var in new_x.items():
        new_y[key] = new_x_var + coeff * (new_x_var - x[key])

      # Increase the step size of the samples that didn't need reducing.
      new_step_size = jnp.where(ini_need_lower,
                                new_step_size, self._beta_h * new_step_size)

      return it + 1, new_x, new_y, new_gamma, new_step_size, objectives

    def not_all_converged(not_converged_args):
      x, objectives = not_converged_args
      primal, dual = non_convex_bound.dual(x, objectives)
      dgap_value = primal - dual
      if self._check_relative_dual_gap:
        bound_scale = 0.5 * (jnp.abs(primal) + jnp.abs(dual))
        termination_gap = (1 + bound_scale) * self._termination_dual_gap
      else:
        termination_gap = self._termination_dual_gap

      return (dgap_value > termination_gap).any()

    def continue_criterion(opt_loop_args):
      it, x, *_, objectives = opt_loop_args
      not_all_iterations = (it < self._num_steps)
      opt_not_converged = jax.lax.cond(
          (it % self._check_convergence_every) == 0.,
          not_all_converged,
          lambda _: jnp.array(True),
          operand=(x, objectives))
      return jnp.logical_and(opt_not_converged, not_all_iterations)

    ## Define the function to optimize a chunk of the nodes of the activation.
    def optimize_chunk(batch_index: int,
                       current_bounds: Tuple[Tensor, Tensor]
                       ) -> Tuple[Tensor, Tensor]:
      var_shapes, batch_objectives = _create_opt_problems(
          non_convex_bound, batch_index, self._max_parallel_nodes)
      x = {key: 0.5 * jnp.ones(shape) for key, shape in var_shapes.items()}
      y = x
      batch_dims = batch_objectives.shape[:2]
      gamma = jnp.array(0.)
      step_size = self._max_step_size * jnp.ones(batch_dims)
      it = jnp.array(0)

      _, final_x, _, _, _, _ = jax.lax.while_loop(
          continue_criterion,
          fista_with_linesearch_step,
          (it, x, y, gamma, step_size, batch_objectives))

      _, dual_vals = non_convex_bound.dual(jax.lax.stop_gradient(final_x),
                                           batch_objectives)

      batch_lbs, batch_ubs = _unpack_opt_problem(dual_vals, batch_objectives)

      current_lbs, current_ubs = current_bounds

      lbs = batch_lbs + current_lbs
      ubs = batch_ubs + current_ubs

      return (lbs, ubs)

    return _chunked_optimization(non_convex_bound.shape,
                                 self._max_parallel_nodes,
                                 optimize_chunk)


class PGDOptimizer(BoundOptimizer):
  """Projected Gradient Optimizer.

  Optimization can either by taking gradients with respect to the primal or the
  dual objective.

  Passing a number of steps equal to zero will result in the bound derived from
  the initialization.
  """

  def __init__(self, num_steps: int, step_size: float,
               optimize_dual: bool = False,
               max_parallel_nodes: int = 0):
    self._num_steps = num_steps
    self._step_size = step_size
    self._optimize_dual = optimize_dual
    self._max_parallel_nodes = max_parallel_nodes

  def optimize(self, non_convex_bound: nonconvex.NonConvexBound
               )->bound_propagation.Bound:
    # If we are going to actually perform optimization, define the function to
    # minimize (either the primal, or the negative of the dual),
    # its gradient and the projection function to use.
    if self._num_steps:
      if self._optimize_dual:
        def fun_to_opt(opt_vars, objectives):
          _, dual_vals = non_convex_bound.dual(opt_vars, objectives)
          return -jnp.sum(dual_vals)
      else:
        def fun_to_opt(opt_vars, objectives):
          final_acts = non_convex_bound.evaluate(opt_vars)
          obj = jnp.sum(final_acts * objectives)
          return obj
      grad_fun = jax.grad(fun_to_opt)
      proj_fun = lambda x: jnp.clip(x, 0., 1.)

      # Define the optimizer. Because we are minimizing the objective function,
      # we will scale the gradient by a negative step size.
      tx = optax.scale(-self._step_size)

    # Define the function to optimize a chunk of the nodes of the activation.
    def optimize_chunk(batch_index: int,
                       current_bounds: Tuple[Tensor, Tensor]
                       ) -> Tuple[Tensor, Tensor]:
      var_shapes, batch_objectives = _create_opt_problems(
          non_convex_bound, batch_index, self._max_parallel_nodes)

      var_set = {key: 0.5 * jnp.ones(shape)
                 for key, shape in var_shapes.items()}

      # Perform the optimization.
      if self._num_steps:
        state = tx.init(var_set)

        def opt_step(_, state_and_var):
          state, var_set = state_and_var
          grads = grad_fun(var_set, batch_objectives)
          updates, new_state = tx.update(grads, state, var_set)
          unc_var_set = optax.apply_updates(var_set, updates)
          new_var_set = jax.tree_map(proj_fun, unc_var_set)
          return new_state, new_var_set

        _, var_set = jax.lax.fori_loop(0, self._num_steps, opt_step,
                                       (state, var_set))

      # Compute the resulting bound and unpack it.
      _, dual_vals = non_convex_bound.dual(jax.lax.stop_gradient(var_set),
                                           batch_objectives)

      batch_lbs, batch_ubs = _unpack_opt_problem(dual_vals, batch_objectives)

      current_lbs, current_ubs = current_bounds

      lbs = batch_lbs + current_lbs
      ubs = batch_ubs + current_ubs

      return (lbs, ubs)

    return _chunked_optimization(non_convex_bound.shape,
                                 self._max_parallel_nodes,
                                 optimize_chunk)


def _create_nostep_optimizer():
  return nonconvex.OptimizingConcretizer(
      PGDOptimizer(0, 0., optimize_dual=False, max_parallel_nodes=512))

nonconvex_constopt_bound_propagation = functools.partial(
    nonconvex.build_nonconvex_formulation,
    nonconvex_duals.WolfeNonConvexBound, _create_nostep_optimizer)
