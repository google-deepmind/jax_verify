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

"""Provides optimizers to use with the NonConvexBound in `nonconvex.py`.
"""

import abc
import math
from typing import Tuple, Callable, Dict, Optional, List

import jax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src import utils
from jax_verify.src.nonconvex import nonconvex
import numpy as np
import optax


Tensor = jnp.ndarray
Index = bound_propagation.Index
ParamSet = nonconvex.ParamSet
BranchPlane = Tuple[Index, int, float]
BranchConstraint = Tuple[BranchPlane, int]


class BoundOptimizer(metaclass=abc.ABCMeta):
  """Abstract Class to define the API of optimizer.

  Each subclass defines an optimization algorithm to solve the optimization
  problem defined by a NonConvexBound. This is done by overloading the
  `optimize` function.
  """

  @abc.abstractmethod
  def optimize_fun(self, non_convex_bound: nonconvex.NonConvexBound
                   ) -> Callable[[ParamSet, ParamSet], ParamSet]:
    pass


class OptimizingConcretizer(nonconvex.Concretizer):
  """Concretizer based on optimizing the intermediate bounds.

  This needs to be initialized with an optimizer, and concrete bounds will be
  obtained by solving the relaxation for the intermediate activations.
  """

  def __init__(
      self, optimizer: BoundOptimizer,
      max_parallel_nodes: int = 512,
      branching_constraints: Optional[List[BranchConstraint]] = None,
      branching_optimizer: Optional[optax.GradientTransformation] = None,
      branching_opt_number_steps: int = 0):
    self._optimizer = optimizer
    self._max_parallel_nodes = max_parallel_nodes
    self._branching_constraints = []
    self._branching_optimizer = None
    self._branching_opt_number_steps = branching_opt_number_steps

    if branching_constraints is not None:
      self._branching_constraints = branching_constraints
      # Keep the optimizer for the branching constraints dual variables.
      if (branching_optimizer is None) or (branching_opt_number_steps == 0):
        raise ValueError('If branching constraints are imposed, an optimizer '
                         'and a number of optimization steps for the lagrangian'
                         ' variables corresponding to them needs to be '
                         'provided.')
      self._branching_optimizer = branching_optimizer
      self._branching_opt_number_steps = branching_opt_number_steps

  def accept_input(self, *_args, **_kwargs):
    # There is no need to update anything.
    pass

  def accept_primitive(self, *_args, **_kwargs):
    # For now, the optimizer is not dependent on what propagation has been
    # achieved, so just reuse the same.
    # Potentially, in the future, we could handle adapting the hyperparameters.
    pass

  def get_bounds(self, to_opt_bound: nonconvex.NonConvexBound
                 ) -> bound_propagation.Bound:
    optimize_fun = self._optimizer.optimize_fun(to_opt_bound)

    def optimize_chunk(chunk_index: int) -> Tuple[Tensor, Tensor]:
      var_shapes, chunk_objectives = _create_opt_problems(
          to_opt_bound, chunk_index, self._max_parallel_nodes)

      ini_var_set = {key: 0.5 * jnp.ones(shape)
                     for key, shape in var_shapes.items()}

      def solve_problem(objectives: ParamSet) -> Tensor:
        # Optimize the bound for primal variables.
        opt_var_set = optimize_fun(objectives, ini_var_set)
        # Compute the resulting bound
        _, bound_vals = to_opt_bound.dual(jax.lax.stop_gradient(opt_var_set),
                                          objectives)
        return bound_vals

      if any(node_idx <= to_opt_bound.index
             for ((node_idx, *_), _) in self._branching_constraints):
        # There exists constraints that needs to be taken into account.

        # The dual vars per constraint are scalars, but we need to apply them
        # for each of the optimization objective.
        nb_targets = chunk_objectives[to_opt_bound.index].shape[0]
        # Create the dual variables for them.
        active_branching_constraints = [
            (node_idx, neuron_idx, val, side)
            for (node_idx, neuron_idx, val), side in self._branching_constraints
            if node_idx <= to_opt_bound.index
        ]
        nb_constraints = len(active_branching_constraints)
        dual_vars = [jnp.zeros([nb_targets])] * nb_constraints

        # Define the objective function to optimize. The branching constraints
        # are lifted into the objective function.
        def unbranched_objective(dual_vars: ParamSet) -> Tuple[float, Tensor]:
          objectives = chunk_objectives.copy()
          base_term = jnp.zeros([nb_targets])
          for ((node_idx, neuron_idx, val, side),
               branch_dvar) in zip(active_branching_constraints, dual_vars):
            # Adjust the objective function to incorporate the dual variables.
            if node_idx not in objectives:
              objectives[node_idx] = jnp.zeros(var_shapes[node_idx])

            # The branching constraint is encoded as:
            #   side * neuron >= side * val
            # (when side==1, this is neuron >= lb,
            #  and when side==-1, this is -neuron >= -ub )
            # To put in a canonical form \lambda_b() <= 0, this is:
            # \lambda_b() = side * val - side * neuron

            # Lifting the branching constraints takes us from the problem:
            #   min_{z} f(z)
            #   s.t.    \mu_i() <= z_i <= \eta_i()  \forall i
            #           \lambda_b() <= 0            \forall b
            #
            # to
            #   max_{\rho_b} min_{z} f(z) + \rho_b \lambda_b()
            #                s.t \mu_i() <= z_i <= \eta_i()  \forall i
            #   s.t  rho_b >= 0

            # Add the term corresponding to the dual variables to the linear
            # objective function.
            coeff_to_add = -side * branch_dvar
            index_to_update = jax.ops.index[:, neuron_idx]
            flat_node_obj = jnp.reshape(objectives[node_idx], (nb_targets, -1))
            flat_updated_node_obj = jax.ops.index_add(flat_node_obj,
                                                      index_to_update,
                                                      coeff_to_add)
            updated_node_obj = jnp.reshape(flat_updated_node_obj,
                                           var_shapes[node_idx])
            objectives[node_idx] = updated_node_obj

            # Don't forget the terms based on the bound.
            base_term = base_term + (side * val * branch_dvar)

          network_term = solve_problem(objectives)
          bound = network_term + base_term

          return bound.sum(), bound

        def evaluate_bound(ini_dual_vars: List[Tensor]) -> Tensor:
          ini_state = self._branching_optimizer.init(ini_dual_vars)
          eval_and_grad_fun = jax.grad(unbranched_objective, argnums=0,
                                       has_aux=True)

          # The carry consists of:
          # - The best set of dual variables seen so far.
          # - The current set of dual variables.
          # - The best bound obtained so far.
          # - The state of the optimizer.
          # For each of the step, we will:
          # - Evaluate the bounds by the current set of dual variables.
          # - Update the best set of dual variables if progress was achieved.
          # - Do an optimization step on the current set of dual variables.
          # This way, we are guaranteed that we keep track of the dual variables
          # producing the best bound at the end.
          def opt_step(
              carry: Tuple[List[Tensor], List[Tensor],
                           Tensor, optax.OptState], _
          ) -> Tuple[Tuple[List[Tensor], List[Tensor],
                           Tensor, optax.OptState], None]:
            best_lagdual, lagdual, best_bound, state = carry
            # Compute the bound and their gradients.
            lagdual_grads, new_bound = eval_and_grad_fun(lagdual)

            # Update the lagrangian dual variables for the best bound seen.
            improve_best = new_bound > best_bound
            new_best_lagdual = []
            for best_dvar, new_dvar in zip(best_lagdual, lagdual):
              new_best_lagdual.append(jnp.where(improve_best,
                                                new_dvar, best_dvar))
            # Update the best bound seen
            new_best_bound = jnp.maximum(best_bound, new_bound)

            # Perform optimization step
            updates, new_state = self._branching_optimizer.update(
                lagdual_grads, state, lagdual)
            unc_dual = optax.apply_updates(lagdual, updates)
            new_lagdual = jax.tree_map(lambda x: jnp.maximum(x, 0.), unc_dual)
            return ((new_best_lagdual, new_lagdual, new_best_bound, new_state),
                    None)

          dummy_bound = -float('inf')*jnp.ones([nb_targets])
          initial_carry = (ini_dual_vars, ini_dual_vars, dummy_bound, ini_state)

          (best_lagdual, *_), _ = jax.lax.scan(
              opt_step, initial_carry, None,
              length=self._branching_opt_number_steps)

          _, bound_vals = unbranched_objective(
              jax.lax.stop_gradient(best_lagdual))

          return bound_vals

        bound_vals = evaluate_bound(dual_vars)
      else:
        bound_vals = solve_problem(chunk_objectives)

      chunk_lbs, chunk_ubs = _unpack_opt_problem(bound_vals)
      return chunk_lbs, chunk_ubs

    return _chunked_optimization(to_opt_bound.shape,
                                 self._max_parallel_nodes,
                                 optimize_chunk)


def _chunked_optimization(
    bound_shape: Tuple[int, ...],
    max_parallel_nodes: int,
    optimize_chunk: Callable[[int], Tuple[Tensor, Tensor]],
) -> ibp.IntervalBound:
  """Perform optimization of the target in chunks.

  Args:
    bound_shape: Shape of the bound to compute
    max_parallel_nodes: How many activations to optimize at once. If =0, perform
      optimize all the nodes simultaneously.
    optimize_chunk: Function to optimize a chunk and return updated bounds.
  Returns:
    bounds: Optimized bounds.
  """
  nb_opt = int(np.prod(bound_shape))
  if (max_parallel_nodes == 0) or (nb_opt <= max_parallel_nodes):
    flat_lbs, flat_ubs = optimize_chunk(0)
  else:
    nb_opt_chunk = math.ceil(nb_opt / max_parallel_nodes)
    chunk_indices = jnp.arange(nb_opt_chunk)
    (map_lbs, map_ubs) = jax.lax.map(optimize_chunk, chunk_indices)
    # Remove the padding elements
    flat_lbs = jnp.reshape(map_lbs, (-1,))[:nb_opt]
    flat_ubs = jnp.reshape(map_ubs, (-1,))[:nb_opt]
  lbs = jnp.reshape(flat_lbs, bound_shape)
  ubs = jnp.reshape(flat_ubs, bound_shape)
  bounds = ibp.IntervalBound(lbs, ubs)
  return bounds


def _create_opt_problems(
    non_convex_bound: nonconvex.NonConvexBound,
    chunk_index: int,
    nb_parallel_nodes: int,
) -> Tuple[Dict[Index, Tuple[int, ...]], ParamSet]:
  """Define the objective function and the necessary variables shape.

  Iteratively yields the objectives to minimize in order to limit memory usage.

  Args:
    non_convex_bound: Bound for which to create the optimization problems.
    chunk_index: Index of the optimization chunk to generate.
    nb_parallel_nodes: How large should the optimization chunks be. If 0,
      optimize all problems at once.
  Returns:
    var_to_opt: shapes of the variables to optimize to compute the bounds.
    objectives_by_layer: Objectives to minimize, in the form of a dictionary
      mapping the position of activations to the linear coefficients of the
      objective function.
  """
  # Create the objective matrix
  lb_obj = utils.objective_chunk(
      non_convex_bound.shape, chunk_index, nb_parallel_nodes)
  # Get the objective for the upper bounds.
  ub_obj = -lb_obj
  obj = jnp.concatenate([lb_obj, ub_obj], axis=0)

  # Generate the shape of the variables necessary to solve the problem
  var_to_opt = {}
  for pos, var_shape in non_convex_bound.variables.items():
    var_to_opt[pos] = (obj.shape[0],) + var_shape

  objectives_by_layer = {non_convex_bound.index: obj}
  return var_to_opt, objectives_by_layer


def _unpack_opt_problem(dual_vals: Tensor) -> Tuple[Tensor, Tensor]:
  """Extract the lower bounds and upper bounds from the result of optmization.

  Args:
    dual_vals: Value of the dual returned by the optimization process.
  Returns:
    lb: Tensor containing lower bounds were they were computed and 0 elsewhere.
    ub: Tensor containing upper bounds were they were computed and 0 elsewhere.
  """
  lb_duals, ub_duals = jnp.split(dual_vals, 2, axis=0)

  return lb_duals, -ub_duals


def _pgd_step(current: ParamSet,
              grad: ParamSet,
              step_size: Tensor) -> ParamSet:
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
               termination_dual_gap: float = 1e-2):
    self._num_steps = num_steps
    self._max_step_size = max_step_size
    self._min_step_size = min_step_size
    self._beta_l = beta_l
    self._beta_h = beta_h
    self._check_convergence_every = check_convergence_every
    self._check_relative_dual_gap = check_relative_dual_gap
    self._termination_dual_gap = termination_dual_gap

  def optimize_fun(self, non_convex_bound: nonconvex.NonConvexBound
                   )->Callable[[ParamSet, ParamSet], ParamSet]:
    """Returns a function optimizing the primal variables.

    Args:
      non_convex_bound: NonConvex object to define the objective function over
    Returns:
      optimize: Optimization function.
    """
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
    def quad_approx(x: ParamSet,
                    y: ParamSet,
                    grad_y: ParamSet,
                    step_size: Tensor) -> Tensor:
      quad_approx = 0
      for key, x_var in x.items():
        y_var = y[key]
        grady_var = grad_y[key]
        dims_to_reduce = tuple(range(1, y_var.ndim))
        quad_approx = quad_approx + (
            ((x_var - y_var)*grady_var).sum(axis=dims_to_reduce)
            + 0.5 / step_size * ((x_var - y_var)**2).sum(axis=dims_to_reduce))
      return quad_approx

    def should_decrease(step_size: Tensor,
                        y_stats: Tuple[ParamSet, Tensor, ParamSet],
                        objectives: ParamSet) -> Tensor:
      y, f_y, grad_y = y_stats
      new_x = _pgd_step(y, grad_y, -step_size)
      val_newx, _ = non_convex_bound.primal_fn(new_x, objectives)
      val_qapprox = f_y + quad_approx(new_x, y, grad_y, step_size)
      per_sp_insufficient_progress = (val_newx >= val_qapprox)
      step_size_not_min = step_size > self._min_step_size
      return jnp.logical_and(step_size_not_min, per_sp_insufficient_progress)

    def lower_stepsize_if_needed(
        ls_loop_args:
        Tuple[Tensor, Tensor, Tuple[ParamSet, Tensor, ParamSet], ParamSet],
    ) -> Tuple[Tensor, Tensor, Tuple[ParamSet, Tensor, ParamSet], ParamSet]:
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
    def fista_with_linesearch_step(
        opt_loop_args: Tuple[int, ParamSet, ParamSet, Tensor, Tensor, ParamSet],
    ) -> Tuple[int, ParamSet, ParamSet, Tensor, Tensor, ParamSet]:
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

    def not_all_converged(not_converged_args: Tuple[ParamSet, ParamSet],
                          ) -> bool:
      x, objectives = not_converged_args
      primal, dual = non_convex_bound.dual(x, objectives)
      dgap_value = primal - dual
      if self._check_relative_dual_gap:
        bound_scale = 0.5 * (jnp.abs(primal) + jnp.abs(dual))
        termination_gap = (1 + bound_scale) * self._termination_dual_gap
      else:
        termination_gap = self._termination_dual_gap

      return (dgap_value > termination_gap).any()

    def continue_criterion(
        opt_loop_args: Tuple[int, ParamSet, ParamSet, Tensor, Tensor, ParamSet],
    ) -> Tensor:
      it, x, *_, objectives = opt_loop_args
      not_all_iterations = (it < self._num_steps)
      opt_not_converged = jax.lax.cond(
          (it % self._check_convergence_every) == 0.,
          not_all_converged,
          lambda _: jnp.array(True),
          operand=(x, objectives))
      return jnp.logical_and(opt_not_converged, not_all_iterations)

    ## Define the function to optimize a chunk of the nodes of the activation.
    def optimize(objectives: ParamSet, x: ParamSet) -> ParamSet:
      y = x
      target_dims = objectives[non_convex_bound.index].shape[0]
      gamma = jnp.array(0.)
      step_size = self._max_step_size * jnp.ones(target_dims)
      it = jnp.array(0)

      _, final_x, _, _, _, _ = jax.lax.while_loop(
          continue_criterion,
          fista_with_linesearch_step,
          (it, x, y, gamma, step_size, objectives))

      return final_x

    return optimize


class PGDOptimizer(BoundOptimizer):
  """Projected Gradient Optimizer.

  Optimization can either by taking gradients with respect to the primal or the
  dual objective.

  Passing a number of steps equal to zero will result in the bound derived from
  the initialization.
  """

  def __init__(self, num_steps: int, step_size: float,
               optimize_dual: bool = False):
    self._num_steps = num_steps
    self._step_size = step_size
    self._optimize_dual = optimize_dual

  def optimize_fun(self, non_convex_bound: nonconvex.NonConvexBound,
                   ) -> Callable[[ParamSet, ParamSet], ParamSet]:
    """Returns a function optimizing the primal variables.

    Args:
      non_convex_bound: NonConvex object to define the objective function over
    Returns:
      optimize: Optimization function.
    """
    # If we are going to actually perform optimization, define the function to
    # minimize (either the primal, or the negative of the dual),
    # its gradient and the projection function to use.
    if self._num_steps:
      def fun_to_opt(opt_vars, objectives):
        if self._optimize_dual:
          _, dual_vals = non_convex_bound.dual(opt_vars, objectives)
          obj = -jnp.sum(dual_vals)
        else:
          obj, _ = non_convex_bound.primal_sumfn(opt_vars, objectives)
        return obj
      grad_fun = jax.grad(fun_to_opt)
      proj_fun = lambda x: jnp.clip(x, 0., 1.)

      # Define the optimizer. Because we are minimizing the objective function,
      # we will scale the gradient by a negative step size.
      tx = optax.scale(-self._step_size)

    # Define the function to optimize a chunk of the nodes of the activation.
    def optimize(objectives: ParamSet, var_set: ParamSet) -> ParamSet:

      # Perform the optimization.
      if self._num_steps:
        state = tx.init(var_set)

        def opt_step(_, state_and_var):
          state, var_set = state_and_var
          grads = grad_fun(var_set, objectives)
          updates, new_state = tx.update(grads, state, var_set)
          unc_var_set = optax.apply_updates(var_set, updates)
          new_var_set = jax.tree_map(proj_fun, unc_var_set)
          return new_state, new_var_set

        _, var_set = jax.lax.fori_loop(0, self._num_steps, opt_step,
                                       (state, var_set))

      return var_set

    return optimize

