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

"""Provides optimizers to use with the NonConvexBound in `nonconvex.py`.
"""

import abc
import functools
from typing import Tuple, Callable, Dict, Optional, List, Union

import jax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src import optimizers
from jax_verify.src import utils
from jax_verify.src.nonconvex import nonconvex
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

  def concrete_bound(
      self,
      graph: bound_propagation.PropagationGraph,
      env: Dict[jax.core.Var, Union[bound_propagation.Bound, Tensor]],
      nonconvex_bound: nonconvex.NonConvexBound) -> bound_propagation.Bound:
    return self.get_bounds(nonconvex_bound)

  def get_bounds(self, to_opt_bound: nonconvex.NonConvexBound
                 ) -> bound_propagation.Bound:
    optimize_fun = self._optimizer.optimize_fun(to_opt_bound)

    def bound_fn(obj: Tensor) -> Tuple[Tensor, Tensor]:
      var_shapes, chunk_objectives = _create_opt_problems(to_opt_bound, obj)
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
            index_to_update = jnp.index_exp[:, neuron_idx]
            flat_node_obj = jnp.reshape(objectives[node_idx], (nb_targets, -1))
            flat_updated_node_obj = flat_node_obj.at[index_to_update].add(
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

    return ibp.IntervalBound(*utils.chunked_bounds(
        to_opt_bound.shape, self._max_parallel_nodes, bound_fn))


def _create_opt_problems(
    non_convex_bound: nonconvex.NonConvexBound,
    obj: Tensor,
) -> Tuple[Dict[Index, Tuple[int, ...]], ParamSet]:
  """Define the objective function and the necessary variables shape.

  Iteratively yields the objectives to minimize in order to limit memory usage.

  Args:
    non_convex_bound: Bound for which to create the optimization problems.
    obj: One-hot tensor of shape (nb_parallel_nodes, *obj_shape) specifying
      the elements of the objective to optimise.
  Returns:
    var_to_opt: shapes of the variables to optimize to compute the bounds.
    objectives_by_layer: Objectives to minimize, in the form of a dictionary
      mapping the position of activations to the linear coefficients of the
      objective function.
  """
  # Get the objective for the upper bounds.
  lb_obj = obj
  ub_obj = -obj
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


class LinesearchFistaOptimizer(BoundOptimizer):
  """FISTA with line search."""

  def __init__(self,
               num_steps: int,
               max_step_size: float = 100.0,
               min_step_size: float = 1e-5,
               beta_l: float = 0.5,
               beta_h: float = 1.5,
               check_convergence_every: int = 1,
               check_relative_dual_gap: bool = False,
               termination_dual_gap: float = 1e-2):
    self._optimizer = optimizers.LinesearchFistaOptimizer(
        num_steps, max_step_size, min_step_size,
        beta_l, beta_h, check_convergence_every)
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

    def fun_to_opt(objectives, opt_vars):
      """Target functions to minimize.

      The functions to minimize are the primal objectives, given
      by non_convex_bound.primal function. This function also returns the
      intermediate activation as an auxiliary output, which we want to
      ignore.

      Args:
        objectives: Linear coefficients of the objective function on the
          activations.
        opt_vars: Value of the parameters to evaluate.
      Returns:
        obj: sum of the objective functions.
      """
      obj, _ = non_convex_bound.primal_fn(opt_vars, objectives)
      return obj

    proj_fun = lambda opt_var: jnp.clip(opt_var, 0., 1.)
    project_all_params = lambda opt_vars: jax.tree_map(proj_fun, opt_vars)

    def any_not_done(objectives, opt_vars):
      primal, dual = non_convex_bound.dual(opt_vars, objectives)
      dgap_value = primal - dual
      if self._check_relative_dual_gap:
        bound_scale = 0.5 * (jnp.abs(primal) + jnp.abs(dual))
        termination_gap = (1 + bound_scale) * self._termination_dual_gap
      else:
        termination_gap = self._termination_dual_gap
      return (dgap_value > termination_gap).any()

    def optimize(objectives: ParamSet, var_set: ParamSet) -> ParamSet:
      obj_fun = functools.partial(fun_to_opt, objectives)
      not_conv_fun = functools.partial(any_not_done, objectives)

      opt_fun = self._optimizer.optimize_fn(obj_fun, project_all_params,
                                            not_conv_fun)
      return opt_fun(var_set)

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
    # Define the optimizer. Because we are minimizing the objective function,
    # we will scale the gradient by a negative step size.
    gradient_transform = optax.scale(-step_size)
    self._optimizer = optimizers.OptaxOptimizer(
        gradient_transform, num_steps=num_steps)
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
    def fun_to_opt(opt_vars, objectives):
      if self._optimize_dual:
        _, dual_vals = non_convex_bound.dual(opt_vars, objectives)
        obj = -jnp.sum(dual_vals)
      else:
        obj, _ = non_convex_bound.primal_sumfn(opt_vars, objectives)
      return obj
    proj_fun = lambda x: jnp.clip(x, 0., 1.)
    project_all_params = lambda x: jax.tree_map(proj_fun, x)

    # Define the function to optimize a chunk of the nodes of the activation.
    def optimize(objectives: ParamSet, var_set: ParamSet) -> ParamSet:
      obj_fun = lambda opt_vars: fun_to_opt(opt_vars, objectives)
      opt_fun = self._optimizer.optimize_fn(obj_fun, project_all_params)
      return opt_fun(var_set)

    return optimize
