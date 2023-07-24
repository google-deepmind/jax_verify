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

"""Tests for the bound propagation on NonConvex bounds.

We don't really check the value but at least that the different propagation
works and that the bound can be evaluated.
"""

import collections
import functools

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src.mip_solver import cvxpy_relaxation_solver
from jax_verify.src.mip_solver import relaxation
from jax_verify.src.nonconvex import duals
from jax_verify.src.nonconvex import nonconvex
from jax_verify.src.nonconvex import optimizers
from jax_verify.tests import test_utils
import numpy as np


def _random_objectives_primal_variables(rng_key, nonconvex_bound,
                                        nb_opt_targets):
  # Get a set of primal variables
  var_set = {}
  for pos, var_shape in nonconvex_bound.variables.items():  # pytype: disable=attribute-error  # jax-ndarray
    rng_key, new_key = jax.random.split(rng_key)
    var_set[pos] = jax.random.uniform(
        new_key, shape=(nb_opt_targets, *var_shape))

  objectives_dict = {}
  for index, prev_bound in nonconvex_bound.previous_bounds.items():  # pytype: disable=attribute-error  # jax-ndarray
    rng_key, new_key = jax.random.split(rng_key)
    bound_shape = prev_bound.shape
    linfun_shape = (nb_opt_targets, *bound_shape)
    objectives_dict[index] = jax.random.normal(new_key, shape=linfun_shape)
  return objectives_dict, var_set


class NonConvexBoundTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('base_bound', jax_verify.nonconvex_ibp_bound_propagation),
      ('inner_opt', jax_verify.nonconvex_constopt_bound_propagation))
  def test_fc_nonconvex(self, boundprop_fun):

    @hk.without_apply_rng
    @hk.transform
    def linear_model(inp):
      return hk.Linear(1)(inp)

    z = jnp.array([[1., 2., 3.]])
    params = {'linear':
              {'w': jnp.ones((3, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}
    input_bounds = jax_verify.IntervalBound(z-1., z+1.)
    fun = functools.partial(linear_model.apply, params)

    output_bounds = boundprop_fun(fun, input_bounds)

    concretizer = optimizers.OptimizingConcretizer(
        optimizers.PGDOptimizer(0, 0.), 0)
    final_bounds = concretizer.get_bounds(output_bounds)

    self.assertTrue(all(final_bounds.upper >= final_bounds.lower))

  @parameterized.named_parameters(
      ('base_bound', jax_verify.nonconvex_ibp_bound_propagation),
      ('inner_opt', jax_verify.nonconvex_constopt_bound_propagation))
  def test_conv2d_nonconvex(self, boundprop_fun):

    @hk.without_apply_rng
    @hk.transform
    def conv2d_model(inp):
      return hk.Conv2D(output_channels=1, kernel_shape=(2, 2),
                       padding='VALID', stride=1, with_bias=True)(inp)

    z = jnp.array([1., 2., 3., 4.])
    z = jnp.reshape(z, [1, 2, 2, 1])

    params = {'conv2_d':
              {'w': jnp.ones((2, 2, 1, 1), dtype=jnp.float32),
               'b': jnp.array([2.])}}

    fun = functools.partial(conv2d_model.apply, params)
    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    output_bounds = boundprop_fun(fun, input_bounds)

    concretizer = optimizers.OptimizingConcretizer(
        optimizers.PGDOptimizer(0, 0.), 0)
    final_bounds = concretizer.get_bounds(output_bounds)

    self.assertTrue(all(final_bounds.upper >= final_bounds.lower))

  @parameterized.named_parameters(
      ('base_bound', jax_verify.nonconvex_ibp_bound_propagation),
      ('inner_opt', jax_verify.nonconvex_constopt_bound_propagation))
  def test_relu_nonconvex(self, boundprop_fun):
    def relu_model(inp):
      return jax.nn.relu(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    output_bounds = boundprop_fun(relu_model, input_bounds)

    concretizer = optimizers.OptimizingConcretizer(
        optimizers.PGDOptimizer(0, 0.), 0)
    final_bounds = concretizer.get_bounds(output_bounds)

    self.assertTrue((final_bounds.upper >= final_bounds.lower).all())

  @parameterized.named_parameters(
      ('base_bound', jax_verify.nonconvex_ibp_bound_propagation),
      ('inner_opt', jax_verify.nonconvex_constopt_bound_propagation))
  def test_softplus_nonconvex(self, boundprop_fun):
    def softplus_model(inp):
      return jax.nn.softplus(inp)
    z = jnp.array([[-2., 3.]])

    input_bounds = jax_verify.IntervalBound(z - 1., z + 1.)

    output_bounds = boundprop_fun(softplus_model, input_bounds)

    concretizer = optimizers.OptimizingConcretizer(
        optimizers.PGDOptimizer(0, 0.), 0)
    final_bounds = concretizer.get_bounds(output_bounds)

    self.assertTrue((final_bounds.upper >= final_bounds.lower).all())

  def test_dualopt_noncrossing_bounds(self):
    randgen = np.random.RandomState(42)
    params = [
        (randgen.normal(size=(784, 2)), randgen.normal(size=(2,))),
        (randgen.normal(size=(2, 10)), randgen.normal(size=(10,))),
    ]

    with jax_verify.open_file('mnist/x_test_first100.npy', 'rb') as f:
      mnist_x = np.load(f)
    inps = jnp.reshape(mnist_x[:4], (4, 784))

    def logits_fun(inp):
      preact_1 = jnp.dot(inp, params[0][0]) + params[0][1]
      act_1 = jax.nn.softplus(preact_1)
      out = jnp.dot(act_1, params[1][0]) + params[1][1]
      return out

    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)
    nonconvex_ibp_bounds = jax_verify.nonconvex_ibp_bound_propagation(
        logits_fun, input_bounds)

    # Optimizing on the dual is the way to directly optimize the bounds that we
    # have so it increases the chance of finding places where the dual is badly
    # calculated.
    concretizer = optimizers.OptimizingConcretizer(
        optimizers.PGDOptimizer(5, 1, optimize_dual=True), 0)
    final_pgddual_ibp_bounds = concretizer.get_bounds(nonconvex_ibp_bounds)

    self.assertTrue((final_pgddual_ibp_bounds.upper >=
                     final_pgddual_ibp_bounds.lower).all())

  @parameterized.named_parameters(
      ('Wolfe', duals.WolfeNonConvexBound),
      ('LinLagrangian', duals.LinLagrangianNonConvexBound),
      ('MinLagrangian', duals.MinLagrangianNonConvexBound)
  )
  def test_negative_dualgap(self, bound_cls):
    batch_size = 2
    nb_opt_targets = 3
    key = jax.random.PRNGKey(42)
    problem_key, var_key = jax.random.split(key)
    fun, (lb, ub) = test_utils.set_up_toy_problem(problem_key, batch_size,
                                                  [64, 2, 2, 2])
    input_bounds = jax_verify.IntervalBound(lb, ub)

    algorithm = nonconvex.nonconvex_algorithm(
        bound_cls, nonconvex.BaseBoundConcretizer(),
        base_boundprop=ibp.bound_transform)

    # NonConvex bound
    nonconvex_ibp_bounds, _ = bound_propagation.bound_propagation(
        algorithm, fun, input_bounds)
    objectives, var_set = _random_objectives_primal_variables(
        var_key, nonconvex_ibp_bounds, nb_opt_targets)

    primal, dual = nonconvex_ibp_bounds.dual(var_set, objectives)  # pytype: disable=attribute-error  # jax-ndarray
    self.assertTrue((dual <= primal).all())

  def test_collect_lagrangian_layers(self):
    batch_size = 2
    nb_opt_targets = 3
    key = jax.random.PRNGKey(42)
    problem_key, primal_var_key, dual_var_key = jax.random.split(key, num=3)
    fun, (lb, ub) = test_utils.set_up_toy_problem(problem_key, batch_size,
                                                  [32, 18, 2])
    input_bounds = jax_verify.IntervalBound(lb, ub)

    # Get the MinLagrangian and LinLagrangian propagation
    linlag_bound_algorithm = nonconvex.nonconvex_algorithm(
        duals.LinLagrangianNonConvexBound,
        nonconvex.BaseBoundConcretizer(), base_boundprop=ibp.bound_transform)
    minlag_bound_algorithm = nonconvex.nonconvex_algorithm(
        duals.MinLagrangianNonConvexBound,
        nonconvex.BaseBoundConcretizer(), base_boundprop=ibp.bound_transform)

    # NonConvex bound
    linlag_bound, _ = bound_propagation.bound_propagation(
        linlag_bound_algorithm, fun, input_bounds)
    minlag_bound, _ = bound_propagation.bound_propagation(
        minlag_bound_algorithm, fun, input_bounds)

    objectives, var_set = _random_objectives_primal_variables(
        primal_var_key, linlag_bound, nb_opt_targets)
    _, acts = linlag_bound.primal_fn(var_set, objectives)  # pytype: disable=attribute-error  # jax-ndarray

    dual_vars = {}
    for index, primal_var in var_set.items():
      dual_var_key, new_key = jax.random.split(dual_var_key)
      dual_vars[index] = jax.random.normal(new_key, shape=primal_var.shape)

    ## Test separately each layers
    for index in linlag_bound.previous_bounds:  # pytype: disable=attribute-error  # jax-ndarray
      linlag_intermediate_bound = linlag_bound.previous_bounds[index]  # pytype: disable=attribute-error  # jax-ndarray
      minlag_intermediate_bound = minlag_bound.previous_bounds[index]  # pytype: disable=attribute-error  # jax-ndarray

      lagrangian_level_fn = linlag_intermediate_bound.lagrangian_level_fn  # pytype: disable=attribute-error  # jax-ndarray
      lagrangian_varterms_fn = minlag_intermediate_bound.lagrangian_varterms_fn  # pytype: disable=attribute-error  # jax-ndarray

      dvar = dual_vars[index]
      # Get all the per variables term for a level, and evaluate them
      lagrangian_dict = collections.defaultdict(list)
      lagrangian_varterms_fn(dvar, lagrangian_dict)
      per_var_lagrangian = 0
      for var_idx, lag_terms in lagrangian_dict.items():
        for term in lag_terms:
          out_term = term[1](acts[var_idx])
          dims_to_reduce = tuple(range(1, out_term.ndim))
          per_var_lagrangian = per_var_lagrangian + out_term.sum(dims_to_reduce)
      # Get simply the lagrangian for a level and evaluate it.
      per_level_lagrangian = lagrangian_level_fn(dvar, acts)
      # The two should give exactly the same results.
      diff = jnp.abs(per_level_lagrangian - per_var_lagrangian).max()
      self.assertAlmostEqual(
          diff, 0, delta=1e-3,
          msg=f'Difference in the lagrangian computation for layer {index}')

  def test_collect_lagrangian_network(self):
    batch_size = 2
    nb_opt_targets = 3
    key = jax.random.PRNGKey(42)
    problem_key, primal_var_key, dual_var_key = jax.random.split(key, num=3)
    fun, (lb, ub) = test_utils.set_up_toy_problem(problem_key, batch_size,
                                                  [64, 2, 2, 2])
    input_bounds = jax_verify.IntervalBound(lb, ub)

    # Get the MinLagrangian and LinLagrangian propagation
    linlag_bound_algorithm = nonconvex.nonconvex_algorithm(
        duals.LinLagrangianNonConvexBound,
        nonconvex.BaseBoundConcretizer(), base_boundprop=ibp.bound_transform)
    minlag_bound_algorithm = nonconvex.nonconvex_algorithm(
        duals.MinLagrangianNonConvexBound,
        nonconvex.BaseBoundConcretizer(), base_boundprop=ibp.bound_transform)

    # NonConvex bound
    linlag_bound, _ = bound_propagation.bound_propagation(
        linlag_bound_algorithm, fun, input_bounds)
    minlag_bound, _ = bound_propagation.bound_propagation(
        minlag_bound_algorithm, fun, input_bounds)

    objectives, var_set = _random_objectives_primal_variables(
        primal_var_key, minlag_bound, nb_opt_targets)
    primal, acts = minlag_bound.primal_fn(var_set, objectives)  # pytype: disable=attribute-error  # jax-ndarray

    dual_vars = {}
    for index, primal_var in var_set.items():
      dual_var_key, new_key = jax.random.split(dual_var_key)
      dual_vars[index] = jax.random.normal(new_key, shape=primal_var.shape)

    all_lagrangian_terms = minlag_bound.collect_lagrangian_varterms(  # pytype: disable=attribute-error  # jax-ndarray
        objectives, dual_vars)
    per_var_lagrangian = primal
    for var_index, lag_terms in all_lagrangian_terms.items():
      for term in lag_terms:
        all_contrib = term[1](acts[var_index])
        dims_to_reduce = tuple(range(1, all_contrib.ndim))
        var_contrib = all_contrib.sum(axis=dims_to_reduce)
        per_var_lagrangian = per_var_lagrangian + var_contrib
    per_level_lagrangian, _ = linlag_bound._lagrangian_fn(acts, objectives,  # type: ignore  # jax-ndarray
                                                          dual_vars)
    diff = jnp.abs(per_level_lagrangian - per_var_lagrangian).max()
    self.assertAlmostEqual(
        diff, 0, delta=1e-3,
        msg='The two lagrangian implementation are not equivalent.')
    # Let's also sanity check that we can correctly optimize our lagrangian
    # terms.
    for var_idx, lag_terms in all_lagrangian_terms.items():
      lower = minlag_bound.previous_bounds[var_idx].lower  # pytype: disable=attribute-error  # jax-ndarray
      lower = jnp.repeat(jnp.expand_dims(lower, 0), nb_opt_targets, axis=0)
      upper = minlag_bound.previous_bounds[var_idx].upper  # pytype: disable=attribute-error  # jax-ndarray
      upper = jnp.repeat(jnp.expand_dims(upper, 0), nb_opt_targets, axis=0)

      def eval_lagrangian_terms(var_act, lag_terms=lag_terms):
        per_var_lagrangians = []
        for term in lag_terms:
          out_term = term[1](var_act)
          dims_to_reduce = tuple(range(1, out_term.ndim))
          per_var_lagrangians.append(out_term.sum(dims_to_reduce))
        return sum(per_var_lagrangians)

      minimizing_input = duals._optimize_lagrangian_terms(
          lag_terms, lower, upper)
      minimized_varlagrangian = eval_lagrangian_terms(minimizing_input)

      for _ in range(10):
        unif = np.random.uniform(size=lower.shape)
        candidate_input = lower + unif * (upper - lower)
        candidate_varlagrangian = eval_lagrangian_terms(candidate_input)

        min_gap = (candidate_varlagrangian - minimized_varlagrangian).min()
        self.assertGreater(min_gap, 0,
                           msg=('Minimization of the lagrangian with regards to'
                                f'variable {var_idx} is not correct.'))

  @parameterized.named_parameters(
      ('Wolfe', duals.WolfeNonConvexBound),
      ('LinLagrangian', duals.LinLagrangianNonConvexBound),
      ('MinLagrangian', duals.MinLagrangianNonConvexBound)
  )
  def test_comparefista_to_cvxpy(self, bound_cls):
    batch_size = 2
    key = jax.random.PRNGKey(42)
    fun, (lb, ub) = test_utils.set_up_toy_problem(key, batch_size, [32, 18, 2])
    input_bounds = jax_verify.IntervalBound(lb, ub)

    # NonConvex Result
    nonconvex_ibp_bound_algorithm = nonconvex.nonconvex_algorithm(
        bound_cls, nonconvex.BaseBoundConcretizer(),
        base_boundprop=ibp.bound_transform)
    nonconvex_ibp_bounds, _ = bound_propagation.bound_propagation(
        nonconvex_ibp_bound_algorithm, fun, input_bounds)
    fista_optimizer = optimizers.LinesearchFistaOptimizer(
        40, beta_l=0.8, termination_dual_gap=1e-6)
    fista_concretizer = optimizers.OptimizingConcretizer(
        fista_optimizer, 0)
    dual_bound = fista_concretizer.get_bounds(nonconvex_ibp_bounds)

    relaxation_transform = relaxation.RelaxationTransform(
        jax_verify.ibp_transform)
    cvxpy_final_var, env = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(relaxation_transform),
        fun, input_bounds)
    nb_targets = np.prod(cvxpy_final_var.shape[1:])  # pytype: disable=attribute-error  # jax-ndarray
    for batch_index in range(batch_size):
      for target_index in range(nb_targets):
        objective = (jnp.arange(nb_targets) == target_index).astype(jnp.float32)
        objective_bias = 0.

        cvxpy_lower, _, _ = relaxation.solve_relaxation(
            cvxpy_relaxation_solver.CvxpySolver, objective, objective_bias,
            cvxpy_final_var, env, batch_index)
        cvxpy_neg_upper, _, _ = relaxation.solve_relaxation(
            cvxpy_relaxation_solver.CvxpySolver, -objective, objective_bias,
            cvxpy_final_var, env, batch_index)
        cvxpy_upper = - cvxpy_neg_upper

        nonconvex_lower = dual_bound.lower[batch_index, target_index]
        nonconvex_upper = dual_bound.upper[batch_index, target_index]
        self.assertGreaterEqual(
            nonconvex_upper, nonconvex_lower,
            msg='Bounds are crossing.')
        self.assertAlmostEqual(
            cvxpy_lower, nonconvex_lower, delta=1e-2,
            msg='Inaccurate lower bound.')
        self.assertAlmostEqual(
            cvxpy_upper, nonconvex_upper, delta=1e-2,
            msg='Inaccurate upper bound ')

  def test_chunked_optimization(self):
    batch_size = 3
    key = jax.random.PRNGKey(42)

    fun, (lb, ub) = test_utils.set_up_toy_problem(key, batch_size, [32, 16, 10])
    input_bounds = jax_verify.IntervalBound(lb, ub)

    nonconvex_bound = jax_verify.nonconvex_ibp_bound_propagation(
        fun, input_bounds)

    optimizer = optimizers.PGDOptimizer(10, 0.1)
    chunked_concretizer = optimizers.OptimizingConcretizer(
        optimizer, 3)
    full_concretizer = optimizers.OptimizingConcretizer(optimizer, 0)

    full_bounds = full_concretizer.get_bounds(nonconvex_bound)
    chunked_bounds = chunked_concretizer.get_bounds(nonconvex_bound)

    np.testing.assert_array_almost_equal(chunked_bounds.lower,
                                         full_bounds.lower)
    np.testing.assert_array_almost_equal(chunked_bounds.upper,
                                         full_bounds.upper)

  def test_objfun_derivation(self):
    batch_size = 3
    nb_opt_targets = 8
    key = jax.random.PRNGKey(42)
    problem_key, objective_key = jax.random.split(key)

    fun, (lb, ub) = test_utils.set_up_toy_problem(problem_key, batch_size,
                                                  [32, 16, 10])
    input_bounds = jax_verify.IntervalBound(lb, ub)
    nonconvex_bound = jax_verify.nonconvex_ibp_bound_propagation(
        fun, input_bounds)

    objectives, var_set = _random_objectives_primal_variables(
        objective_key, nonconvex_bound, nb_opt_targets)
    # On-paper derived derivatives for the standard dot product objective
    # is just `objectives`.
    ref_dual_vars = objectives.copy()

    # Compute it using autograd.
    dual_vars, _ = nonconvex_bound._compute_dualvars_convexgrad(var_set,
                                                                objectives)
    for node_idx in ref_dual_vars:
      np.testing.assert_array_almost_equal(dual_vars[node_idx],
                                           ref_dual_vars[node_idx])


if __name__ == '__main__':
  absltest.main()
