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

"""Mixture of strategies for solving the inner maximization."""
from typing import Any

import jax.numpy as jnp

from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import verify_utils

InnerVerifInstance = verify_utils.InnerVerifInstance


class MixedStrategy(dual_build.InnerMaxStrategy):
  """Solves inner maximisations with a combination of solvers."""

  def __init__(self, solvers, solver_weights):
    self._solvers = solvers
    self._solver_weights = solver_weights

  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.ndarray,
      step: int,
  ) -> jnp.ndarray:
    """Solve maximization problem of opt_instance with a combination of solvers.

    Args:
      inner_dual_vars: Dual variables for the inner maximisation.
      opt_instance: Verification instance that defines optimization problem to
        be solved.
      key: Jax PRNG key.
      step: outer optimization iteration number.

    Returns:
      final_value: final value of the objective function found by PGA.
    """
    # some renaming to simplify variable names
    layer_idx = opt_instance.idx
    solver_weights_for_layer = self._solver_weights[layer_idx]
    solvers_for_layer = self._solvers[layer_idx]
    final_value = 0.
    for solver, solver_weight, inner_var in zip(solvers_for_layer,
                                                solver_weights_for_layer,
                                                inner_dual_vars):
      final_value += solver_weight * solver.solve_max(inner_var, opt_instance,
                                                      key, step)
    return final_value  # pytype: disable=bad-return-type  # jnp-array

  def init_layer_inner_params(self, opt_instance):
    """Returns initial inner maximisation duals and their types."""

    dual_vars_types = [
        solver.init_layer_inner_params(opt_instance)
        for solver in self._solvers[opt_instance.idx]
    ]
    return zip(*dual_vars_types)

  def supports_stochastic_parameters(self):
    for solvers_for_layer in self._solvers:
      for solver in solvers_for_layer:
        if not solver.supports_stochastic_parameters():
          return False
    return True
