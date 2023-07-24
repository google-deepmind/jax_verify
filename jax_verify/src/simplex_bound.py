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

"""Bound with L1 constraints."""
import jax
from jax import numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import opt_utils
from jax_verify.src.types import Tensor


class SimplexIntervalBound(bound_propagation.IntervalBound):
  """Represent a bound for which we have a constraint on the sum of coordinates.

  Each coordinate is subject to interval constraints, and the sum of all
  coordinates must be equal to a given value.

  """

  def __init__(self, lower_bound: Tensor, upper_bound: Tensor,
               simplex_sum: float):
    super(SimplexIntervalBound, self).__init__(lower_bound, upper_bound)
    self._simplex_sum = simplex_sum

  @property
  def simplex_sum(self) -> float:
    return self._simplex_sum

  @classmethod
  def from_jittable(
      cls,
      jittable_simplexint_bound: bound_propagation.JittableInputBound
  ) -> 'SimplexIntervalBound':
    return cls(jittable_simplexint_bound.lower,
               jittable_simplexint_bound.upper,
               jittable_simplexint_bound.kwargs['simplex_sum'])

  def to_jittable(self) -> bound_propagation.JittableInputBound:
    return bound_propagation.JittableInputBound(
        self.lower, self.upper, {SimplexIntervalBound: None},
        {'simplex_sum': self.simplex_sum})

  def project_onto_bound(self, tensor: Tensor) -> Tensor:
    return opt_utils.project_onto_interval_simplex(self.lower, self.upper,
                                                   self.simplex_sum, tensor)


def concretize_linear_function_simplexinterval_constraints(
    linexp, input_bound: SimplexIntervalBound) -> Tensor:
  """Compute the lower bound of a linear function under Simplex constraints."""

  solve_lin = jax.vmap(opt_utils.fractional_exact_knapsack,
                       in_axes=(0, None, None, None))

  # We are maximizing -lin_coeffs*x in order to minimize lin_coeffs*x
  neg_sum_lin_bound = solve_lin(-linexp.lin_coeffs, input_bound.simplex_sum,
                                input_bound.lower, input_bound.upper)
  return linexp.offset - neg_sum_lin_bound


def concretizing_input_simplexinterval_constraints(
    linexp, input_bound: SimplexIntervalBound) -> Tensor:
  """Compute the input that achieves the lower bound of a linear function."""
  flat_lower = jnp.reshape(input_bound.lower, (-1,))
  flat_upper = jnp.reshape(input_bound.upper, (-1,))
  flat_lin_coeffs = jnp.reshape(linexp.lin_coeffs, (-1, flat_lower.size))

  def single_linexpr_concretizing_inp(coeffs):
    _, sorted_lower, sorted_upper, sorted_idx = jax.lax.sort(
        (coeffs, flat_lower, flat_upper, jnp.arange(coeffs.size)), num_keys=1)
    sorted_assignment = opt_utils.sorted_knapsack(sorted_lower, sorted_upper,
                                                  input_bound.simplex_sum,
                                                  backward=False)
    # This is a cute trick to avoid using a jnp.take_along_axis, which is
    # usually quite slow, particularly on TPU, when you do permutation.
    # jax.lax.sort can take multiple arguments, and will sort them according
    # to the ordering of the first tensor.
    # When we did the sorting of the weights, we also sorted the index of each
    # coordinate. By sorting by it, we will recover the initial ordering.
    _, assignment = jax.lax.sort((sorted_idx, sorted_assignment), num_keys=1)
    return assignment

  flat_conc_input = jax.vmap(single_linexpr_concretizing_inp)(flat_lin_coeffs)
  return jnp.reshape(flat_conc_input, linexp.lin_coeffs.shape)
