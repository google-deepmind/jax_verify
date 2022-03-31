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

"""Pre-canned non-convex methods."""
from typing import Callable

import jax.numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import ibp
from jax_verify.src import synthetic_primitives
from jax_verify.src.nonconvex import duals
from jax_verify.src.nonconvex import nonconvex
from jax_verify.src.nonconvex import optimizers


Tensor = jnp.ndarray
Index = bound_propagation.Index
TransformContext = bound_propagation.TransformContext
Nest = bound_propagation.Nest


def nonconvex_ibp_bound_propagation(
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[graph_traversal.GraphInput],
    graph_simplifier=synthetic_primitives.default_simplifier,
) -> Nest[nonconvex.NonConvexBound]:
  """Builds the non-convex objective using IBP.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only arguments the network inputs.
    *bounds: Bounds on the inputs of the function.
    graph_simplifier: What graph simplifier to use.
  Returns:
    output_bounds: NonConvex bounds that can be optimized with a solver.
  """
  algorithm = nonconvex.nonconvex_algorithm(
      duals.WolfeNonConvexBound,
      nonconvex.BaseBoundConcretizer(),
      base_boundprop=ibp.bound_transform)
  output_bounds, _ = bound_propagation.bound_propagation(
      algorithm, function, *bounds, graph_simplifier=graph_simplifier)
  return output_bounds


def nonconvex_constopt_bound_propagation(
    function: Callable[..., Nest[Tensor]],
    *bounds: Nest[graph_traversal.GraphInput],
    graph_simplifier=synthetic_primitives.default_simplifier,
) -> Nest[nonconvex.NonConvexBound]:
  """Builds the optimizable objective.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only arguments the network inputs.
    *bounds: Bounds on the inputs of the function.
    graph_simplifier: What graph simplifier to use.
  Returns:
    output_bounds: NonConvex bounds that can be optimized with a solver.
  """
  nostep_optimizer = optimizers.OptimizingConcretizer(
      optimizers.PGDOptimizer(0, 0., optimize_dual=False),
      max_parallel_nodes=512)
  algorithm = nonconvex.nonconvex_algorithm(
      duals.WolfeNonConvexBound, nostep_optimizer)
  output_bounds, _ = bound_propagation.bound_propagation(
      algorithm, function, *bounds, graph_simplifier=graph_simplifier)
  return output_bounds
