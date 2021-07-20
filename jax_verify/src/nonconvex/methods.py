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

"""Pre-canned non-convex methods."""
import functools

from jax_verify.src import ibp
from jax_verify.src.nonconvex import duals
from jax_verify.src.nonconvex import nonconvex
from jax_verify.src.nonconvex import optimizers


nonconvex_ibp_bound_propagation = functools.partial(
    nonconvex.build_nonconvex_formulation,
    duals.WolfeNonConvexBound,
    lambda: nonconvex.BaseBoundConcretizer(ibp.bound_transform))


def _create_nostep_optimizer() -> optimizers.OptimizingConcretizer:
  return optimizers.OptimizingConcretizer(
      optimizers.PGDOptimizer(0, 0., optimize_dual=False),
      max_parallel_nodes=512)

nonconvex_constopt_bound_propagation = functools.partial(
    nonconvex.build_nonconvex_formulation,
    duals.WolfeNonConvexBound, _create_nostep_optimizer)
