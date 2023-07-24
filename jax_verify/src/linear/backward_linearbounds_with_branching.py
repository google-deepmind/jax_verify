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

"""Implementation of Beta-Crown.

Branching decisions are given by a Tuple of fixed tensor.

LayIdx is an integer tensor with size (max_splits, MAX_JAXPR_DEPTH) indicating
which layer to split on. MAX_JAXPR_DEPTH corresponds to a maximum nesting of
indexes that we will specify.

All other tensors are 1D with length max_splits.
NeurIdx is a 1D integer tensor indicating the neuron in that layer.
BranchVal is a 1D floating point tensor indicating where the cut is.
IsUpperBranch is a boolean tensor indicating if the inequality is
  neur > branch_val (True)
  neur < branch_val (False)
"""
from typing import Mapping, Optional, Sequence, Tuple

from jax import numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import concretization
from jax_verify.src import graph_traversal
from jax_verify.src import optimizers
from jax_verify.src.branching import branch_selection
from jax_verify.src.linear import backward_crown
from jax_verify.src.linear import linear_relaxations
from jax_verify.src.types import Index, Nest, Primitive, SpecFn, Tensor  # pylint: disable=g-multiple-import
import optax


def _update_lin_with_lagrangian(eqn_lincoeffs: Tensor,
                                eqn_offset: Tensor,
                                active_branch_mask: Tensor,
                                lagrangian_variables: Tensor,
                                neur_idxs: Tensor,
                                branch_val: Tensor,
                                is_upper: Tensor) -> Tuple[Tensor, Tensor]:
  """Include the contribution of the lagrangian to the linear bounds.

  Args:
    eqn_lincoeffs: (nb_targets, *act_shape)[float ]coefficients of the linear
      function bounding the output as a function of this layer's activation.
    eqn_offset: (nb_targets,)[float] constant term of the linear function
      bounding the output.
    active_branch_mask: (nb_splits,)[bool] Boolean array indicating which
      branching constraints are active at this level.
    lagrangian_variables: (nb_splits,)[float] Value of the lagrangian multiplier
      associated with each branching constraint.
    neur_idxs: (nb_splits,)[int] Indication of what neuron is branched on. The
      integers represent the location in the flattened activation array.
    branch_val: (nb_splits,)[float] Cut-off point of the split constraint.
    is_upper: (nb_splits,)[bool] Whether the constraint enforce the activation
      to be above branch_val or below.
  Returns:
    lagrangianed_lincoeffs: Coefficients of the linear function, including
      the lagrangian contribution.
    lagrangianed_offset: Constant term of the linear function, including the
      lagrangian contribution.
  """
  # masked_lagrangian is (nb_splits,)
  masked_lagrangian = active_branch_mask * lagrangian_variables
  signed_masked_lagrangian = jnp.where(is_upper, -masked_lagrangian,
                                       masked_lagrangian)

  # We obtain here a scalar term, this is the contribution to the constant
  # term (not a function of the input) of the linear equation by the
  # lagrangian.
  # We sum over the splits (because we take into consideration all the
  # lagrangians of that layer, and we have masked the irrelevant ones).
  lagrangian_constant_term = (-signed_masked_lagrangian * branch_val).sum()
  lagrangianed_offset = (eqn_offset +
                         jnp.expand_dims(lagrangian_constant_term, 0))

  # We now compute the new linear function to be lower-bounded.
  # We want to add the lagrangian contribution to the function, for each of
  # the target.

  # We start by converting the array into the flat array
  flat_lincoeffs = jnp.reshape(eqn_lincoeffs, (eqn_lincoeffs.shape[0], -1))
  # We now gather the lin coefficients to update.
  to_update = flat_lincoeffs.at[:, neur_idxs]
  # and add the lagrangian contribution to them.
  flat_lagrangianed_lincoeffs = to_update.add(signed_masked_lagrangian,
                                              mode='drop')
  lagrangianed_lincoeffs = jnp.reshape(flat_lagrangianed_lincoeffs,
                                       eqn_lincoeffs.shape)

  return lagrangianed_lincoeffs, lagrangianed_offset


class ConstrainedLinearBoundBackwardTransform(
    backward_crown.LinearBoundBackwardConcretizingTransform):
  """Backward transformation adding the linear contributions from branching.

  Those contributions come from the defined branching decisions and the
  lagrangian variables associated with them, which should have been bound.
  """

  def __init__(
      self,
      base_backward_transform: (
          backward_crown.LinearBoundBackwardConcretizingTransform),
      branching_decisions: branch_selection.JittableBranchingDecisions,
      lagrangian_variables: Tensor,
      concretization_fn: linear_relaxations.ConcretizationFn = (
          linear_relaxations.concretize_linear_expression),
  ):
    super().__init__(concretization_fn)
    self._base_backward_transform = base_backward_transform
    self._branching_decisions = branching_decisions
    self._lagrangian_variables = lagrangian_variables

  def concretize_args(self, primitive: Primitive) -> bool:
    return self._base_backward_transform.concretize_args(primitive)

  def primitive_backtransform(
      self,
      context: graph_traversal.TransformContext[
          linear_relaxations.LinearExpression],
      primitive: Primitive,
      eqn_outval: linear_relaxations.LinearExpression,
      *args: bound_propagation.LayerInput,
      **params,
  ) -> Sequence[Sequence[Optional[linear_relaxations.LinearExpression]]]:

    lay_idxs, neur_idxs, branch_val, is_upper = self._branching_decisions
    index = context.index
    max_jaxpr_depth = lay_idxs.shape[1]
    index_tensor = jnp.array(index + (0,) * (max_jaxpr_depth - len(index)))
    branching_in_this_layer = (lay_idxs == index_tensor).all(axis=1)
    active_branch_mask = branching_in_this_layer.astype(jnp.float32)

    # Include the lagrangian terms into the linear bounds that we are
    # propagating.
    # Note: I had an attempt at gating this function which is still relatively
    # expensive behind a jax.lax.cond, in case the active_branch_mask was all
    # false but this resulted in the whole bounding process being almost twice
    # as slow.
    lagrangianed_lincoeffs, lagrangianed_offset = _update_lin_with_lagrangian(
        eqn_outval.lin_coeffs, eqn_outval.offset,
        active_branch_mask, self._lagrangian_variables,
        neur_idxs, branch_val, is_upper)

    # This new linear expression is equivalent to the initial linear expression
    # except that it now also include the contribution of the lagrangian.
    lagrangianed_eqn_outval = linear_relaxations.LinearExpression(
        lagrangianed_lincoeffs, lagrangianed_offset)
    # We can now pass it on to the underlying primitive, to propagate backward.
    return self._base_backward_transform.primitive_backtransform(
        context, primitive, lagrangianed_eqn_outval, *args, **params)


def slope_and_lagrangian_optimizer(
    slope_opt: optax.GradientTransformation,
    lag_opt: optax.GradientTransformation,
) -> optax.GradientTransformation:
  """Creates an optimizer that handles the slopes and dual parameters.

  We want to optimize them differently.

  Args:
    slope_opt: Optax optimizer for the slope coefficients.
    lag_opt: Optax optimizer for the Lagrangian variable coefficients.
  Returns:
    optimizer: Optax optimizer for the combined set of params.
  """
  param_schema = ('slope_params', 'dual_params')
  opt_for_param = {'slope_params': slope_opt, 'dual_params': lag_opt}
  return optax.multi_transform(transforms=opt_for_param,
                               param_labels=param_schema)


class BranchedOptimizingLinearBoundBackwardTransform(
    backward_crown.OptimizingLinearBoundBackwardTransform):
  """Backward transform that concretize bounds through optimization.

  The optimization is done over both the relaxation parameters and the
  lagrangian variables associated with the branching constraints.
  """

  def __init__(
      self,
      branching_decisions: branch_selection.JittableBranchingDecisions,
      relaxer: linear_relaxations.ParameterizedLinearBoundsRelaxer,
      primitive_needs_concrete_bounds: Tuple[Primitive, ...],
      optimizer: optimizers.Optimizer,
      concretization_fn: linear_relaxations.ConcretizationFn = (
          linear_relaxations.concretize_linear_expression),
  ):
    """Constructs a per-node concretizer that performs an inner optimisation.

    This supports the addition of additional branching constraints.

    Args:
      branching_decisions: Branching decisions that needs to be enforced.
      relaxer: Specifies the parameterised linear relaxation to use for each
        primitive operation.
      primitive_needs_concrete_bounds: Which primitive operations need to be
        concretised.
      optimizer: Optimizer to use to compute the bound.
      concretization_fn: Function to use to concretize the linear bounds.
    """
    super().__init__(relaxer, primitive_needs_concrete_bounds, optimizer,
                     concretization_fn)
    self._branching_decisions = branching_decisions

  def _initial_params(
      self, scanner, input_bounds,
  ) -> Tuple[Mapping[Index, Tensor], Tensor]:
    slope_params = super()._initial_params(scanner, input_bounds)
    dual_vars = jnp.zeros(self._branching_decisions[0].shape[0])
    return slope_params, dual_vars

  def _project_params(self, scanner, unc_params):
    unc_slope_params, unc_dual_vars = unc_params
    slope_params = super()._project_params(scanner, unc_slope_params)
    dual_vars = jnp.maximum(unc_dual_vars, 0.)
    return slope_params, dual_vars

  def _bind(
      self,
      node_relaxations: Mapping[
          Index, linear_relaxations.ParameterizedNodeRelaxation],
      all_params: Tuple[Mapping[Index, Tensor], Tensor],
  ) -> backward_crown.LinearBoundBackwardConcretizingTransform:
    slope_params, dual_vars = all_params
    base_backward_transform = super()._bind(node_relaxations, slope_params)

    return ConstrainedLinearBoundBackwardTransform(
        base_backward_transform, self._branching_decisions, dual_vars,
        self._concretization_fn)


def lagrangian_backward_linear_compute_bounds(
    slope_optimizer: optax.GradientTransformation,
    lag_optimizer: optax.GradientTransformation,
    num_opt_steps: int,
    function: SpecFn,
    branching_decisions: branch_selection.JittableBranchingDecisions,
    *bounds: Nest[graph_traversal.GraphInput],
) -> Nest[bound_propagation.LayerInput]:
  """Performs bound computation in the style of Beta-Crown.

  https://arxiv.org/abs/2103.06624

  Args:
    slope_optimizer: Optax gradient transformation to use for optimizing the
      alpha (slope of the relaxation of non-linearities) parameters.
    lag_optimizer: Optax gradient transformation to use for optimizing the beta
      (lagrangian variables for the branching constraints) parameters.
    num_opt_steps: How many optimization steps to take for each bound
      computation.
    function: Function performing computation to obtain bounds for. Takes as
      only arguments the network inputs.
    branching_decisions: 4-tuple of tensors describing the branching
      constraints to impose. Detailed description at the top of the module.
    *bounds: Bounds on the input to the network.
  Returns:
    output_bound: Bounds on the output of the function.
  """
  parameterized_relaxer = linear_relaxations.parameterized_relaxer
  concretize_args_primitive = backward_crown.CONCRETIZE_ARGS_PRIMITIVE
  optimizer = optimizers.OptaxOptimizer(
      slope_and_lagrangian_optimizer(slope_optimizer, lag_optimizer),
      num_steps=num_opt_steps)

  backward_concretizer = concretization.ChunkedBackwardConcretizer(
      BranchedOptimizingLinearBoundBackwardTransform(
          branching_decisions, parameterized_relaxer, concretize_args_primitive,
          optimizer),
      max_chunk_size=128)
  backward_algorithm = concretization.BackwardConcretizingAlgorithm(
      backward_concretizer)

  output_bound, _ = bound_propagation.bound_propagation(
      backward_algorithm, function, *bounds)
  return output_bound
