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

"""Functions to elide the specification objective with the model."""

import dataclasses

import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import verify_utils
import ml_collections
import numpy as np

ConfigDict = ml_collections.ConfigDict
DataSpec = verify_utils.DataSpec
IntervalBound = jax_verify.IntervalBound
SpecType = verify_utils.SpecType
Tensor = jnp.array
LayerParams = verify_utils.LayerParams
ModelParams = verify_utils.ModelParams
ModelParamsElided = verify_utils.ModelParamsElided


def elide_adversarial_spec(
    params: ModelParams,
    data_spec: DataSpec,
) -> ModelParamsElided:
  """Elide params to have last layer merged with the adversarial objective.

  Args:
    params: parameters of the model under verification.
    data_spec: data specification.

  Returns:
    params_elided: elided parameters with the adversarial objective folded in
      the last layer (and bounds adapted accordingly).
  """

  def elide_fn(w_fin, b_fin):
    label_onehot = jnp.eye(w_fin.shape[-1])[data_spec.true_label]
    target_onehot = jnp.eye(w_fin.shape[-1])[data_spec.target_label]
    obj_orig = target_onehot - label_onehot
    obj_bp = jnp.matmul(w_fin, obj_orig)
    const = jnp.expand_dims(jnp.vdot(obj_orig, b_fin), axis=-1)
    obj = jnp.reshape(obj_bp, (obj_bp.size, 1))
    return obj, const

  last_params = params[-1]
  w_elided, b_elided = elide_fn(last_params.w, last_params.b)
  last_params_elided = verify_utils.FCParams(w_elided, b_elided)

  if last_params.has_bounds:
    w_bound_elided, b_bound_elided = jax_verify.interval_bound_propagation(
        elide_fn, last_params.w_bound, last_params.b_bound)
    last_params_elided = dataclasses.replace(
        last_params_elided, w_bound=w_bound_elided, b_bound=b_bound_elided)

  params_elided = params[:-1] + [last_params_elided]
  return params_elided


def elide_adversarial_softmax_spec(
    params: ModelParams,
    data_spec: DataSpec,
) -> ModelParamsElided:
  """Elide params to have uncertainty objective appended as a new last layer.

  Args:
    params: parameters of the model under verification.
    data_spec: data specification.

  Returns:
    params_elided: parameters with the uncertainty objective appended as
      the last 'layer'.
  """
  op_size = params[-1].w.shape[-1]
  e = np.zeros((op_size, 1))
  e[data_spec.target_label] = 1.
  e[data_spec.true_label] = -1.
  params_elided = params + [verify_utils.FCParams(jnp.array(e), jnp.zeros(()))]

  return params_elided


def elide_uncertainty_spec(
    params: ModelParams,
    data_spec: DataSpec,
    probability_threshold: float,
) -> ModelParamsElided:
  """Elide params to have uncertainty objective appended as a new last layer.

  Args:
    params: parameters of the model under verification.
    data_spec: data specification.
    probability_threshold: Maximum probability threshold for OOD detection.

  Returns:
    params_elided: parameters with the uncertainty objective appended as
      the last 'layer'.
  """
  op_size = params[-1].w.shape[-1]
  e = np.zeros((op_size, 1))
  e[data_spec.target_label] = 1.
  e -= probability_threshold
  params_elided = params + [verify_utils.FCParams(jnp.array(e), jnp.zeros(()))]

  return params_elided
