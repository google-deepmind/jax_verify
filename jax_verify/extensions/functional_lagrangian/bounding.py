# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
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

"""Functions to obtain interval bounds on the activations."""

import dataclasses
import functools
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import specification
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import boundprop_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils
from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
import ml_collections


ConfigDict = ml_collections.ConfigDict
DataSpec = verify_utils.DataSpec
IntervalBound = jax_verify.IntervalBound
SpecType = verify_utils.SpecType
Tensor = jnp.array
LayerParams = verify_utils.LayerParams
ModelParams = verify_utils.ModelParams
ModelParamsElided = verify_utils.ModelParamsElided


def make_elided_params_and_bounds(
    config: ConfigDict,
    data_spec: DataSpec,
    spec_type: SpecType,
    params: ModelParams,
) -> Tuple[ModelParamsElided, Sequence[sdp_utils.IntBound], Tensor, float]:
  """Make the elided parameters and bounds according to the specification."""
  if spec_type in (verify_utils.SpecType.UNCERTAINTY,
                   verify_utils.SpecType.PROBABILITY_THRESHOLD):
    probability_threshold = 0.
    if spec_type == verify_utils.SpecType.PROBABILITY_THRESHOLD:
      probability_threshold = config.problem.probability_threshold
    params_elided = specification.elide_uncertainty_spec(
        params, data_spec, probability_threshold)
    # special care of indexing below since the spec layer has been added on top
    # of the logits layer in params_elided
    params_elided_boundprop = specification.elide_uncertainty_spec(
        params, data_spec, 0.)

    # Add specification layer to graph and perform bound propagation through
    # the whole model
    start_time = time.time()
    bounds = get_bounds(
        x=data_spec.input,
        epsilon=data_spec.epsilon,
        input_bounds=data_spec.input_bounds,
        params=params_elided_boundprop,
        config=config)
    bp_bound = upper_bound_log_softmax(
        bounds[-3].lb,
        bounds[-3].ub,  # bounds on inputs of logits layer
        params_elided_boundprop[-2],  # logits layer parameters
        data_spec.target_label)
    elapsed_time = time.time() - start_time
  elif spec_type == spec_type.ADVERSARIAL_SOFTMAX:
    start_time = time.time()
    params_elided = specification.elide_adversarial_softmax_spec(
        params, data_spec)

    bounds = get_bounds(
        x=data_spec.input,
        epsilon=data_spec.epsilon,
        input_bounds=data_spec.input_bounds,
        params=params_elided,
        config=config)

    # upper bound on difference between target logits and true logits
    bp_bound = (
        bounds[-2].ub[:, data_spec.target_label] -
        bounds[-2].lb[:, data_spec.true_label])
    elapsed_time = time.time() - start_time

  else:
    params_elided = specification.elide_adversarial_spec(params, data_spec)
    start_time = time.time()
    bounds = get_bounds(
        x=data_spec.input,
        epsilon=data_spec.epsilon,
        input_bounds=data_spec.input_bounds,
        params=params_elided,
        config=config)
    elapsed_time = time.time() - start_time
    bp_bound = bounds[-1].ub_pre

  return params_elided, bounds, bp_bound, elapsed_time


def _make_all_act_fn(params: ModelParams):
  """Make forward function."""

  def all_act_fn(
      inputs: Tensor,
      *given_params: Sequence[Tuple[Tensor, Tensor]],
  ) -> Tensor:
    given_params = iter(given_params)

    net_params = []
    for layer_params in params:
      if layer_params.has_bounds:
        w, b = next(given_params)
        kwargs = {}
        if layer_params.w_bound is not None:
          kwargs['w'] = w
        if layer_params.b_bound is not None:
          kwargs['b'] = b
        layer_params = dataclasses.replace(layer_params, **kwargs)
      net_params.append(layer_params)
    return sdp_utils.predict_cnn(
        net_params, inputs, include_preactivations=True)

  return all_act_fn


def _compute_jv_bounds(
    input_bound: sdp_utils.IntBound,
    params: ModelParams,
    method: str,
) -> List[sdp_utils.IntBound]:
  """Compute bounds with jax_verify."""

  jv_input_bound = jax_verify.IntervalBound(input_bound.lb, input_bound.ub)

  # create a function that takes as arguments the input and all parameters
  # that have bounds (as specified in param_bounds) and returns all
  # activations
  all_act_fun = _make_all_act_fn(params)

  # use jax_verify to perform (bilinear) interval bound propagation
  jv_param_bounds = [(p.w_bound, p.b_bound) for p in params if p.has_bounds]

  if method == 'ibp':
    _, jv_bounds = jax_verify.interval_bound_propagation(
        all_act_fun, jv_input_bound, *jv_param_bounds)
  elif method == 'fastlin':
    _, jv_bounds = jax_verify.forward_fastlin_bound_propagation(
        all_act_fun, jv_input_bound, *jv_param_bounds)
  elif method == 'ibpfastlin':
    _, jv_bounds = jax_verify.ibpforwardfastlin_bound_propagation(
        all_act_fun, jv_input_bound, *jv_param_bounds)
  elif method == 'crown':
    _, jv_bounds = jax_verify.backward_crown_bound_propagation(
        all_act_fun, jv_input_bound, *jv_param_bounds)
  elif method == 'nonconvex':
    _, jv_bounds = jax_verify.nonconvex_constopt_bound_propagation(
        all_act_fun, jv_input_bound, *jv_param_bounds)
  else:
    raise ValueError('Unsupported method.')

  # re-format bounds with internal convention
  bounds = []
  for intermediate_bound in jv_bounds:
    bounds.append(
        sdp_utils.IntBound(
            lb_pre=intermediate_bound.lower,
            ub_pre=intermediate_bound.upper,
            lb=jnp.maximum(intermediate_bound.lower, 0),
            ub=jnp.maximum(intermediate_bound.upper, 0)))

  return bounds


def _compute_standard_bounds(
    x: Tensor,
    epsilon: float,
    input_bounds: Sequence[int],
    params: ModelParams,
    config: Union[ConfigDict, Dict[str, Any]],
):
  """Perform bound-propagation and return bounds.

  Args:
    x: input to the model under verification.
    epsilon: radius of l-infinity ball around x.
    input_bounds: feasibility bounds of inputs (e.g. [0, 1]).
    params: parameters of the model under verification.
    config: experiment ConfigDict.

  Returns:
    List of bounds per layer, including the input bounds as the first element.
  """

  for param in params:
    if param.has_bounds:
      raise ValueError('Unsupported bilinear bound propagation.')

  if config['boundprop_type'] == 'nonconvex':
    bounds = boundprop_utils.boundprop(
        params,
        jnp.expand_dims(x, axis=0),
        epsilon,
        input_bounds,
        'nonconvex',
        nonconvex_boundprop_steps=config['nonconvex_boundprop_steps'],
        nonconvex_boundprop_nodes=config['nonconvex_boundprop_nodes'],
    )
  elif config['boundprop_type'] == 'crown_ibp':
    bounds = boundprop_utils.boundprop(params, jnp.expand_dims(x, axis=0),
                                       epsilon, input_bounds, 'crown_ibp')
  else:
    # initial bounds for boundprop
    init_bounds = sdp_utils.init_bound(x, epsilon, input_bounds=input_bounds)
    bounds = [init_bounds] + _compute_jv_bounds(
        input_bound=init_bounds, params=params, method=config['boundprop_type'])

  return bounds


def get_bounds(
    x: Tensor,
    epsilon: float,
    input_bounds: Sequence[int],
    params: ModelParams,
    config: Union[ConfigDict, Dict[str, Any]],
) -> List[sdp_utils.IntBound]:
  """Perform bound-propagation and return bounds.

  The code assumes that the sequential model can be split into two parts. The
  first part (potentially empty) does not contain any bound on the parameters
  and can thus use boundprop as usual. The second part (potentially empty)
  contains parameter bounds and thus employs a method that supports bilinear
  bound propagation.

  Args:
    x: input to the model under verification.
    epsilon: radius of l-infinity ball around x.
    input_bounds: feasibility bounds of inputs (e.g. [0, 1]).
    params: parameters of the model under verification.
    config: experiment ConfigDict.

  Returns:
    List of bounds per layer, including the input bounds as the first element.
  """
  if config['boundprop_type'] != config['bilinear_boundprop_type']:
    # when using a different boundprop method for bilinear operations, partition
    # parameters used for "standard" boundprop vs bilinear boundprop
    first_idx_with_param_bounds = 0
    for param in params:
      if param.has_bounds:
        break
      first_idx_with_param_bounds += 1

    params_standard_boundprop, params_bilinear_boundprop = (
        params[:first_idx_with_param_bounds],
        params[first_idx_with_param_bounds:])
  else:
    params_standard_boundprop = []
    params_bilinear_boundprop = params

  if params_standard_boundprop:
    bounds_standard = _compute_standard_bounds(
        x=x,
        epsilon=epsilon,
        input_bounds=input_bounds,
        params=params_standard_boundprop,
        config=config,
    )
  else:
    bounds_standard = [
        sdp_utils.init_bound(x, epsilon, input_bounds=input_bounds)
    ]

  if params_bilinear_boundprop:
    bounds_bilinear = _compute_jv_bounds(
        input_bound=bounds_standard[-1],
        params=params_bilinear_boundprop,
        method=config['bilinear_boundprop_type'],
    )
  else:
    bounds_bilinear = []

  return bounds_standard + bounds_bilinear


class BoundsFromCnn(bound_propagation.BoundTransform):
  """Precomputed bounds from a sequential CNN."""

  def __init__(self, bounds: Sequence[sdp_utils.IntBound]):
    self._cnn_bounds = bounds
    self._cnn_layer_indices = {}

  def input_transform(self, context, lower_bound, upper_bound):
    if context.index not in self._cnn_layer_indices:
      self._cnn_layer_indices[context.index] = 0, False
    return self._bounds_from_cnn_layer(context.index)

  def primitive_transform(self, context, primitive, *args, **kwargs):
    if context.index not in self._cnn_layer_indices:
      layer_index, was_preact = list(self._cnn_layer_indices.values())[-1]

      if not was_preact:
        # Previous op was a ReLU. Move into a new layer.
        layer_index += 1
      is_preact = primitive != synthetic_primitives.relu_p

      self._cnn_layer_indices[context.index] = layer_index, is_preact

    return self._bounds_from_cnn_layer(context.index)

  def _bounds_from_cnn_layer(self, index):
    layer_index, is_preact = self._cnn_layer_indices[index]
    if is_preact:
      return jax_verify.IntervalBound(self._cnn_bounds[layer_index].lb_pre,
                                      self._cnn_bounds[layer_index].ub_pre)
    else:
      return jax_verify.IntervalBound(self._cnn_bounds[layer_index].lb,
                                      self._cnn_bounds[layer_index].ub)


def _get_reciprocal_bound(l: jnp.array, u: jnp.array,
                          logits_params: LayerParams, label: int) -> jnp.array:
  """Helped for computing bound on label softmax given interval bounds on pre logits."""

  def fwd(x, w, b):
    wdiff = jnp.reshape(w[:, label], [-1, 1]) - w
    bdiff = b[label] - b
    return x @ wdiff + bdiff

  x_bound = jax_verify.IntervalBound(
      lower_bound=jnp.reshape(l, [l.shape[0], -1]),
      upper_bound=jnp.reshape(u, [u.shape[0], -1]))

  params_bounds = []
  if logits_params.w_bound is None:
    fwd = functools.partial(fwd, w=logits_params.w)
  else:
    params_bounds.append(logits_params.w_bound)

  if logits_params.b_bound is None:
    fwd = functools.partial(fwd, b=logits_params.b)
  else:
    params_bounds.append(logits_params.b_bound)

  fwd_bound = jax_verify.interval_bound_propagation(fwd, x_bound,
                                                    *params_bounds)

  return fwd_bound


def upper_bound_log_softmax(
    l: Tensor,
    u: Tensor,
    logits_params: LayerParams,
    target_label: int,
) -> Tensor:
  """Get bound on target label softmax given interval bounds on pre logits.

  Args:
    l: Array of lower bounds on pre-logits layer.
    u: Array of upper bounds on pre-logits layer.
    logits_params: parameters of the final logits layer.
    target_label: Target label whose softmax we want to bound.

  Returns:
    Upper bound on log softmax of target label.
  """
  fwd_bound = _get_reciprocal_bound(l, u, logits_params, target_label)
  return -jax.nn.logsumexp(-fwd_bound.upper)
