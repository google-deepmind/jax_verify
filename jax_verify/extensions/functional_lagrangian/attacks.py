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

"""Adversarial attacks."""

import dataclasses
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import optax


IntervalBound = jax_verify.IntervalBound
Tensor = jnp.DeviceArray
PRNGKey = jnp.DeviceArray
DataSpec = verify_utils.DataSpec
LayerParams = verify_utils.LayerParams
ModelParams = verify_utils.ModelParams
ModelParamsElided = verify_utils.ModelParamsElided


def sample_truncated_normal(
    mean: Tensor,
    std: Tensor,
    bounds: IntervalBound,
    prng_key: PRNGKey
) -> Tensor:
  """Draw sample from truncated normal distribution."""
  rescaled_lower = (bounds.lower - mean) / std
  rescaled_upper = (bounds.upper - mean) / std

  unit_noise = jax.random.truncated_normal(
      prng_key, lower=rescaled_lower, upper=rescaled_upper,
      shape=mean.shape, dtype=mean.dtype)

  return mean + unit_noise * std


def sample_dropout(
    mean: Tensor,
    dropout_rate: float,
    prng_key: PRNGKey,
) -> Tensor:
  """Draw sample from dropout."""
  if mean.ndim != 2:
    raise ValueError(
        f'Dropout only supports 2D parameters (found {mean.ndim} instead).')

  retention_rate = 1.0 - dropout_rate

  # reconstruct initial parameter by reverting expectation
  initial_parameter = mean / retention_rate

  # in "parameter space", dropping an input correspond to dropping an entire row
  retention_mask_per_input = jax.random.bernoulli(
      prng_key, p=retention_rate, shape=[mean.shape[0]])
  retention_mask = jnp.expand_dims(retention_mask_per_input, 1)

  return initial_parameter * retention_mask


def make_params_sampling_fn(
    params: Union[ModelParams, ModelParamsElided],
) -> Callable[[PRNGKey], Union[ModelParams, ModelParamsElided]]:
  """Make function that samples new parameters at each call."""

  def sample_fn(key: PRNGKey):
    sampled_params = []
    for layer_params in params:
      if layer_params.w_std is not None:
        assert layer_params.dropout_rate == 0.0
        key, key_w = jax.random.split(key)
        w_sampled = sample_truncated_normal(
            mean=layer_params.w,
            std=layer_params.w_std,
            bounds=layer_params.w_bound,
            prng_key=key_w,
        )
        layer_params = dataclasses.replace(layer_params, w=w_sampled)
      elif layer_params.dropout_rate > 0.0:
        key, key_dropout = jax.random.split(key)
        w_sampled = sample_dropout(
            mean=layer_params.w,
            dropout_rate=layer_params.dropout_rate,
            prng_key=key_dropout,
        )
        layer_params = dataclasses.replace(layer_params, w=w_sampled)
      if layer_params.b_std is not None:
        key, key_b = jax.random.split(key)
        b_sampled = sample_truncated_normal(
            mean=layer_params.b,
            std=layer_params.b_std,
            bounds=layer_params.b_bound,
            prng_key=key_b,
        )
        layer_params = dataclasses.replace(layer_params, b=b_sampled)
      sampled_params.append(layer_params)
    return sampled_params

  return sample_fn


def make_forward(
    model_params: ModelParams,
    num_samples: int,
) -> Callable[[Tensor, PRNGKey], Tensor]:
  """Make forward_fn with parameter sampling and averaging in softmax space.

  Args:
    model_params: model parameters.
    num_samples: number of samples drawn per call to forward_fn.

  Returns:
    function that draws parameter samples, averages their results in softmax
      space and takes the log.
  """

  sampling_fn = make_params_sampling_fn(model_params)

  def single_forward(inputs, prng_key):
    sampled_params = sampling_fn(prng_key)
    logits = sdp_utils.predict_cnn(
        sampled_params, jnp.expand_dims(inputs, axis=0))
    return jax.nn.log_softmax(logits)

  def multiple_forward(inputs, prng_key):
    different_keys = jax.random.split(prng_key, num_samples)
    sampled_logits = jax.vmap(single_forward, in_axes=[None, 0])(
        inputs, different_keys)
    sampled_softmax = jax.nn.softmax(sampled_logits)
    averaged_softmax = jnp.mean(sampled_softmax, 0)
    return jnp.log(averaged_softmax)

  if num_samples == 1:
    return single_forward
  else:
    return multiple_forward


def _run_attack(
    max_objective_fn: Callable[[Tensor, PRNGKey], Tensor],
    projection_fn: Callable[[Tensor], Tensor],
    x_init: Tensor,
    prng_key: PRNGKey,
    num_steps: int,
    learning_rate: float,
):
  """Run attack."""

  opt = optax.chain(optax.scale(-1),  # maximization
                    optax.adam(learning_rate))
  grad_fn = jax.grad(max_objective_fn)

  def body_fn(it, inputs):
    del it  # unused
    x, prng_in, opt_state = inputs
    prng_out, prng_used = jax.random.split(prng_in)
    grad_x = grad_fn(x, prng_used)
    updates, opt_state = opt.update(grad_x, opt_state, x)
    x = optax.apply_updates(x, updates)
    x = projection_fn(x)
    return x, prng_out, opt_state

  opt_state = opt.init(x_init)
  init_state = (x_init, prng_key, opt_state)
  x, prng_final, _ = jax.lax.fori_loop(0, num_steps, body_fn, init_state)

  return max_objective_fn(x, prng_final)


def adversarial_attack(
    params: ModelParams,
    data_spec: DataSpec,
    spec_type: verify_utils.SpecType,
    key: PRNGKey,
    num_steps: int,
    learning_rate: float,
    num_samples: int = 1,
) -> float:
  """Adversarial attack on uncertainty spec (with parameter sampling)."""
  l = jnp.clip(data_spec.input-data_spec.epsilon,
               data_spec.input_bounds[0], data_spec.input_bounds[1])
  u = jnp.clip(data_spec.input+data_spec.epsilon,
               data_spec.input_bounds[0], data_spec.input_bounds[1])
  projection_fn = lambda x: jnp.clip(x, l, u)

  forward_fn = make_forward(params, num_samples)

  def max_objective_fn_uncertainty(x, prng_key):
    logits = jnp.reshape(forward_fn(x, prng_key), [-1])
    return logits[data_spec.target_label]

  def max_objective_fn_adversarial(x, prng_key):
    logits = jnp.reshape(forward_fn(x, prng_key), [-1])
    return logits[data_spec.target_label] - logits[data_spec.true_label]

  def max_objective_fn_adversarial_softmax(x, prng_key):
    logits = jnp.reshape(forward_fn(x, prng_key), [-1])
    probs = jax.nn.softmax(logits, axis=-1)
    return probs[data_spec.target_label] - probs[data_spec.true_label]

  if (spec_type in (verify_utils.SpecType.UNCERTAINTY,
                    verify_utils.SpecType.PROBABILITY_THRESHOLD)):
    max_objective_fn = max_objective_fn_uncertainty
  elif spec_type == verify_utils.SpecType.ADVERSARIAL:
    max_objective_fn = max_objective_fn_adversarial
  elif spec_type == verify_utils.SpecType.ADVERSARIAL_SOFTMAX:
    max_objective_fn = max_objective_fn_adversarial_softmax
  else:
    raise ValueError('Unsupported spec.')

  return _run_attack(
      max_objective_fn=max_objective_fn,
      projection_fn=projection_fn,
      x_init=data_spec.input,
      prng_key=key,
      num_steps=num_steps,
      learning_rate=learning_rate)
