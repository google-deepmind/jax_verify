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

"""Implementation of Square (https://arxiv.org/pdf/1912.00049)."""

from typing import Callable, List, Tuple

import chex
import jax
import jax.numpy as jnp

from jax_verify.extensions.functional_lagrangian.inner_solvers.pga import utils


def _schedule(values: List[float],
              boundaries: List[int],
              dtype=jnp.float32) -> Callable[[chex.Array], chex.Numeric]:
  """Schedule the value of p, the proportion of elements to be modified."""
  large_step = max(boundaries) + 1
  boundaries = boundaries + [large_step, large_step + 1]
  num_values = len(values)
  values = jnp.array(values, dtype=jnp.float32)
  large_step = jnp.array([large_step] * len(boundaries), dtype=jnp.int32)
  boundaries = jnp.array(boundaries, dtype=jnp.int32)

  def _get(step):
    """Returns the value according to the current step and schedule."""
    b = boundaries - jnp.minimum(step + 1, large_step + 1)
    b = jnp.where(b < 0, large_step, b)
    idx = jnp.minimum(jnp.argmin(b), num_values - 1)
    return values[idx].astype(dtype)

  return _get


class Square:
  """Performs a blackbox optimization as in https://arxiv.org/pdf/1912.00049."""

  def __init__(
      self,
      num_steps: int,
      epsilon: chex.Numeric,
      initialize_fn: utils.InitializeFn,
      bounds: Tuple[chex.ArrayTree, chex.ArrayTree],
  ):
    """Creates a Square attack."""
    self._num_steps = num_steps
    self._initialize_fn = initialize_fn
    self._project_fn = utils.linf_project_fn(epsilon=epsilon, bounds=bounds)
    self._epsilon = epsilon
    self._p_init = p = .8
    self._p_schedule = _schedule([
        p, p / 2, p / 4, p / 4, p / 8, p / 16, p / 32, p / 64, p / 128, p / 256,
        p / 512
    ], [10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000])

  def __call__(
      self,
      loss_fn: utils.LossFn,
      rng: chex.PRNGKey,
      x: chex.Array,
  ) -> chex.Array:
    if len(x.shape) != 4:
      raise ValueError(f'Unsupported tensor shape: {x.shape}')
    h, w, c = x.shape[1:]
    batch_size = x.shape[0]
    broadcast_shape = [batch_size] + [1] * (len(x.shape) - 1)
    min_size = 1

    def init_fn(rng):
      init_x = self._project_fn(self._initialize_fn(rng, x), x)
      init_loss = loss_fn(init_x)
      return init_x, init_loss

    def random_window_mask(rng, size, dtype):
      height_rng, width_rng = jax.random.split(rng)
      height_offset = jax.random.randint(
          height_rng,
          shape=(batch_size, 1, 1, 1),
          minval=0,
          maxval=h - size,
          dtype=jnp.int32)
      width_offset = jax.random.randint(
          width_rng,
          shape=(batch_size, 1, 1, 1),
          minval=0,
          maxval=w - size,
          dtype=jnp.int32)
      h_range = jnp.reshape(jnp.arange(h), [1, h, 1, 1])
      w_range = jnp.reshape(jnp.arange(w), [1, 1, w, 1])
      return jnp.logical_and(
          jnp.logical_and(height_offset <= h_range,
                          h_range < height_offset + size),
          jnp.logical_and(width_offset <= w_range,
                          w_range < width_offset + size)).astype(dtype)

    def random_linf_perturbation(rng, x, size):
      rng, perturbation_rng = jax.random.split(rng)
      perturbation = jax.random.randint(
          perturbation_rng, shape=(batch_size, 1, 1, c), minval=0,
          maxval=2) * 2 - 1
      return random_window_mask(rng, size, x.dtype) * perturbation

    def body_fn(i, loop_inputs):
      best_x, best_loss, rng = loop_inputs

      p = self._get_p(i)
      size = jnp.maximum(
          jnp.round(jnp.sqrt(p * h * w / c)).astype(jnp.int32), min_size)
      rng, next_rng = jax.random.split(rng)

      perturbation = random_linf_perturbation(next_rng, best_x, size)
      current_x = best_x + perturbation * self._epsilon

      current_x = self._project_fn(current_x, x)
      loss = loss_fn(current_x)

      cond = loss < best_loss
      best_x = jnp.where(jnp.reshape(cond, broadcast_shape), current_x, best_x)
      best_loss = jnp.where(cond, loss, best_loss)
      return best_x, best_loss, rng

    rng, next_rng = jax.random.split(rng)
    best_x, best_loss = init_fn(next_rng)
    loop_inputs = (best_x, best_loss, rng)
    return jax.lax.fori_loop(0, self._num_steps, body_fn, loop_inputs)[0]

  def _get_p(self, step):
    """Schedule on `p`."""
    step = step / self._num_steps * 10000.
    return self._p_schedule(jnp.array(step))
