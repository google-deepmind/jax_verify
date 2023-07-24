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

"""Optimizers used in the PGA strategy."""

import collections
from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jax_verify.extensions.functional_lagrangian.inner_solvers.pga import utils

_State = collections.namedtuple('State', ['iteration', 'rng', 'state'])  # pylint: disable=invalid-name


def grad_fn(
    loss_fn: utils.LossFn,
) -> Callable[[chex.Array], Tuple[chex.Array, chex.Array]]:
  """Returns the analytical gradient as computed by `jax.grad`."""

  def reduced_loss_fn(x):
    loss = loss_fn(x)
    return jnp.sum(loss), loss

  return jax.grad(reduced_loss_fn, has_aux=True)


class IteratedFGSM:
  """L-infinity normalized steps."""

  def __init__(self, learning_rate: chex.Numeric):
    self._learning_rate = learning_rate

  def init(self, loss_fn: utils.LossFn, rng: chex.PRNGKey,
           x: chex.Array) -> _State:
    del x
    self._loss_fn = loss_fn
    return _State(jnp.array(0, dtype=jnp.int32), rng, ())

  def minimize(self, x: chex.Array,
               state: _State) -> Tuple[chex.Array, chex.Array, _State]:
    """Performs a single minimization step."""
    lr = jnp.array(self._learning_rate)
    g, loss = grad_fn(self._loss_fn)(x)
    if g is None:
      raise ValueError('loss_fn does not depend on input.')
    g = jnp.sign(g)
    g, s = self._update(lr, g, state.state)
    new_state = _State(state.iteration + 1, state.rng, s)
    return x - g, loss, new_state

  def _update(
      self,
      learning_rate: chex.Numeric,
      gradients: chex.Array,
      state: chex.Array,
  ) -> Tuple[chex.Array, chex.Array]:
    return learning_rate.astype(gradients.dtype) * gradients, state  # pytype: disable=attribute-error  # numpy-scalars


class PGD:
  """Uses the above defined optimizers to minimize and loss function."""

  def __init__(
      self,
      optimizer,
      num_steps: int,
      initialize_fn: Optional[utils.InitializeFn] = None,
      project_fn: Optional[utils.ProjectFn] = None,
  ):
    self._optimizer = optimizer
    if initialize_fn is None:
      initialize_fn = lambda rng, x: x
    self._initialize_fn = initialize_fn
    if project_fn is None:
      project_fn = lambda x, origin_x: x
    self._project_fn = project_fn
    self._num_steps = num_steps

  def __call__(
      self,
      loss_fn: utils.LossFn,
      rng: chex.PRNGKey,
      x: chex.Array,
  ) -> chex.Array:

    def _optimize(rng, x):
      """Optimizes loss_fn."""

      def body_fn(_, inputs):
        opt_state, current_x = inputs
        current_x, _, opt_state = self._optimizer.minimize(current_x, opt_state)
        current_x = self._project_fn(current_x, x)
        return opt_state, current_x

      rng, next_rng = jax.random.split(rng)
      opt_state = self._optimizer.init(loss_fn, next_rng, x)
      current_x = self._project_fn(self._initialize_fn(rng, x), x)
      _, current_x = jax.lax.fori_loop(0, self._num_steps, body_fn,
                                       (opt_state, current_x))
      return current_x

    x = _optimize(rng, x)
    return jax.lax.stop_gradient(x)


class Restarted:
  """Repeats an optimization multiple times."""

  def __init__(
      self,
      optimizer,
      restarts_using_tiling: int = 1,
      has_batch_dim: bool = True,
  ):
    self._wrapped_optimizer = optimizer
    if (isinstance(restarts_using_tiling, int) and restarts_using_tiling > 1 and
        not has_batch_dim):
      raise ValueError('Cannot use tiling when `has_batch_dim` is False.')
    self._has_batch_dim = has_batch_dim
    if (isinstance(restarts_using_tiling, int) and restarts_using_tiling < 1):
      raise ValueError('Fewer than one restart requested.')
    self._restarts_using_tiling = restarts_using_tiling

  def __call__(
      self,
      loss_fn: utils.LossFn,
      rng: chex.PRNGKey,
      inputs: chex.Array,
  ) -> chex.Array:
    """Performs an optimization multiple times by tiling the inputs."""
    if not self._has_batch_dim:
      opt_inputs = self._wrapped_optimizer(loss_fn, rng, inputs)
      opt_losses = loss_fn(opt_inputs)
      return opt_inputs, opt_losses  # pytype: disable=bad-return-type  # numpy-scalars

    # Tile the inputs and labels.
    batch_size = inputs.shape[0]

    # Tile inputs.
    shape = inputs.shape[1:]
    # Shape is [num_restarts * batch_size, ...].
    inputs = jnp.tile(inputs, [self._restarts_using_tiling] + [1] * len(shape))

    # Optimize.
    opt_inputs = self._wrapped_optimizer(loss_fn, rng, inputs)
    opt_losses = loss_fn(opt_inputs)
    opt_losses = jnp.reshape(opt_losses,
                             [self._restarts_using_tiling, batch_size])

    # Extract best.
    i = jnp.argmin(opt_losses, axis=0)
    j = jnp.arange(batch_size)

    shape = opt_inputs.shape[1:]
    return jnp.reshape(opt_inputs,
                       (self._restarts_using_tiling, batch_size) + shape)[i, j]
