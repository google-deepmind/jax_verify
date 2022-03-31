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

"""Attacks test."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import attacks
from jax_verify.extensions.functional_lagrangian import verify_utils

EPS = 0.1


def make_data_spec(prng_key):
  """Create data specification from config."""
  x = jax.random.normal(prng_key, [8])
  input_bounds = (x - EPS, x + EPS)

  return verify_utils.DataSpec(
      input=x,
      true_label=0,
      target_label=1,
      epsilon=EPS,
      input_bounds=input_bounds)


def make_params(prng_key, dropout_rate=0.0, std=None):
  prng_key_seq = hk.PRNGSequence(prng_key)

  w1 = jax.random.normal(next(prng_key_seq), [8, 4])
  b1 = jax.random.normal(next(prng_key_seq), [4])

  w2 = jax.random.normal(next(prng_key_seq), [4, 2])
  b2 = jax.random.normal(next(prng_key_seq), [2])

  if std is not None:
    w1_std = std * jnp.ones([8, 4])
    b1_std = std * jnp.ones([4])
    w1_bound = jax_verify.IntervalBound(w1 - 3 * w1_std, w1 + 3 * w1_std)
    b1_bound = jax_verify.IntervalBound(b1 - 3 * b1_std, b1 + 3 * b1_std)
  else:
    w1_std, b1_std, w1_bound, b1_bound = None, None, None, None

  params = [
      verify_utils.FCParams(
          w=w1,
          b=b1,
          w_std=w1_std,
          b_std=b1_std,
          w_bound=w1_bound,
          b_bound=b1_bound,
      ),
      verify_utils.FCParams(
          w=w2,
          b=b2,
          dropout_rate=dropout_rate,
      )
  ]
  return params


class AttacksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.prng_seq = hk.PRNGSequence(1234)
    self.data_spec = make_data_spec(next(self.prng_seq))

  def test_forward_deterministic(self):
    params = make_params(next(self.prng_seq))
    self._check_deterministic_behavior(params)

  def test_forward_almost_no_randomness(self):
    params = make_params(next(self.prng_seq), std=1e-8, dropout_rate=1e-8)
    self._check_deterministic_behavior(params)

  def test_forward_gaussian(self):
    params = make_params(next(self.prng_seq), std=1.0)
    self._check_stochastic_behavior(params)

  def test_forward_dropout(self):
    params = make_params(next(self.prng_seq), dropout_rate=0.8)
    self._check_stochastic_behavior(params)

  def test_adversarial_integration(self):
    spec_type = verify_utils.SpecType.ADVERSARIAL
    params = make_params(next(self.prng_seq), std=0.1, dropout_rate=0.2)
    attacks.adversarial_attack(
        params,
        self.data_spec,
        spec_type,
        next(self.prng_seq),
        num_steps=5,
        learning_rate=0.1,
        num_samples=3)

  def test_adversarial_uncertainty_integration(self):
    spec_type = verify_utils.SpecType.ADVERSARIAL
    params = make_params(next(self.prng_seq), std=0.1, dropout_rate=0.2)
    attacks.adversarial_attack(
        params,
        self.data_spec,
        spec_type,
        next(self.prng_seq),
        num_steps=5,
        learning_rate=0.1,
        num_samples=3)

  def _make_value_and_grad(self, params, num_samples):
    forward_fn = attacks.make_forward(params, num_samples)

    def objective_fn(x, prng_key):
      out = jnp.reshape(forward_fn(x, prng_key), [2])
      return out[1] - out[0]

    return jax.value_and_grad(objective_fn)

  def _check_deterministic_behavior(self, params):
    # build function with 1 sample
    value_and_grad_fn = self._make_value_and_grad(params, num_samples=1)
    # forward first time
    out_1 = value_and_grad_fn(self.data_spec.input, next(self.prng_seq))

    # forward again gives the same result
    out_1_again = value_and_grad_fn(self.data_spec.input, next(self.prng_seq))
    chex.assert_tree_all_close(out_1, out_1_again, rtol=1e-5)

    # forward with 3 samples should still give the same result
    value_and_grad_fn = self._make_value_and_grad(params, num_samples=3)
    out_3 = value_and_grad_fn(self.data_spec.input, next(self.prng_seq))
    chex.assert_tree_all_close(out_3, out_1, rtol=1e-5)

  def _check_stochastic_behavior(self, params):
    value_and_grad_fn = self._make_value_and_grad(params, num_samples=2)
    prng = next(self.prng_seq)

    # forward a first time
    out_2 = value_and_grad_fn(self.data_spec.input, prng)

    # forward with a different seed does not give the same result
    out_2_diff = value_and_grad_fn(self.data_spec.input, next(self.prng_seq))
    with self.assertRaises(AssertionError):
      chex.assert_tree_all_close(out_2, out_2_diff)

    # forward with 3 samples and the same prng is not the same
    value_and_grad_fn = self._make_value_and_grad(params, num_samples=3)
    out_3_same_prng = value_and_grad_fn(self.data_spec.input, prng)
    with self.assertRaises(AssertionError):
      chex.assert_tree_all_close(out_2, out_3_same_prng)


if __name__ == '__main__':
  absltest.main()
