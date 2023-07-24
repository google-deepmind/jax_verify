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

"""Elementary tests for the Lagrangian forms."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
from jax_verify.extensions.functional_lagrangian import lagrangian_form

INPUT_SHAPES = (('batched_0d', [1]), ('batched_1d', [1, 2]),
                ('batched_2d', [1, 2, 3]), ('batched_3d', [1, 2, 3, 4]))


class ShapeTest(chex.TestCase):

  def setUp(self):
    super(ShapeTest, self).setUp()
    self._prng_seq = hk.PRNGSequence(13579)

  def _assert_output_shape(self, form, shape):
    x = jax.random.normal(next(self._prng_seq), shape)
    params = form.init_params(next(self._prng_seq), x.shape[1:])
    out = form.apply(x, params, step=0)
    assert out.ndim == 1

  @parameterized.named_parameters(*INPUT_SHAPES)
  def test_linear(self, shape):
    form = lagrangian_form.Linear()
    self._assert_output_shape(form, shape)

  @parameterized.named_parameters(*INPUT_SHAPES)
  def test_linear_exp(self, shape):
    form = lagrangian_form.LinearExp()
    self._assert_output_shape(form, shape)


if __name__ == '__main__':
  absltest.main()
