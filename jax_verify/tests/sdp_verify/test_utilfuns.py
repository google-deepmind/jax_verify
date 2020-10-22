# coding=utf-8
# Copyright 2020 The jax_verify Authors.
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

# Lint as: python3
"""Tests for cvxpy_verify.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import jax.random as random
from jax_verify.src.sdp_verify import utils
from jax_verify.tests.sdp_verify import test_utils


class ParamExtractionTest(parameterized.TestCase):
  """Test the functions extracting network parameters from functions."""

  def check_fun_extract(self, fun_to_extract, example_inputs):
    extracted_params = utils.get_layer_params(fun_to_extract, example_inputs)

    eval_original = fun_to_extract(example_inputs)
    eval_extracted = utils.predict_cnn(extracted_params, example_inputs)

    self.assertAlmostEqual(jnp.abs(eval_original - eval_extracted).max(), 0.0)

  def test_cnn_extract(self):
    """Test that weights from a CNN can be extracted."""
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)

    input_sizes = (1, 2, 2, 1)
    layer_sizes = [input_sizes, {
        'n_h': 2,
        'n_w': 2,
        'n_cout': 2,
        'padding': 'VALID',
        'stride': 1,
        'n_cin': 1
    }, 3]
    cnn_params = test_utils.make_cnn_params(layer_sizes, k1)

    fun_to_extract = functools.partial(utils.predict_cnn, cnn_params)
    example_inputs = random.normal(k2, input_sizes)

    self.check_fun_extract(fun_to_extract, example_inputs)

  def test_cnn_withpreproc(self):
    """Test extraction of weights from a CNN with input preprocessing."""
    key = random.PRNGKey(0)
    k1, k2, k3, k4 = random.split(key, num=4)

    input_sizes = (1, 2, 2, 3)
    layer_sizes = [input_sizes, {
        'n_h': 2,
        'n_w': 2,
        'n_cout': 2,
        'padding': 'VALID',
        'stride': 1,
        'n_cin': 3
    }, 3]
    cnn_params = test_utils.make_cnn_params(layer_sizes, k1)
    example_inputs = random.normal(k2, input_sizes)
    input_mean = random.normal(k3, (3,))
    input_std = random.normal(k4, (3,))

    def fun_to_extract(inputs):
      inp = (inputs - input_mean) / input_std
      return utils.predict_cnn(cnn_params, inp)

    self.check_fun_extract(fun_to_extract, example_inputs)

  def test_mlp_extract(self):
    """Test that weights from a MLP can be extracted."""
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)

    input_sizes = (5,)
    layer_sizes = (5, 8, 5)
    mlp_params = test_utils.make_mlp_params(layer_sizes, k1)

    fun_to_extract = functools.partial(utils.predict_mlp, mlp_params)
    example_inputs = random.normal(k2, input_sizes)
    self.check_fun_extract(fun_to_extract, example_inputs)

  def test_mlp_withpreproc(self):
    """Test extraction of weights from a MLP with input preprocessing."""
    key = random.PRNGKey(0)
    k1, k2, k3, k4 = random.split(key, num=4)

    input_sizes = (5,)
    layer_sizes = (5, 8, 5)
    mlp_params = test_utils.make_mlp_params(layer_sizes, k1)
    example_inputs = random.normal(k2, input_sizes)
    input_mean = random.normal(k3, input_sizes)
    input_std = random.normal(k4, input_sizes)

    def fun_to_extract(inputs):
      inp = (inputs - input_mean) / input_std
      return utils.predict_mlp(mlp_params, inp)

    self.check_fun_extract(fun_to_extract, example_inputs)


if __name__ == '__main__':
  absltest.main()
