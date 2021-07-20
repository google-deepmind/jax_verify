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

"""Tests for propagating bounds through the networks defined in the Model Zoo.

We do not perform any check on the returned values but simply ensure that the
bound propagation can be performed on those networks.
"""
import pickle
from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
from jax_verify.src.mip_solver import cvxpy_relaxation_solver
from jax_verify.src.mip_solver import relaxation
from jax_verify.tests import model_zoo
import numpy as np


class ModelZooModelTests(parameterized.TestCase):

  @parameterized.named_parameters(
      ('SmallResidualModel', model_zoo.SmallResidualModel),
      ('TinyModel', model_zoo.TinyModel)
  )
  def test_ibp(self, model_cls):

    @hk.transform_with_state
    def model_pred(inputs, is_training, test_local_stats=False):
      model = model_cls()
      return model(inputs, is_training, test_local_stats)

    inps = jnp.zeros((4, 28, 28, 1), dtype=jnp.float32)
    params, state = model_pred.init(jax.random.PRNGKey(42), inps,
                                    is_training=True)

    def logits_fun(inputs):
      return model_pred.apply(params, state, None, inputs,
                              False, test_local_stats=False)[0]

    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)
    jax_verify.interval_bound_propagation(logits_fun, input_bounds)

  @parameterized.named_parameters(
      ('SmallResidualModel', model_zoo.SmallResidualModel),
      ('TinyModel', model_zoo.TinyModel)
  )
  def test_fastlin(self, model_cls):

    @hk.transform_with_state
    def model_pred(inputs, is_training, test_local_stats=False):
      model = model_cls()
      return model(inputs, is_training, test_local_stats)

    inps = jnp.zeros((4, 28, 28, 1), dtype=jnp.float32)
    params, state = model_pred.init(jax.random.PRNGKey(42), inps,
                                    is_training=True)

    def logits_fun(inputs):
      return model_pred.apply(params, state, None, inputs,
                              False, test_local_stats=False)[0]

    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)
    jax_verify.forward_fastlin_bound_propagation(logits_fun, input_bounds)

  @parameterized.named_parameters(
      ('SmallResidualModel', model_zoo.SmallResidualModel),
      ('TinyModel', model_zoo.TinyModel)
  )
  def test_ibpfastlin(self, model_cls):

    @hk.transform_with_state
    def model_pred(inputs, is_training, test_local_stats=False):
      model = model_cls()
      return model(inputs, is_training, test_local_stats)

    inps = jnp.zeros((4, 28, 28, 1), dtype=jnp.float32)
    params, state = model_pred.init(jax.random.PRNGKey(42), inps,
                                    is_training=True)

    def logits_fun(inputs):
      return model_pred.apply(params, state, None, inputs,
                              False, test_local_stats=False)[0]

    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)
    jax_verify.ibpforwardfastlin_bound_propagation(logits_fun, input_bounds)

  @parameterized.named_parameters(
      ('SmallResidualModel', model_zoo.SmallResidualModel),
      ('TinyModel', model_zoo.TinyModel)
  )
  def test_crownibp(self, model_cls):

    @hk.transform_with_state
    def model_pred(inputs, is_training, test_local_stats=False):
      model = model_cls()
      return model(inputs, is_training, test_local_stats)

    inps = jnp.zeros((4, 28, 28, 1), dtype=jnp.float32)
    params, state = model_pred.init(jax.random.PRNGKey(42), inps,
                                    is_training=True)

    def logits_fun(inputs):
      return model_pred.apply(params, state, None, inputs,
                              False, test_local_stats=False)[0]

    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)
    jax_verify.crownibp_bound_propagation(logits_fun, input_bounds)

  @parameterized.named_parameters(
      ('SmallResidualModel', model_zoo.SmallResidualModel),
      ('TinyModel', model_zoo.TinyModel))
  def test_nonconvex(self, model_cls):

    @hk.transform_with_state
    def model_pred(inputs, is_training, test_local_stats=False):
      model = model_cls()
      return model(inputs, is_training, test_local_stats)

    inps = jnp.zeros((4, 28, 28, 1), dtype=jnp.float32)
    params, state = model_pred.init(jax.random.PRNGKey(42), inps,
                                    is_training=True)

    def logits_fun(inputs):
      return model_pred.apply(params, state, None, inputs,
                              False, test_local_stats=False)[0]

    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)
    # Test with IBP for intermediate bounds
    jax_verify.nonconvex_ibp_bound_propagation(logits_fun, input_bounds)

    # Test with nonconvex bound evaluation for intermediate bounds
    jax_verify.nonconvex_constopt_bound_propagation(logits_fun, input_bounds)

  @parameterized.named_parameters(
      ('SmallResidualModel', model_zoo.SmallResidualModel),
      ('TinyModel', model_zoo.TinyModel))
  def test_cvxpy_relaxation(self, model_cls):

    @hk.transform_with_state
    def model_pred(inputs, is_training, test_local_stats=False):
      model = model_cls()
      return model(inputs, is_training, test_local_stats)

    inps = jnp.zeros((4, 28, 28, 1), dtype=jnp.float32)
    params, state = model_pred.init(jax.random.PRNGKey(42), inps,
                                    is_training=True)

    def logits_fun(inputs):
      return model_pred.apply(params, state, None, inputs,
                              False, test_local_stats=False)[0]

    output = logits_fun(inps)
    input_bounds = jax_verify.IntervalBound(inps - 1.0, inps + 1.0)

    boundprop_transform = jax_verify.ibp_transform
    relaxation_transform = relaxation.RelaxationTransform(boundprop_transform)
    var, env = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(relaxation_transform),
        logits_fun, input_bounds)

    objective_bias = 0.
    objective = jax.ops.index_update(jnp.zeros(output.shape[1:]), 0, 1)
    index = 0

    lower_bound, _, _ = relaxation.solve_relaxation(
        cvxpy_relaxation_solver.CvxpySolver, objective, objective_bias,
        var, env, index)

    self.assertLessEqual(lower_bound, output[index, 0])


def _predict_mlp(params, inputs):
  # pylint: disable=invalid-name
  inputs = np.reshape(inputs, (inputs.shape[0], -1))
  for W, b in params[:-1]:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.maximum(outputs, 0)
  W, b = params[-1]
  return jnp.dot(inputs, W) + b


class SavedModelTests(parameterized.TestCase):

  @parameterized.named_parameters(
      ('PGDNN', 'models/raghunathan18_pgdnn.pkl', 20, 19),
  )
  def test_mnist_mlp(self, model_name, num_examples, expected_correct):
    with jax_verify.open_file('mnist/x_test_first100.npy', 'rb') as f:
      mnist_x = np.load(f)
    with jax_verify.open_file('mnist/y_test.npy', 'rb') as f:
      mnist_y = np.load(f)
    with jax_verify.open_file(model_name, 'rb') as f:
      params = pickle.load(f)  # pytype: disable=wrong-arg-types  # due to GFile
    logits = np.array(_predict_mlp(params, mnist_x[:num_examples]))
    pred_labels = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(mnist_y[:num_examples], pred_labels))
    print(num_correct)
    assert num_correct == expected_correct, f'Number correct: {num_correct}'


if __name__ == '__main__':
  absltest.main()
