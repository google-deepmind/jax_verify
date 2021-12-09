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

"""Run verification with out-of-the-box LP solver.

This example uses jax_verify to generate Linear Program (LP) constraints
expressed in CVXPY, which is then solved with a generic LP solver.

Note that this CVXPY example is purely illustrative - it incurs a large overhead
for defining the problem, since CVXPY struggles with the large number of
constraints, particularly with convolutional layers. We will release more
performant implementations with other LP solvers in the future. We also welcome
contributions.
"""
import functools
import pickle
from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.sdp_verify import utils
from jax_verify.src.linear import forward_linear_bounds
import numpy as np

MLP_PATH = 'models/raghunathan18_pgdnn.pkl'
CNN_PATH = 'models/mnist_wongsmall_eps_10_adv.pkl'

flags.DEFINE_string('model', 'mlp', 'mlp or cnn or toy')
flags.DEFINE_string('boundprop_method', 'ibp', 'ibp or fastlin')
FLAGS = flags.FLAGS


def load_model(model_name):
  """Load model parameters and prediction function."""
  # Choose appropriate prediction function
  if model_name in ('mlp', 'toy'):
    model_path = MLP_PATH
    def model_fn(params, inputs):
      inputs = np.reshape(inputs, (inputs.shape[0], -1))
      return utils.predict_mlp(params, inputs)
  elif model_name == 'cnn':
    model_path = CNN_PATH
    model_fn = utils.predict_cnn
  else:
    raise ValueError('')

  # Get parameters
  if model_name == 'toy':
    params = [
        (np.random.normal(size=(784, 2)), np.random.normal(size=(2,))),
        (np.random.normal(size=(2, 10)), np.random.normal(size=(10,))),
    ]
  else:
    with jax_verify.open_file(model_path, 'rb') as f:
      params = pickle.load(f)
  return model_fn, params


def main(unused_args):

  # Load the parameters of an existing model.
  model_pred, params = load_model(FLAGS.model)
  logits_fn = functools.partial(model_pred, params)

  # Load some test samples
  with jax_verify.open_file('mnist/x_test_first100.npy', 'rb') as f:
    inputs = np.load(f)

  # Compute boundprop bounds
  eps = 0.1
  lower_bound = jnp.minimum(jnp.maximum(inputs[:2, ...] - eps, 0.0), 1.0)
  upper_bound = jnp.minimum(jnp.maximum(inputs[:2, ...] + eps, 0.0), 1.0)
  init_bound = jax_verify.IntervalBound(lower_bound, upper_bound)

  if FLAGS.boundprop_method == 'forwardfastlin':
    final_bound = jax_verify.forward_fastlin_bound_propagation(logits_fn,
                                                               init_bound)
    boundprop_transform = forward_linear_bounds.forward_fastlin_transform
  elif FLAGS.boundprop_method == 'ibp':
    final_bound = jax_verify.interval_bound_propagation(logits_fn, init_bound)
    boundprop_transform = jax_verify.ibp_transform
  else:
    raise NotImplementedError('Only ibp/fastlin boundprop are'
                              'currently supported')

  dummy_output = model_pred(params, inputs)

  # Run LP solver
  objective = jnp.where(jnp.arange(dummy_output[0, ...].size) == 0,
                        jnp.ones_like(dummy_output[0, ...]),
                        jnp.zeros_like(dummy_output[0, ...]))
  objective_bias = 0.
  value, _, status = jax_verify.solve_planet_relaxation(
      logits_fn, init_bound, boundprop_transform, objective,
      objective_bias, index=0)
  logging.info('Relaxation LB is : %f, Status is %s', value, status)
  value, _, status = jax_verify.solve_planet_relaxation(
      logits_fn, init_bound, boundprop_transform, -objective,
      objective_bias, index=0)
  logging.info('Relaxation UB is : %f, Status is %s', -value, status)

  logging.info('Boundprop LB is : %f', final_bound.lower[0, 0])
  logging.info('Boundprop UB is : %f', final_bound.upper[0, 0])


if __name__ == '__main__':
  app.run(main)
