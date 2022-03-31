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

"""Bound propagation example usage: IBP, Fastlin, CROWN, CROWN-IBP.

Examples:
  python3 run_boundprop.py
  python3 run_boundprop.py --model=cnn
  python3 run_boundprop.py --boundprop_method=fastlin_bound_propagation
"""
import functools
import pickle
from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.sdp_verify import utils
import numpy as np

MLP_PATH = 'models/raghunathan18_pgdnn.pkl'
CNN_PATH = 'models/mnist_wongsmall_eps_10_adv.pkl'
ALL_BOUNDPROP_METHODS = (
    jax_verify.interval_bound_propagation,
    jax_verify.forward_fastlin_bound_propagation,
    jax_verify.backward_fastlin_bound_propagation,
    jax_verify.ibpforwardfastlin_bound_propagation,
    jax_verify.forward_crown_bound_propagation,
    jax_verify.backward_crown_bound_propagation,
    jax_verify.crownibp_bound_propagation,
)

flags.DEFINE_string('model', 'mlp', 'mlp or cnn')
flags.DEFINE_string('boundprop_method', '',
                    'Any boundprop method, such as `interval_bound_propagation`'
                    ' `forward_fastlin_bound_propagation` or '
                    ' `crown_bound_propagation`.'
                    'Empty string defaults to IBP.')
FLAGS = flags.FLAGS


def load_model(model_name):
  """Load model parameters and prediction function."""
  # Choose appropriate prediction function
  if model_name == 'mlp':
    model_path = MLP_PATH
    def model_fn(params, inputs):
      inputs = np.reshape(inputs, (inputs.shape[0], -1))
      return utils.predict_mlp(params, inputs)
  elif model_name == 'cnn':
    model_path = CNN_PATH
    model_fn = utils.predict_cnn
  else:
    raise ValueError('')

  # Load parameters from file
  with jax_verify.open_file(model_path, 'rb') as f:
    params = pickle.load(f)
  return model_fn, params


def main(unused_args):
  # Load some test samples
  with jax_verify.open_file('mnist/x_test_first100.npy', 'rb') as f:
    inputs = np.load(f)

  # Load the parameters of an existing model.
  model_pred, params = load_model(FLAGS.model)

  # Evaluation of the model on unperturbed images.
  clean_preds = model_pred(params, inputs)

  # Define initial bound
  eps = 0.1
  initial_bound = jax_verify.IntervalBound(
      jnp.minimum(jnp.maximum(inputs - eps, 0.0), 1.0),
      jnp.minimum(jnp.maximum(inputs + eps, 0.0), 1.0))

  # Because our function `model_pred` takes as inputs both the parameters
  # `params` and the `inputs`, we need to wrap it such that it only takes
  # `inputs` as parameters.
  logits_fn = functools.partial(model_pred, params)

  # Apply bound propagation. All boundprop methods take as an input the model
  # `function`, and the inital bounds, and return final bounds with the same
  # structure as the output of `function`. Internally, these methods work by
  # replacing each operation with its boundprop equivalent - see
  # bound_propagation.py for details.
  boundprop_method = (
      jax_verify.interval_bound_propagation if not FLAGS.boundprop_method else
      getattr(jax_verify, FLAGS.boundprop_method))
  assert boundprop_method in ALL_BOUNDPROP_METHODS, 'unsupported method'
  final_bound = boundprop_method(logits_fn, initial_bound)

  logging.info('Lower bound: %s', final_bound.lower)
  logging.info('Upper bound: %s', final_bound.upper)
  logging.info('Clean predictions: %s', clean_preds)

  assert jnp.all(final_bound.lower <= clean_preds), 'Invalid lower bounds'
  assert jnp.all(final_bound.upper >= clean_preds), 'Invalid upper bounds'


if __name__ == '__main__':
  app.run(main)
