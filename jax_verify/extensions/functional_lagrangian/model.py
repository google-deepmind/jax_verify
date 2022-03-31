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

"""API to load model parameters."""

import dataclasses
import os
import pickle
from typing import Any, Optional
import urllib

import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.src import utils as jv_utils
import ml_collections
import numpy as np

ConfigDict = ml_collections.ConfigDict
ModelParams = verify_utils.ModelParams

INTERNAL_MODEL_PATHS = ml_collections.ConfigDict({
    'mnist_ceda': 'models/mnist_ceda.pkl',
    'mnist_cnn': 'models/mnist_lenet_dropout.pkl',
    'cifar_vgg_16': 'models/cifar_vgg_16_dropout.pkl',
    'cifar_vgg_32': 'models/cifar_vgg_32_dropout.pkl',
    'cifar_vgg_64': 'models/cifar_vgg_64_dropout.pkl',
})

PROBA_SAFETY_URL = (
    'https://github.com/matthewwicker/ProbabilisticSafetyforBNNs/raw/master'
    '/MNIST/concurMNIST2/MNIST_Networks')

PROBA_SAFETY_MODEL_PATHS = ml_collections.ConfigDict({
    'mnist_mlp_1_1024': 'VIMODEL_MNIST_1_1024_relu.net.npz',
    'mnist_mlp_1_128': 'VIMODEL_MNIST_1_128_relu.net.npz',
    'mnist_mlp_1_2048': 'VIMODEL_MNIST_1_2048_relu.net.npz',
    'mnist_mlp_1_256': 'VIMODEL_MNIST_1_256_relu.net.npz',
    'mnist_mlp_1_4096': 'VIMODEL_MNIST_1_4096_relu.net.npz',
    'mnist_mlp_1_512': 'VIMODEL_MNIST_1_512_relu.net.npz',
    'mnist_mlp_1_64': 'VIMODEL_MNIST_1_64_relu.net.npz',
    'mnist_mlp_2_1024': 'VIMODEL_MNIST_2_1024_relu.net.npz',
    'mnist_mlp_2_128': 'VIMODEL_MNIST_2_128_relu.net.npz',
    'mnist_mlp_2_256': 'VIMODEL_MNIST_2_256_relu.net.npz',
    'mnist_mlp_2_512': 'VIMODEL_MNIST_2_512_relu.net.npz',
    'mnist_mlp_2_64': 'VIMODEL_MNIST_2_64_relu.net.npz',
})


def _load_pickled_model(root_dir: str, model_name: str) -> ModelParams:
  model_path = getattr(INTERNAL_MODEL_PATHS, model_name.lower())
  if model_path.endswith('mnist_ceda.pkl'):
    with jv_utils.open_file(model_path, 'rb', root_dir=root_dir) as f:
      params_iterables = pickle.load(f, encoding='bytes')
  else:
    with jv_utils.open_file(model_path, 'rb', root_dir=root_dir) as f:
      params_iterables = list(np.load(f, allow_pickle=True).item().values())
  return make_model_params_from_iterables(params_iterables)


def make_model_params_from_iterables(raw_params: Any) -> ModelParams:
  """Make list of LayerParams from list of iterables."""
  conv_field_names = [
      f.name for f in dataclasses.fields(verify_utils.ConvParams)
  ]
  fc_field_names = [
      f.name for f in dataclasses.fields(verify_utils.FCParams)
  ]

  net = []
  for layer_params in raw_params:
    if isinstance(layer_params, tuple):
      w, b = layer_params
      layer = verify_utils.FCParams(w=w, b=b)
    elif (isinstance(layer_params, dict)
          and layer_params.get('type') == 'linear'):
      fc_params = dict(
          (k, v) for k, v in layer_params.items() if k in fc_field_names)
      if fc_params.get('dropout_rate', 0) > 0:
        w = fc_params['w']
        # adapt expected value of 'w'
        fc_params['w'] = w * (1.0 - fc_params['dropout_rate'])
        fc_params['w_bound'] = jax_verify.IntervalBound(
            lower_bound=jnp.minimum(w, 0.0), upper_bound=jnp.maximum(w, 0.0))
      layer = verify_utils.FCParams(**fc_params)
    elif isinstance(layer_params, dict):
      conv_params = dict(
          (k, v) for k, v in layer_params.items() if k in conv_field_names)
      # deal with 'W' vs 'w'
      if 'W' in layer_params:
        conv_params['w'] = layer_params['W']
      layer = verify_utils.ConvParams(**conv_params)
    else:
      raise TypeError(
          f'layer_params type not recognized: {type(layer_params)}.')
    net += [layer]
  return net


def _load_proba_safety_model(
    root_dir: str,
    model_name: str,
    num_std_for_bound: float,
) -> ModelParams:
  """Load model trained in Probabilistic Safety for BNNs paper."""
  model_path = getattr(PROBA_SAFETY_MODEL_PATHS, model_name.lower())
  local_path = os.path.join(root_dir, model_path)
  if not os.path.exists(local_path):
    download_url = os.path.join(PROBA_SAFETY_URL, model_path)
    urllib.request.urlretrieve(download_url, local_path)
  with open(local_path, 'rb') as f:
    data = np.load(f, allow_pickle=True, encoding='bytes')
    if not isinstance(data, np.ndarray):
      data = data['arr_0']

  assert len(data) % 4 == 0

  net = []
  for layer_idx in range(0, len(data) // 2, 2):
    # data: [w_0, b_0, w_1, b_1, ..., w_0_std, b_0_std, w_1_std, b_1_std, ...]
    w = jnp.array(data[layer_idx])
    b = jnp.array(data[layer_idx + 1])

    w_std = jnp.array(data[layer_idx + len(data) // 2])
    b_std = jnp.array(data[layer_idx + len(data) // 2 + 1])

    w_bound = jax_verify.IntervalBound(w - num_std_for_bound * w_std,
                                       w + num_std_for_bound * w_std)
    b_bound = jax_verify.IntervalBound(b - num_std_for_bound * b_std,
                                       b + num_std_for_bound * b_std)

    net += [
        verify_utils.FCParams(
            w=w,
            b=b,
            w_std=w_std,
            b_std=b_std,
            w_bound=w_bound,
            b_bound=b_bound)
    ]

  return net


def load_model(
    root_dir: str,
    model_name: str,
    num_std_for_bound: Optional[float],
) -> ModelParams:
  """Load and process model parameters."""
  if model_name.startswith('mnist_mlp'):
    return _load_proba_safety_model(root_dir, model_name, num_std_for_bound)
  else:
    return _load_pickled_model(root_dir, model_name)
