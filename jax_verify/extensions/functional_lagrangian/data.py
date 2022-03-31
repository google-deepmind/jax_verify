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

"""Data util functions."""

import os
import pickle
from typing import Sequence, Tuple

import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils
from jax_verify.src import utils as jv_utils
import ml_collections
import numpy as np


ConfigDict = ml_collections.ConfigDict
DataSpec = verify_utils.DataSpec
IntervalBound = jax_verify.IntervalBound
SpecType = verify_utils.SpecType
Tensor = jnp.array
LayerParams = verify_utils.LayerParams
ModelParams = verify_utils.ModelParams
ModelParamsElided = verify_utils.ModelParamsElided


DATA_PATH = ml_collections.ConfigDict({
    'emnist_CEDA': 'emnist_CEDA.pkl',
    'mnist': 'mnist',
    'cifar10': 'cifar10',
    'emnist': 'emnist',
    'cifar100': 'cifar100',
})


def load_dataset(
    root_dir: str,
    dataset: str,
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
  """Loads the MNIST/CIFAR/EMNIST test set examples, saved as numpy arrays."""

  data_path = DATA_PATH.get(dataset)

  if dataset == 'emnist_CEDA':
    with jv_utils.open_file(data_path, 'rb', root_dir=root_dir) as f:
      ds = pickle.load(f)
    xs, ys = ds[0], ds[1]
    xs = np.reshape(xs, [-1, 28, 28, 1])
    ys = np.reshape(ys, [-1])
    return xs, ys
  else:
    x_filename = os.path.join(data_path, 'x_test.npy')
    y_filename = os.path.join(data_path, 'y_test.npy')
    with jv_utils.open_file(x_filename, 'rb', root_dir=root_dir) as f:
      xs = np.load(f)
    with jv_utils.open_file(y_filename, 'rb', root_dir=root_dir) as f:
      ys = np.load(f)
  return xs, ys


def make_data_spec(config_problem: ConfigDict, root_dir: str) -> DataSpec:
  """Create data specification from config_problem."""
  xs, ys = load_dataset(root_dir, config_problem.dataset)
  if config_problem.dataset in ('cifar10', 'cifar100'):
    x = sdp_utils.preprocess_cifar(xs[config_problem.dataset_idx])
    epsilon, input_bounds = sdp_utils.preprocessed_cifar_eps_and_input_bounds(
        shape=x.shape,
        epsilon=config_problem.epsilon_unprocessed,
        inception_preprocess=config_problem.scale_center)
  else:
    x = xs[config_problem.dataset_idx]
    epsilon = config_problem.epsilon_unprocessed
    input_bounds = (jnp.zeros_like(x), jnp.ones_like(x))
  true_label = ys[config_problem.dataset_idx]
  target_label = config_problem.target_label_idx
  return DataSpec(
      input=x,
      true_label=true_label,
      target_label=target_label,
      epsilon=epsilon,
      input_bounds=input_bounds)
