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

"""Example architecture to test functions.
"""

import haiku as hk
import jax


class ResidualModel(hk.Module):
  """Model with several complexities.

  Contains Residual connections, stateful model (batch norms), nested modules.
  """

  def __init__(self):
    super().__init__()

    bn_config = {'create_scale': True,
                 'create_offset': True,
                 'decay_rate': 0.999}
   ## Definition of the modules.
    self.conv_block = hk.Sequential([
        hk.Conv2D(32, (3, 3), stride=3, rate=1), jax.nn.relu,
        hk.Conv2D(32, (3, 3), stride=3, rate=1), jax.nn.relu,
        hk.Conv2D(64, (3, 3), stride=3, rate=1), jax.nn.relu,
    ])

    self.conv_res_block = hk.Sequential([
        hk.Conv2D(32, (1, 1), stride=1, rate=1), jax.nn.relu,
        hk.Conv2D(32, (1, 1), stride=1, rate=1), jax.nn.relu,
        hk.Conv2D(64, (1, 1), stride=1, rate=1), jax.nn.relu,
    ])

    self.reshape_mod = hk.Flatten()

    self.lin_res_block = [
        (hk.Linear(128), hk.BatchNorm(name='lin_batchnorm_0', **bn_config)),
        (hk.Linear(256), hk.BatchNorm(name='lin_batchnorm_1', **bn_config))
    ]

    self.final_linear = hk.Linear(10)

  def call_all_act(self, inputs, is_training, test_local_stats=False):
    """Evaluate the model, returning its intermediate activations.

    Args:
      inputs: BHWC array of images.
      is_training: Boolean flag, whether this is during training.
      test_local_stats: Boolean flag, Whether local stats are used
        when is_training=False (for batchnorm).
    Returns:
      all_acts: List with the intermediate activations of interest.
    """
    all_acts = []
    all_acts.append(inputs)

    ##  Forward propagation.
    # First conv layer.
    act = self.conv_block(inputs)
    all_acts.append(act)
    # Convolutional residual block.
    act = act + self.conv_res_block(act)
    all_acts.append(act)
    # Reshape before fully connected part.
    act = self.reshape_mod(act)
    all_acts.append(act)
    # Fully connected residual block.
    lin_block_act = act
    for lin_i, bn_i in self.lin_res_block:
      lin_block_act = lin_i(lin_block_act)
      lin_block_act = bn_i(lin_block_act, is_training, test_local_stats)
      lin_block_act = jax.nn.relu(lin_block_act)
    act = act + lin_block_act
    all_acts.append(act)
    # Final layer.
    act = self.final_linear(act)
    all_acts.append(act)
    return all_acts

  def __call__(self, inputs, is_training, test_local_stats=False):
    """Return only the final prediction of the model.

    Args:
      inputs: BHWC array of images.
      is_training: Boolean flag, whether this is during training.
      test_local_stats: Boolean flag, Whether local stats are used
        when is_training=False (for batchnorm).
    Returns:
      pred: Array with the predictions, corresponding to the last activations.
    """
    all_acts = self.call_all_act(inputs, is_training, test_local_stats)
    return all_acts[-1]


class SmallResidualModel(hk.Module):
  """Small network with residual connections.

  Smaller version of ResidualModel.
  """

  def __init__(self):
    super().__init__()
    bn_config = {'create_scale': True,
                 'create_offset': True,
                 'decay_rate': 0.999}

    # Definition of the modules.
    self.conv_block = hk.Sequential([
        hk.Conv2D(1, (3, 3), stride=3, rate=1), jax.nn.relu,
        hk.Conv2D(1, (3, 3), stride=3, rate=1), jax.nn.relu,
    ])

    self.conv_res_block = hk.Sequential([
        hk.Conv2D(1, (1, 1), stride=1, rate=1), jax.nn.relu,
        hk.Conv2D(1, (1, 1), stride=1, rate=1), jax.nn.relu,
    ])

    self.reshape_mod = hk.Flatten()

    self.lin_res_block = [
        (hk.Linear(16), hk.BatchNorm(name='lin_batchnorm_0', **bn_config))
    ]

    self.final_linear = hk.Linear(10)

  def call_all_act(self, inputs, is_training, test_local_stats=False):
    """Evaluate the model, returning its intermediate activations.

    Args:
      inputs: BHWC array of images.
      is_training: Boolean flag, whether this is during training.
      test_local_stats: Boolean flag, Whether local stats are used
        when is_training=False (for batchnorm).
    Returns:
      all_acts: List with the intermediate activations of interest.
    """
    all_acts = []
    all_acts.append(inputs)

    ##  Forward propagation.
    # First conv layer.
    act = self.conv_block(inputs)
    all_acts.append(act)
    # Convolutional residual block.
    act = act + self.conv_res_block(act)
    all_acts.append(act)
    # Reshape before fully connected part.
    act = self.reshape_mod(act)
    all_acts.append(act)
    # Fully connected residual block.
    lin_block_act = act
    for lin_i, bn_i in self.lin_res_block:
      lin_block_act = lin_i(lin_block_act)
      lin_block_act = bn_i(lin_block_act, is_training, test_local_stats)
      lin_block_act = jax.nn.relu(lin_block_act)
    act = act + lin_block_act
    all_acts.append(act)
    # Final layer.
    act = self.final_linear(act)
    all_acts.append(act)
    return all_acts

  def __call__(self, inputs, is_training, test_local_stats=False):
    """Return only the final prediction of the model.

    Args:
      inputs: BHWC array of images.
      is_training: Boolean flag, whether this is during training.
      test_local_stats: Boolean flag, Whether local stats are used
        when is_training=False (for batchnorm).
    Returns:
      pred: Array with the predictions, corresponding to the last activations.
    """
    all_acts = self.call_all_act(inputs, is_training, test_local_stats)
    return all_acts[-1]


class TinyModel(hk.Module):
  """Tiny network.

  Single conv layer.
  """

  def __init__(self):
    super().__init__()
    # Definition of the modules.
    self.reshape_mod = hk.Flatten()

    self.lin_block = hk.Sequential([
        hk.Linear(20), jax.nn.relu,
    ])

    self.final_linear = hk.Linear(10)

  def call_all_act(self, inputs, is_training, test_local_stats=False):
    """Evaluate the model, returning its intermediate activations.

    Args:
      inputs: BHWC array of images.
      is_training: Boolean flag, whether this is during training.
      test_local_stats: Boolean flag, Whether local stats are used
        when is_training=False (for batchnorm).
    Returns:
      all_acts: List with the intermediate activations of interest.
    """
    all_acts = []
    all_acts.append(inputs)
    act = inputs
    ##  Forward propagation.
    act = self.reshape_mod(act)
    all_acts.append(act)

    # First linear layer.
    act = self.lin_block(act)
    all_acts.append(act)
    # Final layer.
    act = self.final_linear(act)
    all_acts.append(act)
    return all_acts

  def __call__(self, inputs, is_training, test_local_stats=False):
    """Return only the final prediction of the model.

    Args:
      inputs: BHWC array of images.
      is_training: Boolean flag, whether this is during training.
      test_local_stats: Boolean flag, Whether local stats are used
        when is_training=False (for batchnorm).
    Returns:
      pred: Array with the predictions, corresponding to the last activations.
    """
    all_acts = self.call_all_act(inputs, is_training, test_local_stats)
    return all_acts[-1]

