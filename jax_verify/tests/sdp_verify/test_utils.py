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

# Lint as: python3
# pylint: disable=invalid-name
"""Small helper functions for testing."""

import jax
import jax.numpy as jnp
import jax.random as random
from jax_verify.extensions.sdp_verify import utils
import numpy as np

################## Toy Networks ######################


def _init_fc_layer(key, n_in, n_out):
  k1, k2 = random.split(key)
  W = random.normal(k1, (n_in, n_out))/ n_in
  b = random.normal(k2, (n_out,))/ n_in
  return W, b


def _init_conv_layer(key, n_h, n_w, n_cout, n_cin):
  """Random weights for a conv layer.

  Args:
    key: PRNG key
    n_h: Kernel Height
    n_w: Kernel Width
    n_cout: Out Channels
    n_cin: In Channels
  Returns:
    W: weights of the conv filters
    b: biases for the conv layer
  """
  k1, k2 = random.split(key)
  W = random.normal(k1, (n_h, n_w, n_cin, n_cout))
  b = random.normal(k2, (n_cout,))
  return W, b


def make_mlp_params(layer_sizes, key):
  sizes = layer_sizes
  keys = random.split(key, len(sizes))
  params = list(map(_init_fc_layer, keys[1:], sizes[:-1], sizes[1:]))
  return params


def make_cnn_params(layer_sizes, key):
  """Initialize a random network with conv layers, followed by linear layers.

  Args:
    layer_sizes: Layer-0 is the shape of the input in the NHWC format
    e.g. (1, 32, 32, 3). List of layer-wise hyperpameters. Dict of the following
    form for a conv-layer: {'n_h': 3, 'n_w': 3, 'n_cout': 32,
    'padding': 'VALID', 'stride': 1, 'n_cin': 3, input_shape:}. Input shape only
    if it's the first layer. For an FC layer, a single int corresponding to the
    number of op neurons.
    key: PRNG key

  Returns:
    W: weights of the conv filters
    b: biases for the conv layer
  """
  sizes = layer_sizes
  keys = random.split(key, len(sizes))
  params = []

  # NHWC, assert square
  assert len(layer_sizes[0]) == 4
  assert layer_sizes[0][1] == layer_sizes[0][2]
  conv_check = 0
  input_shape = layer_sizes[0][1]

  for counter, size in enumerate(layer_sizes[1:]):
    if isinstance(size, dict):
      size['input_shape'] = input_shape
      size['W'], size['b'] = _init_conv_layer(keys[counter], size['n_h'],
                                              size['n_w'], size['n_cout'],
                                              size['n_cin'])
      if size['padding'] == 'VALID':
        input_shape = int(np.ceil(input_shape - size['n_h'] +1)/size['stride'])
      else:
        input_shape = int(np.ceil(input_shape/size['stride']))
      size['output_shape'] = input_shape
      params.append(size)

    elif isinstance(size, int):
      # Check layer is FC
      if conv_check == 0:
        input_shape = input_shape * input_shape * layer_sizes[counter]['n_cout']
        conv_check = 1
      params.append(_init_fc_layer(keys[counter], input_shape, size))
      input_shape = size
    else:
      raise NotImplementedError('Unknown layer')
  return params


################## Toy Verification Instances ####################


def make_toy_verif_instance(seed=None, label=None, target_label=None, nn='mlp'):
  """Mainly used for unit testing."""
  key = jax.random.PRNGKey(0) if seed is None else jax.random.PRNGKey(seed)
  if nn == 'mlp':
    layer_sizes = '5, 5, 5'
    layer_sizes = np.fromstring(layer_sizes, dtype=int, sep=',')
    params = make_mlp_params(layer_sizes, key)
    inp_shape = (1, layer_sizes[0])
  else:
    if nn == 'cnn_simple':
      pad = 'VALID'
      # Input and filter size match -> filter is applied at just one location.

    else:
      pad = 'SAME'
      # Input is padded on right/bottom to form 3x3 input

    layer_sizes = [(1, 2, 2, 1), {
        'n_h': 2,
        'n_w': 2,
        'n_cout': 2,
        'padding': pad,
        'stride': 1,
        'n_cin': 1
    }, 3]
    inp_shape = layer_sizes[0]
    params = make_cnn_params(layer_sizes, key)

  bounds = utils.boundprop(
      params,
      utils.IntBound(lb=np.zeros(inp_shape),
                     ub=1*np.ones(inp_shape),
                     lb_pre=None,
                     ub_pre=None)
  )
  target_label = 1 if target_label is None else target_label
  label = 2 if label is None else label
  verif_instance = utils.make_nn_verif_instance(
      params,
      bounds,
      target_label=target_label,
      label=label,
      input_bounds=(0., 1.))
  return verif_instance


def make_mlp_layer_from_conv_layer(layer_params, input_bounds):
  """Convert Conv Layer into equivalent MLP layer."""
  assert isinstance(layer_params, dict)
  assert layer_params['padding'] == 'SAME'
  assert layer_params['stride'] == 1
  assert layer_params['n_cin'] == 1
  # only 'SAME' padding supported for now with stride (1,1)
  # to be used for unit-test support only
  # TODO: Add support for 'VALID'

  inp_shape = (layer_params['input_shape'], layer_params['input_shape'])
  w, b = layer_params['W'], layer_params['b']
  op_shape = int(np.ceil(inp_shape[0] / layer_params['stride']))
  pad_h = max((op_shape - 1) * layer_params['stride'] + layer_params['n_h'] -
              inp_shape[0], 0)
  pad_t = pad_h // 2
  pad_b = pad_h - pad_t
  pad_inp_shape = [inp_shape[0] + pad_h, inp_shape[1] + pad_h]
  padded_bounds = jnp.zeros(pad_inp_shape)
  lb = padded_bounds.at[pad_t:-pad_b, pad_t:-pad_b].add(input_bounds.lb[0, :, :,
                                                                        0])
  ub = padded_bounds.at[pad_t:-pad_b, pad_t:-pad_b].add(input_bounds.ub[0, :, :,
                                                                        0])
  pad_filter_shape = pad_inp_shape + [inp_shape[0], inp_shape[1], w.shape[-1]]
  pad_filter = jnp.zeros(pad_filter_shape)
  pad_bias = jnp.zeros(inp_shape + (w.shape[-1],))
  n_h, n_w = w.shape[0], w.shape[1]

  # unrolling the conv into an FC layer, stride=(1,1)
  for i in range(inp_shape[0]):
    for j in range(inp_shape[1]):
      pad_filter = pad_filter.at[i:i + n_h, j:j + n_w, i, j, 0].add(w[:, :, 0,
                                                                      0])
      pad_bias = pad_bias.at[i, j, 0].add(b[0])
      pad_filter = pad_filter.at[i:i + n_h, j:j + n_w, i, j, 1].add(w[:, :, 0,
                                                                      1])
      pad_bias = pad_bias.at[i, j, 1].add(b[1])
  pad_filter_lin = jnp.reshape(
      pad_filter,
      (pad_inp_shape[0] * pad_inp_shape[1], inp_shape[0] * inp_shape[1] * 2))
  pad_bias_lin = jnp.reshape(pad_bias, inp_shape[0] * inp_shape[1] * 2)

  return lb, ub, pad_filter_lin, pad_bias_lin


def make_mlp_verif_instance_from_cnn(verif_instance):
  """Convert CNN verif-instance into equivalent MLP verif-instance."""
  params_cnn = verif_instance.params_full
  assert not any([isinstance(x, dict) for x in params_cnn[1:]])
  # Only supports networks with structure conv-{fc}*
  weights = []
  for layer_params in params_cnn:
    if isinstance(layer_params, dict):
      lb, ub, w_lin, b_lin = make_mlp_layer_from_conv_layer(
          layer_params, verif_instance.bounds[0])
      lb = jnp.reshape(lb, (1, -1)) * 0.
      ub = jnp.reshape(ub, (1, -1)) * 1.
      weights.append((w_lin, b_lin))
    else:
      weights.append(layer_params)
  return lb, ub, weights
