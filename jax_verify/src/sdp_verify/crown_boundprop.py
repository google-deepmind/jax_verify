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
# pylint: disable=invalid-name
"""Crown bound propagation used in SDP verification."""

import functools
import jax.numpy as jnp
import jax_verify
from jax_verify.src.sdp_verify import utils
IntBound = utils.IntBound


def boundprop(params, x, epsilon, input_bounds):
  """Runs CROWN-IBP for each layer separately.

  Args:
    params: Parameters for the NN.
    x: Batch of inputs to NN (dimension 2 for MLP or 4 for CNN)
    epsilon: l-inf perturbation to the input.
    input_bounds: Valid lower and upper for the NN as a tuple -- e.g. (0., 1.)

  Returns:
    layer_bounds: upper and lower bounds across the layers of the NN as a list
    of IntBound-s.
  """

  def get_layer_act(layer_idx, inputs):
    act = utils.predict_cnn(params[:layer_idx], inputs)
    return act

  initial_bound = jax_verify.IntervalBound(
      jnp.maximum(x - epsilon, input_bounds[0]),
      jnp.minimum(x + epsilon, input_bounds[1]))

  out_bounds = [IntBound(
      lb_pre=None, ub_pre=None, lb=initial_bound.lower, ub=initial_bound.upper)]
  for i in range(1, len(params) + 1):
    fwd = functools.partial(get_layer_act, i)
    bound = jax_verify.crownibp_bound_propagation(fwd, initial_bound)
    out_bounds.append(
        IntBound(lb_pre=bound.lower, ub_pre=bound.upper,
                 lb=jnp.maximum(0, bound.lower),
                 ub=jnp.maximum(0, bound.upper)))
  return out_bounds
