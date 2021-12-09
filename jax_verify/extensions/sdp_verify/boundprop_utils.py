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
"""Crown bound propagation used in SDP verification."""

import functools
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.sdp_verify import utils
from jax_verify.src import bound_propagation
from jax_verify.src.nonconvex import duals
from jax_verify.src.nonconvex import nonconvex
from jax_verify.src.nonconvex import optimizers
from jax_verify.src.nonconvex.optimizers import LinesearchFistaOptimizer as FistaOptimizer
IntBound = utils.IntBound


def boundprop(params, x, epsilon, input_bounds, boundprop_type,
              **extra_boundprop_kwargs):
  """Computes interval bounds for NN intermediate activations.

  Args:
    params: Parameters for the NN.
    x: Batch of inputs to NN (dimension 2 for MLP or 4 for CNN)
    epsilon: l-inf perturbation to the input.
    input_bounds: Valid lower and upper for the NN as a tuple -- e.g. (0., 1.)
    boundprop_type: string, indicating method used for bound propagation, e.g.
      'crown_ibp' or 'nonconvex'
    **extra_boundprop_kwargs: any additional kwargs, passed directly to
      underlying boundprop method

  Returns:
    layer_bounds: upper and lower bounds across the layers of the NN as a list
    of IntBound-s.
  """
  boundprop_type_to_method = {
      'crown_ibp': _crown_ibp_boundprop,
      'nonconvex': _nonconvex_boundprop,
  }
  assert boundprop_type in boundprop_type_to_method, 'invalid boundprop_type'
  boundprop_method = boundprop_type_to_method[boundprop_type]
  return boundprop_method(params, x, epsilon, input_bounds,
                          **extra_boundprop_kwargs)


def _crown_ibp_boundprop(params, x, epsilon, input_bounds):
  """Runs CROWN-IBP for each layer separately."""
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


def _nonconvex_boundprop(params, x, epsilon, input_bounds,
                         nonconvex_boundprop_steps=100,
                         nonconvex_boundprop_nodes=128):
  """Wrapper for nonconvex bound propagation."""
  # Get initial bounds for boundprop
  init_bounds = utils.init_bound(x, epsilon, input_bounds=input_bounds,
                                 add_batch_dim=False)

  # Build fn to boundprop through
  all_act_fun = functools.partial(utils.predict_cnn, params,
                                  include_preactivations=True)

  # Collect the intermediate bounds.
  input_bound = jax_verify.IntervalBound(init_bounds.lb, init_bounds.ub)

  optimizer = optimizers.OptimizingConcretizer(
      FistaOptimizer(num_steps=nonconvex_boundprop_steps),
      max_parallel_nodes=nonconvex_boundprop_nodes)
  nonconvex_algorithm = nonconvex.nonconvex_algorithm(
      duals.WolfeNonConvexBound, optimizer)

  all_outputs, _ = bound_propagation.bound_propagation(
      nonconvex_algorithm, all_act_fun, input_bound)
  _, intermediate_nonconvex_bounds = all_outputs

  bounds = [init_bounds]
  for nncvx_bound in intermediate_nonconvex_bounds:
    bounds.append(utils.IntBound(lb_pre=nncvx_bound.lower,
                                 ub_pre=nncvx_bound.upper,
                                 lb=jnp.maximum(nncvx_bound.lower, 0),
                                 ub=jnp.maximum(nncvx_bound.upper, 0)))
  return bounds
