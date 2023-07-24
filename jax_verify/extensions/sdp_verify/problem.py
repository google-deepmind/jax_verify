# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

# pylint: disable=invalid-name
"""Facilities to construct SDP Verification problem instances."""

import collections

import jax.numpy as jnp
from jax_verify.extensions.sdp_verify import utils
import numpy as np


################## SDP Verification Instances ####################

# Dual variables correspond to:
#   lam: ReLU quadratic constraint: z^2 = z*(Wx)
#   nu: IBP quadratic constraint: x^2 <= (l+u)*x - l*u
#   nu_quad: IBP quadratic matrix constraint: (x_i - l_i)(x_j - u_j) <= 0
#   muminus: x'>=0
#   muminus2: Triangle linear Relu relaxation - u(Wx+b) - ul - (u-l)x' >= 0
#       where l = min(l, 0), u = max(u, 0)
#   muplus: x'>=Wx+b
DualVar = collections.namedtuple(
    'DualVar', ['lam', 'nu', 'nu_quad', 'muminus', 'muplus', 'muminus2'])
DualVarFin = collections.namedtuple('DualVarFin', ['nu', 'nu_quad'])
DEFAULT_DISABLED_DUAL_VARS = ('nu_quad', 'muminus2')
NECESSARY_DUAL_VARS = ('lam', 'muplus', 'muminus')


def make_relu_network_lagrangian(dual_vars, params, bounds, obj):
  """Returns a function that computes the Lagrangian for a ReLU network.

  This function assumes `params` represent a feedforward ReLU network i.e.
  x_{i+1} = relu(W_i x_i + b_i). It defines the Lagrangian by applying the
  objective `obj` to the final layer activations, and encoding the Lagrangian
  terms for each of the constraints defining the ReLU network. It then returns
  this function.

  Args:
    dual_vars: A length L+1 list of dual variables at each layer
    params: A length L list of (W, b) pairs, elided network weights
    bounds: A length L+1 list of `IntBound`s, elided bounds at each layer
    obj: function, taking final layer activations as input

  Returns:
    Function that computes Lagrangian L(x) with fixed `dual_vars`.
  """
  layer_sizes = utils.layer_sizes_from_bounds(bounds)

  def lagrangian(xs_list):
    """Computes Lagrangian L(x) with fixed `dual_vars`."""
    assert all([x.shape[0] == 1 for x in xs_list]), 'no batch mode support'

    lag = obj(xs_list[-1])
    for i in range(len(layer_sizes)):
      if i < len(params):
        y = utils.fwd(xs_list[i], params[i])
        # Lagrangian for constraint x' * x' = x' * (Wx+b) where x'=ReLU(Wx+b)
        lag += (jnp.sum(dual_vars[i].lam * xs_list[i + 1] *
                        (y - xs_list[i + 1])))
        # Lagrangian for the constraint x'>=Wx+b
        lag += jnp.sum(dual_vars[i].muplus * (xs_list[i + 1] - y))
        if dual_vars[i].muminus2.shape:
          # Lagrangian for u(Wx+b) - ul - (u-l)x' >= 0, where
          # l = min(l, 0) and u = max(u, 0)
          raise NotImplementedError('dropped support for muminus2')

        # Lagrangian for the constraint x'>=0
        lag += jnp.sum(dual_vars[i].muminus * xs_list[i + 1])

      # Lagrangian for IBP constraint (x-l)(x-u) <= 0
      if dual_vars[i].nu.shape:
        lag += -jnp.sum(dual_vars[i].nu *
                        (xs_list[i] - bounds[i].lb) *(xs_list[i] - bounds[i].ub)
                        )
      if dual_vars[i].nu_quad.shape:
        # IBP quadratic matrix constraint: (x_i - l_i)(x_j - u_j) <= 0
        lag += -jnp.sum(dual_vars[i].nu_quad *
                        jnp.matmul((xs_list[i]-bounds[i].lb).T,
                                   xs_list[i]-bounds[i].ub))
    return lag
  return lagrangian


def relu_robustness_verif_instance_to_sdp(verif_instance):
  """Convert solver-agnostic VerifInstance to SdpDualVerifInstance."""
  assert verif_instance.type in [
      utils.VerifInstanceTypes.MLP_ELIDED, utils.VerifInstanceTypes.CNN_ELIDED]
  elided_bounds = verif_instance.bounds[:-1]
  dual_shapes, dual_types = get_dual_shapes_and_types(elided_bounds)
  def obj(x_final):
    out = jnp.sum(x_final * jnp.reshape(verif_instance.obj, x_final.shape))
    return out + verif_instance.const
  def make_inner_lagrangian(dual_vars):
    return make_relu_network_lagrangian(
        dual_vars, verif_instance.params, elided_bounds, obj)
  return utils.SdpDualVerifInstance(
      make_inner_lagrangian=make_inner_lagrangian,
      bounds=elided_bounds,
      dual_shapes=dual_shapes,
      dual_types=dual_types)


def make_sdp_verif_instance(verif_instance):
  if isinstance(verif_instance, utils._AdvRobustnessVerifInstance):  # pylint: disable=protected-access
    return relu_robustness_verif_instance_to_sdp(verif_instance)
  else:
    raise NotImplementedError('unrecognized verif_instance type')


def make_vae_sdp_verif_instance(params, data_x, bounds):
  """Make SdpDualVerifInstance for VAE reconstruction error spec."""
  elided_params = params[:-1]
  elided_bounds = bounds[:-1]
  dual_shapes, dual_types = get_dual_shapes_and_types(elided_bounds)
  def recon_loss(x_final):
    x_hat = utils.predict_cnn(params[-1:], x_final).reshape(1, -1)
    return jnp.sum(jnp.square(data_x.reshape(x_hat.shape) - x_hat))
  def make_inner_lagrangian(dual_vars):
    return make_relu_network_lagrangian(
        dual_vars, elided_params, elided_bounds, recon_loss)
  return utils.SdpDualVerifInstance(
      make_inner_lagrangian=make_inner_lagrangian,
      bounds=elided_bounds,
      dual_shapes=dual_shapes,
      dual_types=dual_types)


def make_vae_semantic_spec_params(x, vae_params, classifier_params):
  """Defines network f(z_noise) = classifier(reconstruct(x, z_noise))."""
  # Setup - encoder fwd pass
  encoder_params, decoder_params = vae_params
  encoder_mu_params = encoder_params[:-1]
  encoder_sigmasq_params = encoder_params[:-2] + [encoder_params[-1]]
  mu_z = utils.predict_cnn(encoder_mu_params, x)
  log_sigmasq_z = utils.predict_cnn(encoder_sigmasq_params, x)
  sigmasq_z = jnp.exp(log_sigmasq_z)
  # Combine the reparameterization with the first decoder layer
  # z0 = mu + sigma * z
  # z1 = jnp.dot(z0, W) + b
  #    = jnp.dot(mu + sigma * z, W) + b
  #    = jnp.dot(z, sigma * W) + [b + jnp.dot(mu, W)]
  assert isinstance(decoder_params[0], tuple)
  W0_orig, b0_orig = decoder_params[0]
  W0 = W0_orig * jnp.reshape(jnp.sqrt(sigmasq_z), (-1, 1))
  b0 = b0_orig + jnp.dot(mu_z, W0_orig)

  # Now the network is just concatenation of modified decoder + classifier
  # This is also applying a Relu to decoder output, but that's fine
  combined_params = [(W0, b0)] + decoder_params[1:] + classifier_params
  return combined_params


def get_dual_shapes_and_types(bounds_elided):
  """Get shapes and types of dual vars."""
  dual_shapes = []
  dual_types = []
  layer_sizes = utils.layer_sizes_from_bounds(bounds_elided)
  for it in range(len(layer_sizes)):
    m = layer_sizes[it]
    m = [m] if isinstance(m, int) else list(m)
    if it < len(layer_sizes)-1:
      n = layer_sizes[it + 1]
      n = [n] if isinstance(n, int) else list(n)
      shapes = {
          'lam': [1] + n,
          'nu': [1] + m,
          'muminus': [1] + n,
          'muplus': [1] + n,
          'nu_quad': [], 'muminus2': [],
      }
      types = {
          'lam': utils.DualVarTypes.EQUALITY,
          'nu': utils.DualVarTypes.INEQUALITY,
          'muminus': utils.DualVarTypes.INEQUALITY,
          'muplus': utils.DualVarTypes.INEQUALITY,
          'nu_quad': utils.DualVarTypes.INEQUALITY,
          'muminus2': utils.DualVarTypes.INEQUALITY,
      }
      dual_shapes.append(DualVar(**{
          k: np.array(s) for k, s in shapes.items()}))
      dual_types.append(DualVar(**types))
    else:
      shapes = {'nu': [1] + m, 'nu_quad': []}
      types = {'nu': utils.DualVarTypes.INEQUALITY,
               'nu_quad': utils.DualVarTypes.INEQUALITY}
      dual_shapes.append(DualVarFin(**{
          k: np.array(s) for k, s in shapes.items()}))
      dual_types.append(DualVarFin(**types))

  # Add kappa
  N = sum([np.prod(np.array(i)) for i in layer_sizes])
  dual_shapes.append(np.array([1, N+1]))
  dual_types.append(utils.DualVarTypes.INEQUALITY)
  return dual_shapes, dual_types
