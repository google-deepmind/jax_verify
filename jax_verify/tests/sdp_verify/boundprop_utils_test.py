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
"""Tests for crown_boundprop.py."""

import functools
import pickle
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.sdp_verify import boundprop_utils
from jax_verify.extensions.sdp_verify import utils
import numpy as np


class BoundpropTest(parameterized.TestCase):

  def test_crown_boundprop(self):
    """Test CROWN bounds vs FGSM on Wong-Small MNIST CNN."""
    crown_boundprop = functools.partial(boundprop_utils.boundprop,
                                        boundprop_type='crown_ibp')
    self._test_boundprop(crown_boundprop)

  def test_nonconvex_boundprop(self):
    """Test Nonconvex bounds vs FGSM on Wong-Small MNIST CNN."""
    # Minimal test, since this already takes 70s.
    nonconvex_boundprop = functools.partial(
        boundprop_utils.boundprop, boundprop_type='nonconvex',
        nonconvex_boundprop_steps=2)
    self._test_boundprop(nonconvex_boundprop, num_idxs_to_test=1)

  def test_ibp_boundprop(self):
    def boundprop(params, x, epsilon, input_bounds):
      assert len(x.shape) == 4 and x.shape[0] == 1, f'shape check {x.shape}'
      init_bound = utils.init_bound(x[0], epsilon, input_bounds=input_bounds)
      return utils.boundprop(params, init_bound)
    self._test_boundprop(boundprop)

  def _test_boundprop(self, boundprop_method, num_idxs_to_test=10):
    """Test `boundprop_method` on Wong-Small MNIST CNN."""
    with jax_verify.open_file('mnist/x_test_first100.npy', 'rb') as f:
      xs = np.load(f)
    model_name = 'models/mnist_wongsmall_eps_10_adv.pkl'
    with jax_verify.open_file(model_name, 'rb') as f:
      params = pickle.load(f)
    x = xs[0]
    eps = 0.1

    bounds = boundprop_method(params, np.expand_dims(x, axis=0), eps,
                              input_bounds=(0., 1.))
    crown_lbs = utils.flatten([b.lb_pre for b in bounds[1:]])
    crown_ubs = utils.flatten([b.ub_pre for b in bounds[1:]])

    max_idx = crown_lbs.shape[0]
    np.random.seed(0)
    test_idxs = np.random.randint(max_idx, size=num_idxs_to_test)

    @jax.jit
    def fwd(x):
      _, acts = utils.predict_cnn(params, jnp.expand_dims(x, 0),
                                  include_preactivations=True)
      return acts

    get_act = lambda x, idx: utils.flatten(fwd(x), backend=jnp)[idx]

    print('Number of activations:', crown_lbs.shape[0])
    print('Bound shape', [b.lb.shape for b in bounds])
    print('Activation shape', [a.shape for a in fwd(x)])
    assert utils.flatten(fwd(x)).shape == crown_lbs.shape, (
        f'bad shape {crown_lbs.shape}, {utils.flatten(fwd(x)).shape}')

    for idx in test_idxs:
      nom = get_act(x, idx)
      crown_lb = crown_lbs[idx]
      crown_ub = crown_ubs[idx]

      adv_loss = lambda x: get_act(x, idx)  # pylint: disable=cell-var-from-loop
      x_lb = utils.pgd(adv_loss, x, eps, 50, 0.01)
      fgsm_lb = get_act(x_lb, idx)

      adv_loss = lambda x: -get_act(x, idx)  # pylint: disable=cell-var-from-loop
      x_ub = utils.pgd(adv_loss, x, eps, 50, 0.01)
      fgsm_ub = get_act(x_ub, idx)

      print(f'Idx {idx}: Boundprop LB {crown_lb}, FGSM LB {fgsm_lb}, '
            f'Nominal {nom}, FGSM UB {fgsm_ub}, Boundprop UB {crown_ub}')
      margin = 1e-5
      assert crown_lb <= fgsm_lb + margin, f'Bad lower bound. Idx {idx}.'
      assert crown_ub >= fgsm_ub - margin, f'Bad upper bound. Idx {idx}.'

      crown_lb_post, fgsm_lb_post = max(crown_lb, 0), max(fgsm_lb, 0)
      crown_ub_post, fgsm_ub_post = max(crown_ub, 0), max(fgsm_ub, 0)
      assert crown_lb_post <= fgsm_lb_post + margin, f'Idx {idx}.'
      assert crown_ub_post >= fgsm_ub_post - margin, f'Idx {idx}.'


if __name__ == '__main__':
  absltest.main()
