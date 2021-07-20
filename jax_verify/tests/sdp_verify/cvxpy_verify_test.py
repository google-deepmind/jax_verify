# coding=utf-8
# Copyright 2021 The jax_verify Authors.
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
"""Tests for cvxpy_verify.py."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
import jax.numpy as jnp
from jax_verify.src.sdp_verify import cvxpy_verify
from jax_verify.src.sdp_verify import utils
from jax_verify.tests.sdp_verify import test_utils

NO_MIP_SOLVERS_MESSAGE = 'No mixed-integer solver is installed.'


class CvxpyTest(parameterized.TestCase):

  @unittest.skipUnless(MIP_SOLVERS, NO_MIP_SOLVERS_MESSAGE)
  def test_mip_status(self):
    """Test toy MIP is solved optimally by cvxpy."""
    for seed in range(10):
      verif_instance = test_utils.make_toy_verif_instance(seed)
      val, info = cvxpy_verify.solve_mip_mlp_elided(verif_instance)
      status = info['problem'].status
      assert val is not None
      assert status in ('optimal', 'optimal_inaccurate'), f'Status is {status}.'

  def test_sdp_status(self):
    """Test toy SDP is solved optimally by cvxpy."""
    for seed in range(10):
      verif_instance = test_utils.make_toy_verif_instance(seed)
      val, info = cvxpy_verify.solve_sdp_mlp_elided(verif_instance)
      status = info['problem'].status
      assert val is not None
      assert status in ('optimal', 'optimal_inaccurate'), f'Status is {status}.'


def _fgsm_example_and_bound(params, target_label, label):
  model_fn = lambda x: utils.predict_mlp(params, x)
  x = 0.5 * jnp.ones(utils.nn_layer_sizes(params)[0])
  epsilon = 0.5
  x_adv = utils.fgsm_single(model_fn, x, label, target_label, epsilon,
                            num_steps=30, step_size=0.03)
  return x_adv, utils.adv_objective(model_fn, x_adv, label, target_label)

MARGIN = 1e-6


class CrossingBoundsTest(parameterized.TestCase):
  """Check IBP,SDP relaxations <= MIP <= FGSM upper bound."""

  @unittest.skipUnless(MIP_SOLVERS, NO_MIP_SOLVERS_MESSAGE)
  def test_fgsm_vs_mip(self):
    num_repeats = 5
    target_label, label = 1, 2
    for seed in range(num_repeats):
      verif_instance = test_utils.make_toy_verif_instance(
          seed, target_label=target_label, label=label)
      mip_val, _ = cvxpy_verify.solve_mip_mlp_elided(verif_instance)
      _, fgsm_val = _fgsm_example_and_bound(
          verif_instance.params_full, target_label=target_label, label=label)
      assert mip_val > fgsm_val - MARGIN, (
          'MIP exact solution should be greater than FGSM lower bound.')

  @unittest.skipUnless(MIP_SOLVERS, NO_MIP_SOLVERS_MESSAGE)
  def test_sdp_vs_mip(self):
    num_repeats = 5
    loss_margin = 1e-3  # fixed via runs_per_test=300 with random seeds
    for seed in range(num_repeats):
      verif_instance = test_utils.make_toy_verif_instance(seed)
      mip_val, _ = cvxpy_verify.solve_mip_mlp_elided(verif_instance)
      sdp_val, _ = cvxpy_verify.solve_sdp_mlp_elided(verif_instance)
      assert sdp_val > mip_val - loss_margin, (
          'SDP relaxation should be greater than MIP exact solution. '
          f'Vals are MIP: {mip_val} SDP: {sdp_val}')


class MatchingBoundsTest(parameterized.TestCase):

  @unittest.skipUnless(MIP_SOLVERS, NO_MIP_SOLVERS_MESSAGE)
  def test_fgsm_vs_mip(self):
    """Check FGSM and MIP reach same solution/value most of the time."""
    # Note this test only works with fixed seeds
    num_repeats = 5
    expected_successes = 4
    num_successes = 0
    loss_margin = 0.01
    target_label, label = 1, 2
    for seed in range(num_repeats):
      verif_instance = test_utils.make_toy_verif_instance(
          seed, target_label=target_label, label=label)
      mip_val, _ = cvxpy_verify.solve_mip_mlp_elided(verif_instance)
      _, fgsm_val = _fgsm_example_and_bound(
          verif_instance.params_full, target_label=target_label, label=label)
      if abs(mip_val - fgsm_val) < loss_margin:
        num_successes += 1
    assert num_successes >= expected_successes, f'Successes: {num_successes}'


class SdpTest(parameterized.TestCase):

  def test_constraints_numpy(self):
    num_repeats = 5
    margin = 3e-4
    for seed in range(num_repeats):
      verif_instance = test_utils.make_toy_verif_instance(
          seed=seed, label=1, target_label=2)
      obj_value, info = cvxpy_verify.solve_sdp_mlp_elided(verif_instance)
      obj_np, violations = cvxpy_verify.check_sdp_bounds_numpy(
          info['P'].value, verif_instance)
      assert abs(obj_np - obj_value) < margin, 'objective does not match'
      for k, v in violations.items():
        assert v < margin, f'violation of {k} by {v}'


if __name__ == '__main__':
  absltest.main()
