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

"""Tests for sdp_verify.py."""

import os
import random
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
import jax
import jax.numpy as jnp
import jax.scipy
from jax_verify.extensions.sdp_verify import cvxpy_verify
from jax_verify.extensions.sdp_verify import eigenvector_utils
from jax_verify.extensions.sdp_verify import problem
from jax_verify.extensions.sdp_verify import sdp_verify
from jax_verify.extensions.sdp_verify import utils
from jax_verify.tests.sdp_verify import test_utils
import numpy as np
import optax

NO_MIP_SOLVERS_MESSAGE = 'No mixed-integer solver is installed.'


class LanczosTest(parameterized.TestCase):

  def _test_max_eigenvector_lanczos_once(self, seed=0, dynamic_unroll=False):
    """Test max_eigenvector_lanczos against Scipy eigenvector method."""
    dim = 5
    key = jax.random.PRNGKey(seed)
    h_tmp = jax.random.normal(key, shape=(dim, dim))
    h = h_tmp + jnp.transpose(h_tmp)
    hv = lambda v: jnp.matmul(h, v)
    # Do `dim` iterations of Lanczos
    max_eigenvec_lanczos = eigenvector_utils.max_eigenvector_lanczos(
        hv, dim, dim, key, dynamic_unroll=dynamic_unroll)
    _, eigen_vecs_scipy = jax.scipy.linalg.eigh(h)
    max_eigenvec_scipy = eigen_vecs_scipy[:, -1]
    # Eigenvector can be v or -v
    err = min(jnp.linalg.norm(max_eigenvec_lanczos - max_eigenvec_scipy),
              jnp.linalg.norm(max_eigenvec_lanczos + max_eigenvec_scipy))
    # Eigenvectors have unit norm - check for relative error below 1e-5
    assert err < 1e-5, (f'err: {err}, lanczos: {max_eigenvec_lanczos} '
                        f'scipy: {max_eigenvec_scipy}')

  def test_max_eigenvector_lanczos(self):
    for i in range(10):
      self._test_max_eigenvector_lanczos_once(seed=i)
    for i in range(10):
      self._test_max_eigenvector_lanczos_once(seed=i, dynamic_unroll=True)

  def _test_lanczos_dynamic_vs_static_once(self, seed=0):
    def _safe_div(x1, x2):
      return jnp.where(jnp.logical_and(x1 == 0, x2 == 0), x1, x1 / x2)
    dim = 5
    key = jax.random.PRNGKey(seed)
    h_tmp = jax.random.normal(key, shape=(dim, dim))
    h = h_tmp + jnp.transpose(h_tmp)
    hv = lambda v: jnp.matmul(h, v)
    tr1, vecs1 = eigenvector_utils.lanczos_alg(
        hv, dim, dim, key, dynamic_unroll=True)
    tr2, vecs2 = eigenvector_utils.lanczos_alg(
        hv, dim, dim, key, dynamic_unroll=False)
    assert jnp.max(jnp.abs(_safe_div(tr1 - tr2, tr2))) < 1e-4, (
        f'Seed {seed}: large relative error in Lanczos tridiag')
    assert jnp.max(jnp.abs(_safe_div(vecs1 - vecs2, vecs2))) < 1e-4, (
        f'Seed {seed}: large relative error in Lanczos vecs')

  def test_lanczos_dynamic_vs_static(self):
    for i in range(10):
      self._test_lanczos_dynamic_vs_static_once(seed=i)


class SdpDualPrimalTest(parameterized.TestCase):
  """Tests comparing SDP dual bounds to CVXPY exact solution of primal."""

  @unittest.skipUnless(MIP_SOLVERS, NO_MIP_SOLVERS_MESSAGE)
  def test_crossing_bounds(self):
    loss_margin = 1e-3
    seed = random.randint(1, 10000)
    verif_instance = test_utils.make_toy_verif_instance(seed)
    key = jax.random.PRNGKey(0)
    primal_opt, _ = cvxpy_verify.solve_mip_mlp_elided(verif_instance)
    dual_ub, _ = sdp_verify.solve_sdp_dual(
        problem.make_sdp_verif_instance(verif_instance), key, num_steps=1000)
    assert dual_ub > primal_opt - loss_margin, (
        'Dual upper bound should be greater than optimal primal objective.'
        f'Seed is {seed}. Vals are Dual: {dual_ub} Primal: {primal_opt}')

  def _test_tight_duality_gap(self, seed, loss_margin=0.003, num_steps=3000):
    verif_instance = test_utils.make_toy_verif_instance(
        seed, label=1, target_label=2)
    key = jax.random.PRNGKey(0)
    primal_opt, _ = cvxpy_verify.solve_sdp_mlp_elided(verif_instance)
    dual_ub, _ = sdp_verify.solve_sdp_dual(
        problem.make_sdp_verif_instance(verif_instance), key,
        num_steps=num_steps, verbose=False)
    assert dual_ub - primal_opt < loss_margin, (
        'Primal and dual vals should be close. '
        f'Seed: {seed}. Primal: {primal_opt}, Dual: {dual_ub}')
    assert dual_ub > primal_opt - 1e-3, 'crossing bounds'

  def test_tight_duality_gap(self):
    self._test_tight_duality_gap(0)

  def local_test_tight_duality_gap(self):  # pylint: disable=g-unreachable-test-method
    """Local test, meant to be run in parallel with --tests_per_run."""
    seed = random.randint(1, 10000)
    # 5/300 failures at loss_margin=0.01, 0/300 failures at loss_margin=0.3
    self._test_tight_duality_gap(seed, loss_margin=0.03, num_steps=3000)


class SdpVerifyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('MLP', 'mlp'),
      ('CNN', 'cnn')
  )
  def test_sdp_dual_simple_no_crash(self, model_type):
    verif_instance = test_utils.make_toy_verif_instance(
        seed=0, target_label=1, label=2, nn=model_type)
    kwargs = {
        'key': jax.random.PRNGKey(0),
        'opt': optax.adam(1e-3),
        'num_steps': 10,
        'eval_every': 5,
        'verbose': False,
        'use_exact_eig_eval': False,
        'use_exact_eig_train': False,
        'n_iter_lanczos': 5,
        'kappa_reg_weight': 1e-5,
        'kappa_zero_after': 8,
        'device_type': None,
    }
    verif_instance = problem.make_sdp_verif_instance(verif_instance)
    # Check all kwargs work.
    dual_val, _ = sdp_verify.solve_sdp_dual_simple(verif_instance, **kwargs)
    assert isinstance(dual_val, float)
    # Check code runs without kwargs.
    dual_val, _ = sdp_verify.solve_sdp_dual_simple(verif_instance, num_steps=5)
    assert isinstance(dual_val, float)

  def test_dual_sdp_no_crash(self):
    for nn in ['cnn', 'mlp']:
      verif_instance = test_utils.make_toy_verif_instance(
          seed=0, target_label=1, label=2, nn=nn)
      key = jax.random.PRNGKey(0)
      dual_val, _ = sdp_verify.solve_sdp_dual(
          problem.make_sdp_verif_instance(verif_instance), key, num_steps=10,
          n_iter_lanczos=5)
    assert isinstance(dual_val, float)

  def test_correct_dual_var_types(self):
    for nn in ['cnn', 'mlp']:
      verif_instance = test_utils.make_toy_verif_instance(
          seed=0, target_label=1, label=2, nn=nn)
      key = jax.random.PRNGKey(0)
      dual_vars = sdp_verify.init_duals(
          problem.make_sdp_verif_instance(verif_instance), key)
      assert len(dual_vars) == 3, 'Input, one hidden layer, kappa'
      assert isinstance(dual_vars[0], problem.DualVar)
      assert isinstance(dual_vars[1], problem.DualVarFin)
      assert isinstance(dual_vars[2], jax.Array)

  def test_ibp_init_matches_ibp_bound(self):
    for nn in ['cnn', 'mlp']:
      for seed in range(20):
        orig_verif_instance = test_utils.make_toy_verif_instance(seed, nn=nn)
        key = jax.random.PRNGKey(0)
        verif_instance = problem.make_sdp_verif_instance(orig_verif_instance)
        dual_vars = jax.tree_map(lambda s: None if s is None else jnp.zeros(s),
                                 verif_instance.dual_shapes)
        dual_vars = sdp_verify.init_duals_ibp(verif_instance, dual_vars)
        dual_loss = sdp_verify.dual_fun(
            verif_instance, dual_vars, key, exact=True)
        ibp_bound = utils.ibp_bound_elided(orig_verif_instance)
        assert abs(dual_loss - ibp_bound) < 1e-4, (
            f'Loss at initialization should match IBP: {dual_loss} {ibp_bound}')


class SdpVerifyTestCNNvsMLP(parameterized.TestCase):

  @unittest.skipIf('TRAVIS' in os.environ and os.environ['TRAVIS'] == 'true',
                   'Test produces nans on Travis CI but passes locally.')
  def test_cnn_mlp_match_fixed_window(self):
    num_steps = 1000
    for seed in range(1):
      verif_instance = test_utils.make_toy_verif_instance(seed, nn='cnn_simple')
      key = jax.random.PRNGKey(0)
      params_cnn = verif_instance.params_full
      in_shape = int(np.prod(np.array(params_cnn[0]['W'].shape[:-1])))
      # Input and filter size match -> filter is applied at just one location.
      # Number of layer 1 neurons = no. channels out of conv filter (last dim).
      out_shape = params_cnn[0]['W'].shape[-1]
      params_mlp = [(jnp.reshape(params_cnn[0]['W'],
                                 (in_shape, out_shape)), params_cnn[0]['b']),
                    (params_cnn[1][0], params_cnn[1][1])]
      bounds_mlp = sdp_verify.boundprop(
          params_mlp,
          sdp_verify.IntBound(
              lb=np.zeros((1, in_shape)),
              ub=1 * np.ones((1, in_shape)),
              lb_pre=None,
              ub_pre=None))
      verif_instance_mlp = utils.make_nn_verif_instance(params_mlp, bounds_mlp)
      dual_ub_cnn, _ = sdp_verify.solve_sdp_dual(
          problem.make_sdp_verif_instance(verif_instance), key,
          num_steps=num_steps, verbose=False, use_exact_eig_train=True)
      dual_ub_mlp, _ = sdp_verify.solve_sdp_dual(
          problem.make_sdp_verif_instance(verif_instance_mlp), key,
          num_steps=num_steps, verbose=False, use_exact_eig_train=True)
      assert abs(dual_ub_cnn - dual_ub_mlp) < 1e-2, (
          'Dual upper bound for MLP and CNN (simple CNN) should match.'
          f'Seed is {seed}. Vals are CNN: {dual_ub_cnn} MLP: {dual_ub_mlp}')
      # Error below 1e-4 when run-locally with steps > 3000.
      # Setting error here to 1e-2 with 500 steps for faster unit-tests

  def test_cnn_mlp_match_sliding_window(self):
    num_steps = 1000
    for seed in range(1):
      verif_instance = test_utils.make_toy_verif_instance(seed, nn='cnn_slide')
      key = jax.random.PRNGKey(0)
      lb, ub, params_mlp = test_utils.make_mlp_verif_instance_from_cnn(
          verif_instance)
      bounds_mlp = sdp_verify.boundprop(
          params_mlp,
          sdp_verify.IntBound(lb=lb, ub=ub, lb_pre=None, ub_pre=None))
      verif_instance_mlp = utils.make_nn_verif_instance(params_mlp, bounds_mlp)
      dual_ub_cnn, _ = sdp_verify.solve_sdp_dual(
          problem.make_sdp_verif_instance(verif_instance),
          key,
          num_steps=num_steps,
          verbose=False,
          use_exact_eig_train=True)
      dual_ub_mlp, _ = sdp_verify.solve_sdp_dual(
          problem.make_sdp_verif_instance(verif_instance_mlp),
          key,
          num_steps=num_steps,
          verbose=False,
          use_exact_eig_train=True)
      assert abs(dual_ub_cnn - dual_ub_mlp) < 5e-3, (
          'Dual upper bound for MLP and CNN (sliding filter) should match.'
          f'Seed is {seed}. Vals are CNN: {dual_ub_cnn} MLP: {dual_ub_mlp}')

if __name__ == '__main__':
  absltest.main()
