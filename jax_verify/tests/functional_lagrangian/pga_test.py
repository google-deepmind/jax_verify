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

"""Unit-test for projected gradient ascent."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import lagrangian_form as lag_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.functional_lagrangian.inner_solvers import pga
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import numpy as np

LagrangianForm = lag_form.LagrangianForm
X_SHAPE = [1, 7]


class QuadLinDiagForm(LagrangianForm):

  def _init_params_per_sample(self):
    return None

  def _apply(self, x, lp, step=None):
    return jnp.sum(0.5 * lp[0] * x**2 + lp[1] * x)

  def __init__(self):
    self._name = 'QuadLinDiag'


class QuadDiagForm(LagrangianForm):

  def _init_params_per_sample(self):
    return None

  def _apply(self, x, lp, step=None):
    return jnp.sum(0.5 * lp * x**2)

  def __init__(self):
    self._name = 'QuadDiag'


class PGATest(chex.TestCase):

  def setUp(self):
    super().setUp()

    self.prng_key = jax.random.PRNGKey(1234)

    self.bounds = [
        sdp_utils.IntBound(
            lb=-jnp.ones(X_SHAPE),
            ub=jnp.ones(X_SHAPE),
            lb_pre=None,
            ub_pre=None),
        sdp_utils.IntBound(lb=None, ub=None, lb_pre=None, ub_pre=None),
    ]

  def test_intermediate_problem(self):
    prng_keys = jax.random.split(self.prng_key, 7)

    # create a decomposable quadratic problem (see value_per_component further)
    # we use absolute values to ensure that the resulting problem is concave,
    # so that PGA finds the global solution.
    weight_diag = jnp.abs(jax.random.normal(prng_keys[0], X_SHAPE[1:]))
    bias = jax.random.normal(prng_keys[1], X_SHAPE[1:])

    lp_pre_quad = jnp.abs(jax.random.normal(prng_keys[2], X_SHAPE[1:]))
    lp_pre_lin = jax.random.normal(prng_keys[3], X_SHAPE[1:])

    lp_post_quad = -jnp.abs(jax.random.normal(prng_keys[4], X_SHAPE[1:]))
    lp_post_lin = jax.random.normal(prng_keys[5], X_SHAPE[1:])

    lp_pre = (lp_pre_quad, lp_pre_lin)
    lp_post = (lp_post_quad, lp_post_lin)

    affine_fn = lambda x: x * weight_diag + bias
    lagrangian_form = QuadLinDiagForm()

    opt_instance = verify_utils.InnerVerifInstance(
        affine_fns=[affine_fn],
        bounds=self.bounds,
        is_first=False,
        is_last=False,
        lagrangian_form_pre=lagrangian_form,
        lagrangian_form_post=lagrangian_form,
        lagrange_params_pre=lp_pre,
        lagrange_params_post=lp_post,
        idx=0,
        spec_type=verify_utils.SpecType.ADVERSARIAL,
        affine_before_relu=True)

    # run PGA on problem
    pga_opt = pga.PgaStrategy(n_iter=200, lr=0.01)
    value_pga = pga_opt.solve_max(
        inner_dual_vars=None,
        opt_instance=opt_instance,
        key=prng_keys[6],
        step=0)

    def value_per_component(x):
      """Objective function per component."""
      y = jax.nn.relu(weight_diag * x + bias)
      return (0.5 * lp_post_quad * y**2 + lp_post_lin * y -
              (0.5 * lp_pre_quad * x**2 + lp_pre_lin * x))

    # closed-form unconstrained solution if relu is passing
    x_opt_passing = (lp_pre_lin - lp_post_lin * weight_diag -
                     lp_post_quad * weight_diag * bias) / (
                         weight_diag**2 * lp_post_quad - lp_pre_quad)
    # project on feasible set where relu is passing
    x_opt_passing = jnp.clip(
        x_opt_passing,
        a_min=jnp.maximum(-bias / weight_diag, self.bounds[0].lb),
        a_max=self.bounds[0].ub)
    value_opt_passing = value_per_component(x_opt_passing)

    # closed-form unconstrained solution if relu is non-passing
    x_opt_nonpassing = -lp_pre_lin / lp_pre_quad
    # project on feasible set where relu is not passing
    x_opt_nonpassing = jnp.clip(
        x_opt_nonpassing,
        a_min=self.bounds[0].lb,
        a_max=jnp.minimum(-bias / weight_diag, self.bounds[0].ub))
    value_opt_nonpassing = value_per_component(x_opt_nonpassing)

    # best of candidate solutions (each optimal on their subdomain) gives the
    # global solution
    x_opt = jnp.where(value_opt_passing > value_opt_nonpassing, x_opt_passing,
                      x_opt_nonpassing)

    # corresponding optimal objective value
    value_opt = jnp.sum(value_per_component(x_opt))

    np.testing.assert_almost_equal(value_pga, value_opt, decimal=2)

  def test_final_problem(self):
    prng_keys = jax.random.split(self.prng_key, 4)

    # create a decomposable quadratic problem (see value_per_component further)
    # we use absolute values to ensure that the resulting problem is concave,
    # so that PGA finds the global solution.
    objective = jax.random.normal(prng_keys[0], X_SHAPE[1:])
    constant = jnp.zeros([])
    lp_pre = jnp.abs(jax.random.normal(prng_keys[2], X_SHAPE[1:]))

    affine_fn = lambda x: jnp.sum(x * objective) + constant
    lagrangian_form = QuadDiagForm()

    opt_instance = verify_utils.InnerVerifInstance(
        affine_fns=[affine_fn],
        bounds=self.bounds,
        lagrangian_form_pre=lagrangian_form,
        lagrangian_form_post=lagrangian_form,
        is_first=False,
        is_last=True,
        lagrange_params_pre=lp_pre,
        lagrange_params_post=None,
        idx=0,
        spec_type=verify_utils.SpecType.ADVERSARIAL,
        affine_before_relu=True)

    # run PGA on problem
    pga_opt = pga.PgaStrategy(n_iter=500, lr=0.01)
    value_pga = pga_opt.solve_max(
        inner_dual_vars=None,
        opt_instance=opt_instance,
        key=prng_keys[3],
        step=0)

    def value_per_component(x):
      """Objective function per component."""
      return -0.5 * lp_pre * x**2 + objective * x

    # closed-form solution for the decomposable problem
    x_opt = jnp.clip(objective / lp_pre, a_min=-1, a_max=1)

    # corresponding optimal objective value
    value_opt = jnp.sum(value_per_component(x_opt))

    np.testing.assert_almost_equal(value_pga, value_opt, decimal=3)

  def test_integration_combined_layer(self):
    prng_keys = jax.random.split(self.prng_key, 4)

    dim_1 = X_SHAPE[1]
    dim_2 = dim_1 + 1
    weights_1 = jax.random.normal(prng_keys[0], [dim_1, dim_2])
    bias_1 = jnp.zeros([dim_2])
    lp_pre = jnp.abs(jax.random.normal(prng_keys[1], [1, dim_1]))

    lagrangian_form = QuadDiagForm()

    weights_2 = jax.random.normal(prng_keys[2], [dim_2, 1])
    bias_2 = jnp.zeros([])
    bounds_2 = [
        sdp_utils.IntBound(
            lb=-jnp.ones([1, dim_2]),
            ub=jnp.ones([1, dim_2]),
            lb_pre=None,
            ub_pre=None),
        sdp_utils.IntBound(lb=None, ub=None, lb_pre=None, ub_pre=None),
    ]

    opt_instance_1 = verify_utils.InnerVerifInstance(
        affine_fns=[lambda x: x @ weights_1 + bias_1],
        bounds=self.bounds,
        lagrangian_form_pre=lagrangian_form,
        lagrangian_form_post=lagrangian_form,
        is_first=False,
        is_last=False,
        lagrange_params_pre=lp_pre,
        lagrange_params_post=None,
        idx=0,
        spec_type=verify_utils.SpecType.ADVERSARIAL,
        affine_before_relu=True)

    opt_instance_2 = verify_utils.InnerVerifInstance(
        affine_fns=[lambda x: x @ weights_2 + bias_2],
        bounds=bounds_2,
        lagrangian_form_pre=lagrangian_form,
        lagrangian_form_post=lagrangian_form,
        is_first=False,
        is_last=True,
        lagrange_params_pre=None,
        lagrange_params_post=None,
        idx=1,
        spec_type=verify_utils.SpecType.ADVERSARIAL,
        affine_before_relu=True)

    opt_instance = dual_build._merge_instances(
        opt_instance_1,
        opt_instance_2,
    )

    # run PGA on problem
    pga_opt = pga.PgaStrategy(n_iter=5, lr=0.01)
    pga_opt.solve_max(
        inner_dual_vars=None,
        opt_instance=opt_instance,
        key=prng_keys[3],
        step=None)


if __name__ == '__main__':
  absltest.main()
