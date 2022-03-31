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

"""Unit-test for uncertainty spec inner max."""

from absl.testing import absltest
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import lagrangian_form as lag_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.functional_lagrangian.inner_solvers import uncertainty_spec
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import numpy as np

X_SHAPE = [1, 7]


class UncertaintySpecTest(chex.TestCase):

  def setUp(self):
    super(UncertaintySpecTest, self).setUp()

    self._prng_seq = hk.PRNGSequence(13579)
    self._n_classes = X_SHAPE[1]

    self.bounds = [
        sdp_utils.IntBound(
            lb_pre=-0.1 * jnp.ones(X_SHAPE),
            ub_pre=0.1 * jnp.ones(X_SHAPE),
            lb=None,
            ub=None)
    ]

  def test_softmax_upper(self):
    rand_class = jax.random.randint(
        next(self._prng_seq), shape=(), minval=0, maxval=self._n_classes)
    objective = jnp.arange(self._n_classes) == rand_class
    constant = jax.random.uniform(next(self._prng_seq), ())

    affine_fn = lambda x: jnp.sum(x * objective) + constant

    lagrangian_form = lag_form.Linear()
    lp_pre = lagrangian_form.init_params(
        next(self._prng_seq), l_shape=X_SHAPE, init_zeros=False)

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
        spec_type=verify_utils.SpecType.UNCERTAINTY,
        affine_before_relu=True)

    # run PGA to find approximate max
    pga_opt = uncertainty_spec.UncertaintySpecStrategy(
        n_iter=10_000,
        n_pieces=0,
        solve_max=uncertainty_spec.MaxType.EXP,
    )

    value_pga = pga_opt.solve_max(
        inner_dual_vars=None,
        opt_instance=opt_instance,
        key=next(self._prng_seq),
        step=0)

    # use cvxpy to find upper bound
    cvx_opt = uncertainty_spec.UncertaintySpecStrategy(
        n_iter=0,
        n_pieces=10,
        solve_max=uncertainty_spec.MaxType.EXP_BOUND,
    )

    value_cvx = cvx_opt.solve_max(
        inner_dual_vars=None,
        opt_instance=opt_instance,
        key=next(self._prng_seq),
        step=0)

    # evaluate objective function on an arbitrarily chosen feasible point
    def objective_fn(x):
      return (jnp.squeeze(affine_fn(jax.nn.softmax(x)), ()) -
              jnp.squeeze(lagrangian_form.apply(x, lp_pre, step=0), ()))

    middle_x = 0.5 * self.bounds[0].lb_pre + 0.5 * self.bounds[0].ub_pre
    value_middle = objective_fn(middle_x)

    np.testing.assert_array_less(value_middle, value_pga)
    np.testing.assert_array_less(value_pga, value_cvx + 1e-5)


if __name__ == '__main__':
  absltest.main()
