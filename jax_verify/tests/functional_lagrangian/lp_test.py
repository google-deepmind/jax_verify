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

"""Unit-test for linear Lagrangian."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.extensions.functional_lagrangian import dual_build
from jax_verify.extensions.functional_lagrangian import dual_solve
from jax_verify.extensions.functional_lagrangian import lagrangian_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.functional_lagrangian.inner_solvers import lp
from jax_verify.extensions.sdp_verify import utils as sdp_utils
from jax_verify.src import bound_propagation
from jax_verify.src.mip_solver import cvxpy_relaxation_solver
from jax_verify.src.mip_solver import relaxation
from jax_verify.tests.sdp_verify import test_utils as sdp_test_utils
import ml_collections
import numpy as np

NUM_SAMPLES = 1
LAYER_SIZES = [3, 4, 5, 6]


def create_inputs(prng_key):
  return jax.random.uniform(
      prng_key, [NUM_SAMPLES, LAYER_SIZES[0]], minval=0.0, maxval=1.0)


def make_model_fn(params):

  def model_fn(inputs):
    inputs = np.reshape(inputs, (inputs.shape[0], -1))
    return sdp_utils.predict_mlp(params, inputs)

  return model_fn


def get_config():
  config = ml_collections.ConfigDict()

  config.outer_opt = ml_collections.ConfigDict()
  config.outer_opt.lr_init = 0.001
  config.outer_opt.steps_per_anneal = 500
  config.outer_opt.anneal_lengths = ''
  config.outer_opt.anneal_factor = 0.1
  config.outer_opt.num_anneals = 1
  config.outer_opt.opt_name = 'adam'
  config.outer_opt.opt_kwargs = {}

  return config


class LinearTest(chex.TestCase):

  def setUp(self):
    super(LinearTest, self).setUp()

    self.target_label = 1
    self.label = 0
    self.input_bounds = (0.0, 1.0)
    self.layer_sizes = LAYER_SIZES
    self.eps = 0.1

    prng_key = jax.random.PRNGKey(13579)

    self.keys = jax.random.split(prng_key, 5)
    self.network_params = sdp_test_utils.make_mlp_params(
        self.layer_sizes, self.keys[0])

    self.inputs = create_inputs(self.keys[1])

    objective = jnp.zeros(self.layer_sizes[-1])
    objective = objective.at[self.target_label].add(1)
    objective = objective.at[self.label].add(-1)
    self.objective = objective
    self.objective_bias = jax.random.normal(self.keys[2], [])

  def solve_with_jax_verify(self):
    lower_bound = jnp.minimum(jnp.maximum(self.inputs - self.eps, 0.0), 1.0)
    upper_bound = jnp.minimum(jnp.maximum(self.inputs + self.eps, 0.0), 1.0)
    init_bound = jax_verify.IntervalBound(lower_bound, upper_bound)

    logits_fn = make_model_fn(self.network_params)

    solver = cvxpy_relaxation_solver.CvxpySolver
    relaxation_transform = relaxation.RelaxationTransform(
        jax_verify.ibp_transform)

    var, env = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(relaxation_transform),
        logits_fn, init_bound)

    # This solver minimizes the objective -> get max with -min(-objective)
    neg_value_opt, _, _ = relaxation.solve_relaxation(
        solver,
        -self.objective,
        -self.objective_bias,
        var,
        env,
        index=0,
        time_limit_millis=None)
    value_opt = -neg_value_opt

    return value_opt

  def solve_with_functional_lagrangian(self):
    config = get_config()

    init_bound = sdp_utils.init_bound(
        self.inputs[0], self.eps, input_bounds=self.input_bounds)
    bounds = sdp_utils.boundprop(
        self.network_params + [(self.objective, self.objective_bias)],
        init_bound)

    logits_fn = make_model_fn(self.network_params)

    def spec_fn(inputs):
      return jnp.matmul(logits_fn(inputs), self.objective) + self.objective_bias

    input_bounds = jax_verify.IntervalBound(bounds[0].lb, bounds[0].ub)

    lagrangian_form_per_layer = lagrangian_form.Linear()
    lagrangian_form_per_layer = [lagrangian_form_per_layer for bd in bounds]
    inner_opt = lp.LpStrategy()
    env, dual_params, dual_params_types = inner_opt.init_duals(
        jax_verify.ibp_transform, verify_utils.SpecType.ADVERSARIAL, False,
        spec_fn, self.keys[3], lagrangian_form_per_layer, input_bounds)
    opt, num_steps = dual_build.make_opt_and_num_steps(config.outer_opt)
    dual_state = ml_collections.ConfigDict(type_safe=False)
    dual_solve.solve_dual_train(
        env,
        key=self.keys[4],
        num_steps=num_steps,
        opt=opt,
        dual_params=dual_params,
        dual_params_types=dual_params_types,
        dual_state=dual_state,
        affine_before_relu=False,
        spec_type=verify_utils.SpecType.ADVERSARIAL,
        inner_opt=inner_opt,
        logger=(lambda *args: None),
    )

    return dual_state.loss

  def test_lp_against_jax_verify_relaxation(self):
    value_jax_verify = self.solve_with_jax_verify()
    value_functional_lagrangian = self.solve_with_functional_lagrangian()

    np.testing.assert_allclose(
        value_jax_verify, value_functional_lagrangian, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
