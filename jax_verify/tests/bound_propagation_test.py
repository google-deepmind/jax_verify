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

"""Tests for propagating bounds through the network."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation
import tree


@hk.without_apply_rng
@hk.transform
def sequential_model(inp):
  net = hk.Sequential([
      hk.Linear(5), jax.nn.relu,
      hk.Linear(2), jax.nn.relu,
      hk.Linear(1)
  ])
  return net(inp)


def residual_model_all_act(inp):
  mod1 = hk.Linear(5)
  mod2 = hk.Linear(5)
  mod3 = hk.Linear(1)

  act_0 = inp
  act_1 = mod1(act_0)
  act_2 = jax.nn.relu(act_1)
  act_3 = mod2(act_2)
  act_4 = jax.nn.relu(act_3)
  act_5 = act_2 + act_4  # Residual
  final_act = mod3(act_5)
  return [act_0, act_1, act_2, act_3, act_4, act_5, final_act]


@hk.without_apply_rng
@hk.transform
def single_element_list_model(inp):
  all_acts = residual_model_all_act(inp)
  return [all_acts[-1]]


@hk.without_apply_rng
@hk.transform
def dict_output_model(inp):
  all_acts = residual_model_all_act(inp)
  return {i: act for i, act in enumerate(all_acts)}


@hk.without_apply_rng
@hk.transform
def residual_model_intermediate(inp):
  return residual_model_all_act(inp)


@hk.without_apply_rng
@hk.transform
def residual_model(inp):
  all_acts = residual_model_all_act(inp)
  return all_acts[-1]


class BoundPropagationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Sequential', sequential_model),
      ('Residual', residual_model)
  )
  def test_model_structure_nostate(self, model):
    z = jnp.array([[1., 2., 3.]])

    params = model.init(jax.random.PRNGKey(1), z)
    input_bounds = jax_verify.IntervalBound(z - 1.0, z + 1.0)
    fun_to_prop = functools.partial(model.apply, params)
    output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, input_bounds)
    self.assertTrue(all(output_bounds.upper >= output_bounds.lower))

  def test_multioutput_model(self):
    z = jnp.array([[1., 2., 3.]])

    fun = hk.without_apply_rng(
        hk.transform(residual_model_all_act, apply_rng=True))
    params = fun.init(jax.random.PRNGKey(1), z)
    input_bounds = jax_verify.IntervalBound(z - 1.0, z + 1.0)
    fun_to_prop = functools.partial(fun.apply, params)
    output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, input_bounds)

    self.assertLen(output_bounds, 7)

  @parameterized.named_parameters(
      ('Sequential', sequential_model),
      ('Residual', residual_model)
  )
  def test_tight_bounds_nostate(self, model):
    z = jnp.array([[1., 2., 3.]])

    params = model.init(jax.random.PRNGKey(1), z)

    tight_input_bounds = jax_verify.IntervalBound(z, z)
    fun_to_prop = functools.partial(model.apply, params)
    tight_output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, tight_input_bounds)

    model_eval = model.apply(params, z)

    # Because the input lower bound is equal to the input upper bound, the value
    # of the output bounds should be the same and correspond to the value of the
    # forward pass.
    self.assertAlmostEqual(tight_output_bounds.lower.tolist(),
                           tight_output_bounds.upper.tolist())

    self.assertAlmostEqual(tight_output_bounds.lower.tolist(),
                           model_eval.tolist())

  @parameterized.named_parameters(
      ('Sequential', sequential_model),
      ('Residual', residual_model),
      ('ResidualAll', residual_model_intermediate),
      ('1elt_list', single_element_list_model),
      ('dict_output', dict_output_model)
  )
  def test_matching_output_structure(self, model):
    def _check_matching_structures(output_tree, bound_tree):
      """Replace all bounds/arrays with True, then compare pytrees."""
      output_struct = tree.traverse(
          lambda x: True if isinstance(x, jnp.ndarray) else None, output_tree)
      bound_struct = tree.traverse(
          lambda x: True if isinstance(x, bound_propagation.Bound) else None,
          bound_tree)
      tree.assert_same_structure(output_struct, bound_struct)
    z = jnp.array([[1., 2., 3.]])
    params = model.init(jax.random.PRNGKey(1), z)
    input_bounds = jax_verify.IntervalBound(z - 1.0, z + 1.0)
    model_output = model.apply(params, z)
    fun_to_prop = functools.partial(model.apply, params)
    for boundprop_method in [
        jax_verify.interval_bound_propagation,
        jax_verify.crown_bound_propagation,
        jax_verify.fastlin_bound_propagation,
        jax_verify.ibpfastlin_bound_propagation,
    ]:
      output_bounds = boundprop_method(fun_to_prop, input_bounds)
      _check_matching_structures(model_output, output_bounds)


class StaticArgumentModel(hk.Module):

  def __init__(self):
    super().__init__()
    self.lin_1 = hk.Linear(5)
    self.lin_2 = hk.Linear(10)

  def __call__(self, inputs, use_2=True):
    if use_2:
      return self.lin_2(inputs)
    else:
      return self.lin_1(inputs)


class StaticArgumentsModelTest(parameterized.TestCase):

  def test_staticargument_last(self):

    @hk.without_apply_rng
    @hk.transform
    def forward(inputs, use_2):
      model = StaticArgumentModel()
      return model(inputs, use_2)

    z = jnp.array([[1., 2., 3.]])
    params = forward.init(jax.random.PRNGKey(1), z, True)
    input_bounds = jax_verify.IntervalBound(z-1.0, z+1.0)

    def fun_to_prop(inputs):
      return forward.apply(params, inputs, True)

    output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, input_bounds)
    self.assertTrue((output_bounds.upper >= output_bounds.lower).all())

  def test_staticargument_first(self):

    @hk.without_apply_rng
    @hk.transform
    def forward(use_2, inputs):
      model = StaticArgumentModel()
      return model(inputs, use_2)

    z = jnp.array([[1., 2., 3.]])
    params = forward.init(jax.random.PRNGKey(1), True, z)
    input_bounds = jax_verify.IntervalBound(z-1.0, z+1.0)

    fun_to_prop = functools.partial(forward.apply, params, True)

    output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, input_bounds)
    self.assertTrue((output_bounds.upper >= output_bounds.lower).all())

  def test_keywords_argument(self):

    @hk.without_apply_rng
    @hk.transform
    def forward(inputs, use_2=False):
      model = StaticArgumentModel()
      return model(inputs, use_2)

    z = jnp.array([[1., 2., 3.]])
    params = forward.init(jax.random.PRNGKey(1), z, use_2=True)
    input_bounds = jax_verify.IntervalBound(z-1.0, z+1.0)
    fun_to_prop = functools.partial(forward.apply, params, use_2=True)
    output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, input_bounds)
    self.assertTrue((output_bounds.upper >= output_bounds.lower).all())


class ModelWithState(hk.Module):

  def __init__(self):
    super().__init__()
    self.lin_1 = hk.Linear(5)
    self.bn_1 = hk.BatchNorm(True, True, decay_rate=0.999)

  def __call__(self, inputs, is_training, test_local_stats=False):
    act = self.lin_1(inputs)
    bn_act = self.bn_1(act, is_training, test_local_stats)
    return bn_act


class StatefulModelTest(parameterized.TestCase):

  def test_stateful_model(self):

    @hk.transform_with_state
    def forward(inputs, is_training, test_local_stats=False):
      model = ModelWithState()
      return model(inputs, is_training, test_local_stats)

    z = jnp.array([[1., 2., 3.]])
    params, state = forward.init(jax.random.PRNGKey(1), z, True, False)

    def fun_to_prop(inputs):
      outs = forward.apply(params, state, jax.random.PRNGKey(1),
                           inputs, False, False)
      # Ignore the outputs that are not the network outputs.
      return outs[0]

    input_bounds = jax_verify.IntervalBound(z-1.0, z+1.0)
    # Consider as static the state, the random generator, and the flags
    output_bounds = jax_verify.interval_bound_propagation(
        fun_to_prop, input_bounds)
    self.assertTrue((output_bounds.upper >= output_bounds.lower).all())


if __name__ == '__main__':
  absltest.main()
