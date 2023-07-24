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

"""Tests for simplifying network computation graphs."""
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import numpy as jnp
import jax_verify
from jax_verify.src import synthetic_primitives


class SyntheticPrimitiveDetectorTest(parameterized.TestCase):

  def _check_correct_impl(self, graph, simplifier, var_is_bound, *inps):
    simplified_graph = synthetic_primitives.simplify_graph(simplifier, graph,
                                                           var_is_bound)

    graph_outs = jax.core.eval_jaxpr(graph, [], *inps)
    simple_graph_outs = jax.core.eval_jaxpr(simplified_graph, [], *inps)

    for graph_out, simple_graph_out in zip(graph_outs, simple_graph_outs):
      self.assertAlmostEqual(jnp.abs(graph_out - simple_graph_out).max(),
                             0., delta=1e-6)

  def _find_eqn_in_simplified_graph(self, graph, simplifier, var_is_bound,
                                    primitive):
    # Check if the primitive is present. This imitates the recursive
    # parsing done in bound_propagation, because depending on the platform,
    # the primitive might be wrapped in a `custom_jvp_call_jaxpr_p`
    # The loop is necessarily terminating because we always remove one level of
    # nesting in the graph, so we will necessarily reach a level with no
    # subgraph.
    simplified_graph = synthetic_primitives.simplify_graph(simplifier, graph,
                                                           var_is_bound)
    for eqn in simplified_graph.eqns:
      if eqn.primitive in synthetic_primitives.SUBGRAPH_PRIMITIVES:
        sub_graph = synthetic_primitives.jax_primitive_subgraph(
            eqn.primitive, **eqn.params)
        subgraph_var_is_bound = {}
        for sub_invar, eqn_invar in zip(sub_graph.invars, eqn.invars):
          if isinstance(eqn_invar, jax.core.Literal):
            subgraph_var_is_bound[sub_invar] = False
          else:
            subgraph_var_is_bound[sub_invar] = var_is_bound[eqn_invar]
        match = self._find_eqn_in_simplified_graph(
            sub_graph, simplifier, subgraph_var_is_bound, primitive)
        if match:
          return match
      elif eqn.primitive == primitive:
        return eqn
    return None


class ActivationDetectorTest(SyntheticPrimitiveDetectorTest):

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_softplus_detected(self, use_jit):

    def softplus_model(inp):
      return jax.nn.softplus(inp)
    inp = jnp.array([[-2., 3.]])

    if use_jit:
      softplus_model = jax.jit(softplus_model)

    parsed = synthetic_primitives.make_jaxpr_nojit(softplus_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    found_softplus = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.softplus_p)
    self.assertIsNotNone(found_softplus)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_softmax_detected(self, use_jit):

    def softmax_model(x):
      return jax.nn.softmax(x)

    if use_jit:
      softmax_model = jax.jit(softmax_model)

    inp = jnp.array([[-2., 3.]])

    parsed = synthetic_primitives.make_jaxpr_nojit(softmax_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.softmax_p)
    self.assertIsNotNone(match)
    self.assertEqual(match.params['axis'], 1)  # pytype: disable=attribute-error

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_softmax_expended(self, use_jit):

    def softmax_model(x):
      return jax.nn.softmax(x)

    if use_jit:
      softmax_model = jax.jit(softmax_model)

    inp = jnp.array([[-2., 3.],
                     [3., 3.1],
                     [-2., -2.],
                     [3., 3.]])

    parsed = synthetic_primitives.make_jaxpr_nojit(softmax_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    expand_softmax_simplifier = synthetic_primitives.simplifier_composition(
        synthetic_primitives.activation_simplifier,
        synthetic_primitives.expand_softmax_simplifier)

    # Check that all of the components of the softmax that we would expect to
    # find are present.
    softmax_primitives = (
        lax.exp_p,
        lax.reduce_sum_p,
        lax.broadcast_in_dim_p,
        synthetic_primitives.posreciprocal_p,
        lax.mul_p)
    for prim_to_match in softmax_primitives:
      match = self._find_eqn_in_simplified_graph(
          parsed.jaxpr,
          expand_softmax_simplifier,
          var_is_bound,
          prim_to_match)
      self.assertIsNotNone(match)

    self._check_correct_impl(
        parsed.jaxpr, expand_softmax_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_relu_detected(self, use_jit):

    def relu_model(x):
      return jax.nn.relu(x)

    if use_jit:
      relu_model = jax.jit(relu_model)

    inp = jnp.array([[-2., 3.]])
    parsed = synthetic_primitives.make_jaxpr_nojit(relu_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.relu_p)

    self.assertIsNotNone(match)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_max_with_zero_detected_as_relu(self, use_jit):

    def relu_model(x):
      return jnp.maximum(x, 0.)

    if use_jit:
      relu_model = jax.jit(relu_model)

    inp = jnp.array([[-2., 3.]])
    parsed = synthetic_primitives.make_jaxpr_nojit(relu_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.relu_p)

    self.assertIsNotNone(match)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_max_with_one_not_mistaken_for_relu(self, use_jit):

    def notrelu_model(x):
      return jnp.maximum(x, 1.)

    if use_jit:
      notrelu_model = jax.jit(notrelu_model)

    inp = jnp.array([[-2., 3.]])
    parsed = synthetic_primitives.make_jaxpr_nojit(notrelu_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.relu_p)

    self.assertIsNone(match)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_leaky_relu_detected(self, use_jit):

    def leaky_relu_model(x):
      return jax.nn.leaky_relu(x)

    if use_jit:
      leaky_relu_model = jax.jit(leaky_relu_model)

    inp = jnp.array([[-2., 3.]])
    parsed = synthetic_primitives.make_jaxpr_nojit(leaky_relu_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.leaky_relu_p)

    self.assertIsNotNone(match)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_sigmoid_detected(self, use_jit):

    def sigmoid_model(x):
      return jax.nn.sigmoid(x)

    if use_jit:
      sigmoid_model = jax.jit(sigmoid_model)

    inp = jnp.array([[-2., 3.]])
    parsed = synthetic_primitives.make_jaxpr_nojit(sigmoid_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.sigmoid_p)

    self.assertIsNotNone(match)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_clip_detected(self, use_jit):

    def clip_model(x):
      return jnp.clip(x, a_min=0., a_max=1.)

    if use_jit:
      clip_model = jax.jit(clip_model)

    inp = jnp.array([[-2., 3., 0.5]])
    parsed = synthetic_primitives.make_jaxpr_nojit(clip_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.clip_p)

    self.assertIsNotNone(match)

    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.activation_simplifier,
        var_is_bound, inp)

  @parameterized.product(
      tested_fun=[jnp.minimum, jnp.maximum],
      use_jit=[True, False],
      include_linear=[True, False],
      both_inp_bounds=[True, False],
      broadcasting=[True, False])
  def test_elementwise_minmax_replaced(
      self,
      tested_fun,
      use_jit,
      include_linear,
      both_inp_bounds,
      broadcasting,
  ):
    def model_fun(inp_0, inp_1):
      if include_linear:
        lin_weight = jax.random.uniform(jax.random.PRNGKey(0),
                                        inp_0.shape)
        act = inp_0 * lin_weight
      else:
        act = inp_0
      return tested_fun(act, inp_1)

    if use_jit:
      model_fun = jax.jit(model_fun)

    if broadcasting:
      shape_0 = (1, 8)
      shape_1 = (7, 1)
    else:
      shape_0 = (2, 4)
      shape_1 = (2, 4)
    inp_0 = jax.random.uniform(jax.random.PRNGKey(0), shape_0)
    inp_1 = jax.random.uniform(jax.random.PRNGKey(0), shape_1)

    parsed = synthetic_primitives.make_jaxpr_nojit(model_fun, inp_0, inp_1)
    var_is_bound = {parsed.jaxpr.invars[0]: True,
                    parsed.jaxpr.invars[1]: both_inp_bounds}
    # Check that this is rewritten using a ReLU.
    relu_match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.default_simplifier,
        var_is_bound, synthetic_primitives.relu_p)
    self.assertIsNotNone(relu_match)
    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.default_simplifier, var_is_bound,
        inp_0, inp_1)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_linear_detected(self, use_jit):

    inp = jnp.array([[-1., 1.]])
    key = jax.random.PRNGKey(0)
    key_w, key_b = jax.random.split(key, 2)
    w1 = jax.random.uniform(key_w, shape=(2, 5))
    b1 = jax.random.uniform(key_b, shape=(5,))

    def linear_model(inp, w1, b1):
      """Linear function involving several different linear operators."""
      y = inp @ w1 + b1
      centered_y = y - y.mean()
      return centered_y.sum()

    if use_jit:
      linear_model = jax.jit(linear_model)

    parsed = synthetic_primitives.make_jaxpr_nojit(linear_model, inp, w1, b1)
    var_is_bound = {invar: is_bound for invar, is_bound
                    in zip(parsed.jaxpr.invars, [True, False, False])}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.group_linear_sequence,
        var_is_bound,
        synthetic_primitives.linear_p
    )

    # Check that all the components that we expect are there.
    linear_subgraph = match.params['jax_verify_subgraph']
    subgraph_primitives = [eqn.primitive for eqn in linear_subgraph.eqns]

    self.assertIn(lax.dot_general_p, subgraph_primitives)
    self.assertIn(lax.add_p, subgraph_primitives)
    # There is two reduce_sum, one for the final sum, one for the mean.
    self.assertEqual(2, sum(prim == lax.reduce_sum_p
                            for prim in subgraph_primitives))
    # The mean also introduce a div.
    self.assertIn(lax.div_p, subgraph_primitives)
    self.assertIn(lax.sub_p, subgraph_primitives)

    # Let's check that the simplification has not modified the behaviour of the
    # model and can be forwarded through.
    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.group_linear_sequence,
        var_is_bound, inp, w1, b1)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_fusedrelu_detected(self, use_jit):

    inp = jnp.array([-1., 1.])
    key = jax.random.PRNGKey(0)

    key_w, key_b = jax.random.split(key, 2)
    w = jax.random.uniform(key_w, shape=(2, 5))
    b = jax.random.uniform(key_b, shape=(5,))

    def net_model(inp, w, b):
      return jax.nn.relu(inp @ w + b)

    if use_jit:
      net_model = jax.jit(net_model)

    parsed = synthetic_primitives.make_jaxpr_nojit(net_model, inp, w, b)
    var_is_bound = {invar: is_bound for invar, is_bound
                    in zip(parsed.jaxpr.invars, [True] + [False] * 2)}

    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.fused_relu_simplifier,
        var_is_bound,
        synthetic_primitives.fused_relu_p)

    self.assertIsNotNone(match)

    # Let's check that the simplification has not modified the behaviour of the
    # model and can be forwarded through.
    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.fused_relu_simplifier,
        var_is_bound, inp, w, b)

  @parameterized.named_parameters(('jit', True), ('nojit', False))
  def test_support_weaktype_input(self, use_jit):

    inp = jnp.asarray(0.)

    def net_model(inp):
      return jnp.zeros(()) * inp

    if use_jit:
      net_model = jax.jit(net_model)

    parsed = synthetic_primitives.make_jaxpr_nojit(net_model, inp)
    var_is_bound = {parsed.jaxpr.invars[0]: True}
    match = self._find_eqn_in_simplified_graph(
        parsed.jaxpr,
        synthetic_primitives.activation_simplifier,
        var_is_bound,
        synthetic_primitives.convert_float32_p)

    self.assertIsNotNone(match)

    # Let's check that the simplification has not modified the behaviour of the
    # model and can be forwarded through.
    self._check_correct_impl(
        parsed.jaxpr, synthetic_primitives.fused_relu_simplifier,
        var_is_bound, inp)

    # Let's check that propagating bounds through this does not cause errors.
    jax_verify.interval_bound_propagation(
        net_model, jax_verify.IntervalBound(inp, inp))

if __name__ == '__main__':
  absltest.main()
