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

"""Tests for simplifying network computation graphs."""
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives


class ActivationDetectorTest(parameterized.TestCase):

  def test_softplus_detection(self):

    def softplus_model(inp):
      return jax.nn.softplus(inp)
    inp = jnp.array([[-2., 3.]])

    jaxpr_maker = jax.make_jaxpr(softplus_model)
    parsed = jaxpr_maker(inp)

    simplifier = synthetic_primitives.activation_detector
    # Let's check if the Softplus gets identified. This imitates the recursive
    # parsing done in bound_propagation, because depending on the platform,
    # the softplus code might be wrapped in a `custom_jvp_call_jaxpr_p`
    # The loop is necessarily terminating because we always remove one level of
    # nesting in the graph, so we will necessarily reach a level with no
    # subgraph.
    found_softplus = False
    graph = parsed.jaxpr
    while True:
      simple_graph = simplifier(graph)
      sub_graph = bound_propagation._sub_graph(simple_graph.eqns[0])
      if sub_graph is None:
        # This is an actual computation. Let's check if it's indeed the
        # synthetic softplus we expect.
        for eqn in simple_graph.eqns:
          if eqn.primitive == synthetic_primitives.softplus_p:
            found_softplus = True
        break
      else:
        graph = sub_graph

    self.assertTrue(found_softplus)


if __name__ == '__main__':
  absltest.main()
