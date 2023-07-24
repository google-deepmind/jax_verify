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

"""Test the extraction of concretizing values."""

from absl.testing import absltest
import chex

import jax.numpy as jnp
from jax_verify.src import ibp
from jax_verify.src.branching import branch_utils
from jax_verify.src.linear import linear_relaxations


class ConcretizingInputTest(chex.TestCase):

  def test_linf_bound_concretizing_inputs(self):

    # Create a stack of two linear expression to test things.
    linexp = linear_relaxations.LinearExpression(
        jnp.stack([jnp.ones((2, 3)),
                   -jnp.ones((2, 3))]),
        jnp.zeros((2,)))

    input_bound = ibp.IntervalBound(-2 * jnp.ones((2, 3)),
                                    2 * jnp.ones((2, 3)))

    concretizing_inp = branch_utils.minimizing_concretizing_input(
        linexp, input_bound)
    # Check that the shape of the concretizing inp for each linexp is of the
    # shape of the input.
    chex.assert_shape(concretizing_inp, (2, 2, 3))
    # Evaluating the bound given by the concretizing inp
    bound_by_concinp = ((linexp.lin_coeffs * concretizing_inp).sum(axis=(1, 2))
                        + linexp.offset)

    # Result by concretizing directly:
    concretized_bound = linear_relaxations.concretize_linear_expression(
        linexp, input_bound)

    chex.assert_trees_all_close(bound_by_concinp, concretized_bound)

if __name__ == '__main__':
  absltest.main()
