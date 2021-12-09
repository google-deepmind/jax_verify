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

"""Mccormick relaxations for bilinear terms.

Create mc-cormick relaxations of bilinear terms for boundprop and verification.
"""
from typing import Tuple

import jax.numpy as jnp

Tensor = jnp.ndarray


def mccormick_ibp(
    lx: Tensor,
    ux: Tensor,
    ly: Tensor,
    uy: Tensor,
    matrix: Tensor,
) -> Tuple[Tensor, Tensor]:
  """Compute bounds in x.T * matrix * y s.t x in [lx, ux], y in [ly, uy].

  Args:
    lx: Lower bounds on x (d,)
    ux: Upper bounds on x (d,)
    ly: Lower bounds on y (d,)
    uy: Upper bounds on y (d,)
    matrix: (d, d) matrix

  Returns:
    Lower and upper bounds on x.T * matrix * y
  """
  ll = matrix * jnp.outer(lx, ly)
  lu = matrix * jnp.outer(lx, uy)
  ul = matrix * jnp.outer(ux, ly)
  uu = matrix * jnp.outer(ux, uy)

  lb_elementwise = jnp.minimum(jnp.minimum(ll, lu), jnp.minimum(ul, uu))
  ub_elementwise = jnp.maximum(jnp.maximum(ll, lu), jnp.maximum(ul, uu))
  return jnp.sum(lb_elementwise), jnp.sum(ub_elementwise)


def mccormick_outer_product(x: Tensor,
                            y: Tensor,
                            x_lb: Tensor,
                            x_ub: Tensor,
                            y_lb: Tensor,
                            y_ub: Tensor,
                            ) -> Tensor:
  """McCormick Upper Bound on bilinear term x @ y.T.

  Args:
    x: Input tensor
    y: Input tensor
    x_lb: Lower bounds on x
    x_ub: Upper bounds on x
    y_lb: Upper bounds on y
    y_ub: Upper bounds on y

  Returns:
    bd: Nonconvex bound.
  """
  lx = jnp.reshape(x_lb, [-1, 1])
  ux = jnp.reshape(x_ub, [-1, 1])
  ly = jnp.reshape(y_lb, [-1, 1])
  uy = jnp.reshape(y_ub, [-1, 1])
  x = jnp.reshape(x, [-1, 1])
  y = jnp.reshape(y, [-1, 1])
  output_ub_a = jnp.dot(lx, y.T) + jnp.dot(x, uy.T) - jnp.dot(lx, uy.T)
  output_ub_b = jnp.dot(x, ly.T) + jnp.dot(ux, y.T) - jnp.dot(ux, ly.T)
  output_lb_a = jnp.dot(lx, y.T) + jnp.dot(x, ly.T) - jnp.dot(lx, ly.T)
  output_lb_b = jnp.dot(x, uy.T) + jnp.dot(ux, y.T) - jnp.dot(ux, uy.T)
  return (jnp.maximum(output_lb_a, output_lb_b),
          jnp.minimum(output_ub_a, output_ub_b))
