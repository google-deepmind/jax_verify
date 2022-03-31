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

"""Mccormick relaxations for bilinear terms.

Create mc-cormick relaxations of bilinear terms for boundprop and verification.
"""
from typing import Callable, Tuple

import jax.numpy as jnp

Tensor = jnp.ndarray
BilinearFun = Callable[[Tensor, Tensor], Tensor]


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


def posbilinear_mccormick_relaxations(
    fn: BilinearFun,
    x_lb: Tensor, x_ub: Tensor, y_lb: Tensor, y_ub: Tensor,
) -> Tuple[BilinearFun, BilinearFun, BilinearFun, BilinearFun]:
  """Constructs all four McCormick relaxation of a positive bilinear primitive.

  For x in [x_l, x_u] and y in [y_l, y_u], the bound imposed are:
    x·y >= x·y_l + x_l·y - x_l·y_l
    x·y >= x·y_u + x_h·y - x_h·y_u
    x·y <= x·y_u + x_l·y - x_l·y_u
    x·y <= x·y_l + x_u·y - x_l·y_u

  Args:
    fn: Positive definite bilinear function.
    x_lb: Lower bounds on x
    x_ub: Upper bounds on x
    y_lb: Lower bounds on y
    y_ub: Upper bounds on y
  Returns:
    lb_fun0, lb_fun1, ub_fun0, ub_fun1
  """
  def lb_fun0(x, y):
    return fn(x, y_lb) + fn(x_lb, y) - fn(x_lb, y_lb)

  def lb_fun1(x, y):
    return fn(x, y_ub) + fn(x_ub, y) - fn(x_ub, y_ub)

  def ub_fun0(x, y):
    return fn(x, y_ub) + fn(x_lb, y) - fn(x_lb, y_ub)

  def ub_fun1(x, y):
    return fn(x, y_lb) + fn(x_ub, y) - fn(x_ub, y_lb)

  return lb_fun0, lb_fun1, ub_fun0, ub_fun1


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
    y_lb: Lower bounds on y
    y_ub: Upper bounds on y

  Returns:
    bd: Nonconvex bound.
  """
  def outer(x, y):
    x = jnp.reshape(x, [-1, 1])
    y = jnp.reshape(y, [-1, 1])
    return jnp.dot(x, y.T)

  output_lb_a, output_lb_b, output_ub_a, output_ub_b = [
      relax_fn(x, y) for relax_fn in posbilinear_mccormick_relaxations(
          outer, x_lb, x_ub, y_lb, y_ub)]
  return (jnp.maximum(output_lb_a, output_lb_b),
          jnp.minimum(output_ub_a, output_ub_b))
