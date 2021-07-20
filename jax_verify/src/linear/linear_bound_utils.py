# coding=utf-8
# Copyright 2021 The jax_verify Authors.
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

"""Utils for Linear Bounds.
"""
import abc
import functools
from typing import Callable, Dict, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils

Bound = bound_propagation.Bound
Primitive = bound_propagation.Primitive
Tensor = bound_propagation.Tensor
Nest = bound_propagation.Nest
LinFun = Callable[..., Tensor]

EPSILON = 1e-5


class LinearExpression:
  """Describes a set of linear expressions."""

  def __init__(self, lin_coeffs, offset):
    """Creates a LinearExpression object.

    Args:
      lin_coeffs: nb_coeffs x (array shape)
      offset: (array shape)
    """
    self.lin_coeffs = lin_coeffs
    self.offset = offset

  @property
  def shape(self):
    return self.offset.shape

  def __add__(self, other):
    if isinstance(other, LinearExpression):
      return LinearExpression(self.lin_coeffs + other.lin_coeffs,
                              self.offset + other.offset)
    else:
      return LinearExpression(self.lin_coeffs, self.offset + other)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, LinearExpression):
      return LinearExpression(self.lin_coeffs - other.lin_coeffs,
                              self.offset - other.offset)
    else:
      return LinearExpression(self.lin_coeffs, self.offset - other)

  def __rsub__(self, other):
    if isinstance(other, LinearExpression):
      return LinearExpression(other.lin_coeffs - self.lin_coeffs,
                              other.offset - self.offset)
    else:
      return LinearExpression(-self.lin_coeffs, other - self.offset,)

  def __truediv__(self, other):
    return LinearExpression(self.lin_coeffs / other, self.offset / other)


class LinearBoundsRelaxer(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def linearize_primitive(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *inps: Union[Bound, Tensor],
      **params) -> Tuple[LinFun, LinFun]:
    """Obtain the parameters of the linearized relaxation of a given primitive.

    The relaxation holds when the inputs are within range of the bounds
    described by `inps`.

    Args:
      index: Index of the node in the bound propagation.
      primitive: Primitive to relax.
      *inps: Bounds on the inputs of the primitive or Tensors.
      **params: Parameters of the primitive.
    Returns:
      lb_linfun: Function evaluating the linear lower bound relaxing that
        primitive.
      ub_linfun: Function evaluating the linear upper bound relaxing that
        primitive.
    """


class FixedLinearBoundsRelaxer(LinearBoundsRelaxer):
  """Relaxer mapping each primitive to a predefined linear relaxation.

  This relaxation admits no additional parameters.
  """

  def __init__(
      self,
      primitive_mapper: Dict[Primitive, Callable[..., Tuple[LinFun, LinFun]]]):
    self._primitive_mapper = primitive_mapper

  def linearize_primitive(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *inps: Union[Bound, Tensor],
      **params) -> Tuple[LinFun, LinFun]:
    return self._primitive_mapper[primitive](index, *inps, **params)


class ParameterizedNodeRelaxation(metaclass=abc.ABCMeta):
  """Computes upper/lower linearisations using optimisable parameters."""

  @abc.abstractmethod
  def relax(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    """Returns linearised relaxation for given optimisable parameters."""

  @abc.abstractmethod
  def initial_params(self) -> Nest[Tensor]:
    """Returns initial values of optimisable parameters."""

  @abc.abstractmethod
  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    """Projects optimisable parameters to their valid domain."""


class ParameterizedLinearBoundsRelaxer(metaclass=abc.ABCMeta):
  """Relaxer mapping each primitive to an optimisable linear relaxation."""

  @abc.abstractmethod
  def linearize_primitive(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *inps: Union[Bound, Tensor],
      **params) -> ParameterizedNodeRelaxation:
    """Obtain a parameterised family of linear relaxations of a given primitive.

    The relaxations hold when the inputs are within range of the bounds
    described by `inps`.

    Args:
      index: Index of the node in the bound propagation.
      primitive: Primitive to relax.
      *inps: Bounds on the inputs of the primitive or Tensors.
      **params: Parameters of the primitive.
    Returns:
      lb_linfun: Function evaluating the linear lower bound relaxing that
        primitive.
      ub_linfun: Function evaluating the linear upper bound relaxing that
        primitive.
    """


class BindRelaxerParams(LinearBoundsRelaxer):
  """Relaxer formed from binding parameters into a parameterised relaxer."""

  def __init__(
      self,
      node_relaxations: Dict[
          graph_traversal.Index, ParameterizedNodeRelaxation],
      relax_params: Dict[graph_traversal.Index, Tensor],
  ):
    self._node_relaxations = node_relaxations
    self._relax_params = relax_params

  def linearize_primitive(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *inps: Union[Bound, Tensor],
      **params) -> Tuple[LinFun, LinFun]:
    return self._node_relaxations[index].relax(self._relax_params[index], *inps)


class _NoParamRelaxation(ParameterizedNodeRelaxation):
  """Adapts a simple relaxer function into a zero-parameter relaxer function."""

  def __init__(
      self,
      relaxer: Callable[..., Tuple[LinFun, LinFun]],
      *input_shapes: Sequence[int],
      **params):
    super().__init__()
    del input_shapes
    self._relaxer = relaxer
    self._params = params

  def relax(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    return self._relaxer(*inps, **self._params)

  def initial_params(self) -> Nest[Tensor]:
    return ()

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return relax_params


no_params = lambda relaxer: functools.partial(_NoParamRelaxation, relaxer)


def eltwise_linfun_from_coeff(slope: Tensor, offset: Tensor) -> LinFun:
  return lambda x: slope * x + offset


def _crown_relu_relaxer(inp: Bound) -> Tuple[LinFun, LinFun]:
  """Obtain the parameters of a linear ReLU relaxation as in CROWN.

  This relaxes the ReLU with the adaptive choice of lower bounds as described
  for CROWN-ada in https://arxiv.org/abs/1811.00866.

  Args:
    inp: Input to the ReLU.
  Returns:
    lb_linfun, ub_linfun: Linear functions bounding the ReLU
  """
  inp_lower, inp_upper = inp.lower, inp.upper
  relu_on = (inp_lower >= 0.)
  relu_amb = jnp.logical_and(inp_lower < 0., inp_upper >= 0.)
  ub_slope = relu_on.astype(jnp.float32)
  ub_slope += jnp.where(relu_amb,
                        inp_upper / jnp.maximum(inp_upper - inp_lower, 1e-12),
                        jnp.zeros_like(inp_lower))
  ub_offset = jnp.where(relu_amb, - ub_slope * inp_lower,
                        jnp.zeros_like(inp_lower))
  lb_slope = (ub_slope >= 0.5).astype(jnp.float32)
  lb_offset = jnp.zeros_like(inp_lower)

  return (eltwise_linfun_from_coeff(lb_slope, lb_offset),
          eltwise_linfun_from_coeff(ub_slope, ub_offset))


def _fastlin_relu_relaxer(inp: Bound) -> Tuple[LinFun, LinFun]:
  """Obtain the parameters of a linear ReLU relaxation as in FastLin.

  This relaxes the ReLU with the parallel bounds of slope (ub) / (ub - lb)

  Args:
    inp: Input to the ReLU.
  Returns:
    lb_linfun, ub_linfun: Linear functions bounding the ReLU
  """
  inp_lower, inp_upper = inp.lower, inp.upper
  relu_on = (inp_lower >= 0.)
  relu_amb = jnp.logical_and(inp_lower < 0., inp_upper >= 0.)
  slope = relu_on.astype(jnp.float32)
  slope += jnp.where(relu_amb,
                     inp_upper / jnp.maximum(inp_upper - inp_lower, 1e-12),
                     jnp.zeros_like(inp_lower))
  ub_offset = jnp.where(relu_amb, - slope * inp_lower,
                        jnp.zeros_like(inp_lower))
  lb_offset = jnp.zeros_like(inp_lower)

  return (eltwise_linfun_from_coeff(slope, lb_offset),
          eltwise_linfun_from_coeff(slope, ub_offset))


class _ParameterizedReluRelaxation(ParameterizedNodeRelaxation):
  """Linear relaxation of ReLU whose lower bound's slope is parameterised."""

  def __init__(self, input_shape: Sequence[int]):
    super().__init__()
    self._input_shape = input_shape

  def relax(
      self,
      relax_params: Nest[Tensor],
      inp: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    inp_lower, inp_upper = inp.lower, inp.upper
    relu_on = (inp_lower >= 0.)
    relu_amb = jnp.logical_and(inp_lower < 0., inp_upper >= 0.)
    ub_slope = relu_on.astype(jnp.float32)
    ub_slope += jnp.where(relu_amb,
                          inp_upper / jnp.maximum(inp_upper - inp_lower, 1e-12),
                          jnp.zeros_like(inp_lower))
    ub_offset = jnp.where(relu_amb, - ub_slope * inp_lower,
                          jnp.zeros_like(inp_lower))
    lb_slope = jnp.where(relu_amb, relax_params, ub_slope)
    lb_offset = jnp.zeros_like(inp_lower)

    return (eltwise_linfun_from_coeff(lb_slope, lb_offset),
            eltwise_linfun_from_coeff(ub_slope, ub_offset))

  def initial_params(self) -> Nest[Tensor]:
    return .5 * jnp.ones_like(shape=self._input_shape, dtype=jnp.float32)

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return jnp.clip(relax_params, 0., 1.)


def _rvt_exp_relaxer(inp: Bound) -> Tuple[LinFun, LinFun]:
  """Obtain the parameters of a linear exp relaxation as in RVT.

  Lower bound is obtained based on tangent, due to convexity.
  Choice of point where the tangent is computed is taken from
  https://arxiv.org/pdf/2002.06622.pdf

  Chord connecting two endpoints provides upper bound
  for exp(x) due to convexity.

  Args:
    inp: Input to the exp.
  Returns:
    lb_linfun, ub_linfun: Linear functions bounding the Exponential
  """
  lower, upper = inp.lower, inp.upper
  thresh = 12.0
  min_input = -30.

  forced_zeros = upper < min_input
  unsafe_to_relax = (upper - lower) < EPSILON
  unsafe = jnp.logical_or(forced_zeros, unsafe_to_relax)

  def stable_exp(x):
    """If x is greater than thresh, use first order Taylor's expansion."""
    return jnp.where(
        jnp.greater(x, thresh),
        jnp.exp(thresh)*(1 + x - thresh),
        jnp.exp(x))

  point = jnp.minimum((lower+upper)/2, lower + 0.99)
  lb_slope = stable_exp(point)
  lb_offset = lb_slope * (1 - point)

  # If the relaxation is over too narrow a domain, use the lower bound
  # constant as linear lower bounding function.
  lb_slope = jnp.where(unsafe, jnp.zeros_like(lower), lb_slope)
  lb_offset = jnp.where(unsafe, stable_exp(lower), lb_offset)

  ub_slope = (stable_exp(upper)-stable_exp(lower))/(upper - lower)
  ub_offset = stable_exp(lower) - ub_slope*lower
  # If the relaxation is over too narrow a domain, or if even the upper bound
  # is extremely small, we replace the upper bound by a bound with slope of 0.
  ub_slope = jnp.where(unsafe, jnp.zeros_like(lower), ub_slope)
  ub_offset = jnp.where(unsafe, stable_exp(upper), ub_offset)

  lb_fun = eltwise_linfun_from_coeff(lb_slope, lb_offset)
  ub_fun = eltwise_linfun_from_coeff(ub_slope, ub_offset)

  return lb_fun, ub_fun


def _rvt_posreciprocal_relaxer(inp: Bound) -> Tuple[LinFun, LinFun]:
  """Obtain the parameters of a linear relaxation of reciprocal.

  The (unchecked) assumption is that inputs are always positive, and 1/x is
  therefore convex.

  Tangent provides lower bound for 1/x due to convexity when x > 0.
  We use the midpoint to compute the tangent following the suggestion in
  https://arxiv.org/pdf/2002.06622.pdf

  Chord connecting the two endpoints provides upper bound for 1/x due to
  convexity.

  Args:
    inp: Linear bounds for input to reciprocate, assumed to always be positive.
  Returns:
    lb_linfun, ub_linfun: Linear functions bounding the 1/x function for x > 0.
  """
  lower, upper = inp.lower, inp.upper
  safe_lower = jnp.maximum(lower, 1e-6)
  safe_upper = jnp.maximum(upper, 1e-6)

  point = (safe_lower + safe_upper)/2
  lb_slope = -1.0/(point*point)
  lb_offset = 2./point

  ub_slope = -1.0 / (safe_upper * safe_lower)
  ub_offset = 1.0 / safe_upper + 1.0 / safe_lower

  lb_fun = eltwise_linfun_from_coeff(lb_slope, lb_offset)
  ub_fun = eltwise_linfun_from_coeff(ub_slope, ub_offset)

  return lb_fun, ub_fun


def _fixed_abs_relaxer(inp: Bound) -> Tuple[LinFun, LinFun]:
  """Obtains the parameters of a linear relaxation of the abs function.

  This is obtained by linearizing the function (as it is convex) for the lower
  bound, and taking the chord for the upper bound.

  Args:
    inp: Bound on the inputs of the absolute value.
  Returns:
    lb_linfun, ub_linfun.
  """
  lb_fun, ub_fun = activation_relaxation.abs_relaxation(inp.lower, inp.upper)

  # Linearizing in the mid point of the lower bound means that we are going to
  # pick either x or -x as the linear lower bound, depending on which one
  # represents the larger region, which is a reasonably good default.
  mid_point = 0.5 * (inp.lower + inp.upper)
  mid_lb, lb_jvp = jax.linearize(lb_fun, mid_point)
  lb_linfun = lambda x: mid_lb + lb_jvp(x - mid_point)

  # We know that the upper bound is already linear, so we can simply use it.
  return lb_linfun, ub_fun


def _fixed_leaky_relu_relaxer(
    inp: Bound, *, negative_slope: float) -> Tuple[LinFun, LinFun]:
  """Obtains the parameters of a linear relaxation of the LeakyReLU function.

  Args:
    inp: Bound on the inputs of the absolute value.
    negative_slope: Slope for the negative inputs.
  Returns:
    lb_linfun, ub_linfun.
  """
  lb_fun, ub_fun = activation_relaxation.leaky_relu_relaxation(
      inp.lower, inp.upper, negative_slope=negative_slope)

  mid_point = 0.5 * (inp.lower + inp.upper)
  mid_lb, lb_jvp = jax.linearize(lb_fun, mid_point)
  mid_ub, ub_jvp = jax.linearize(ub_fun, mid_point)
  lb_linfun = lambda x: mid_lb + lb_jvp(x - mid_point)
  ub_linfun = lambda x: mid_ub + ub_jvp(x - mid_point)

  # We know that the upper bound is already linear, so we can simply use it.
  return lb_linfun, ub_linfun


def _rvt_posbilinear_relaxer(x: Bound, y: Bound, **params
                             ) -> Tuple[LinFun, LinFun]:
  """Obtains the parameters of a linear relaxation of a bilinear function.

  Rather than using all 4 of the McCormick inequalities,
  https://arxiv.org/pdf/2002.06622.pdf use only two:
  For x in [x_l, x_u] and y in [y_l, y_u], the bound imposed are:
    xy >= y_l*x + x_l*y - x_l*y_l
    xy <= y_u*x + x_l*y - x_l*y_u

  Args:
    x: First input to the positive bilinear primitive.
    y: Second input to the positive bilinear primitive
    **params:
  Returns:
    lb_linfun, ub_linfun
  """
  assert isinstance(x, Bound)
  assert isinstance(y, Bound)
  x_lower = x.lower
  y_lower, y_upper = y.lower, y.upper

  def posbilinear_fun(inp1, inp2):
    return synthetic_primitives.posbilinear_p.bind(inp1, inp2, **params)

  def lb_fun(x, y):
    return (posbilinear_fun(x, y_lower)
            + posbilinear_fun(x_lower, y)
            - posbilinear_fun(x_lower, y_lower))

  def ub_fun(x, y):
    return (posbilinear_fun(x, y_upper)
            + posbilinear_fun(x_lower, y)
            - posbilinear_fun(x_lower, y_upper))

  # These functions are linear by construction.
  return lb_fun, ub_fun


_crown_mapper = {
    synthetic_primitives.leaky_relu_p: _fixed_leaky_relu_relaxer,
    synthetic_primitives.relu_p: _crown_relu_relaxer,
    lax.abs_p: _fixed_abs_relaxer,
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
    synthetic_primitives.posreciprocal_p: _rvt_posreciprocal_relaxer,
}
_crown_mapper = {prim: utils.simple_propagation(relax)
                 for prim, relax in _crown_mapper.items()}
crown_rvt_relaxer = FixedLinearBoundsRelaxer(_crown_mapper)

_fastlin_mapper = {
    synthetic_primitives.leaky_relu_p: _fixed_leaky_relu_relaxer,
    synthetic_primitives.relu_p: _fastlin_relu_relaxer,
    lax.abs_p: _fixed_abs_relaxer,
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posreciprocal_p: _rvt_posreciprocal_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
}
_fastlin_mapper = {prim: utils.simple_propagation(relax)
                   for prim, relax in _fastlin_mapper.items()}
fastlin_rvt_relaxer = FixedLinearBoundsRelaxer(_fastlin_mapper)

_parameterized_mapper = {
    synthetic_primitives.leaky_relu_p: no_params(_fixed_leaky_relu_relaxer),
    synthetic_primitives.relu_p: _ParameterizedReluRelaxation,
    lax.abs_p: no_params(_fixed_abs_relaxer),
    lax.exp_p: no_params(_rvt_exp_relaxer),
    synthetic_primitives.posreciprocal_p: no_params(_rvt_posreciprocal_relaxer),
    synthetic_primitives.posbilinear_p: no_params(_rvt_posbilinear_relaxer),
}
