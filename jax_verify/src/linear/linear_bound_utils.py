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

"""Utils for Linear Bounds.
"""
import abc
import functools
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

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
TensorFun = activation_relaxation.TensorFunction

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


class OpwiseLinearBoundsRelaxer(LinearBoundsRelaxer):
  """Relaxer mapping each primitive to a predefined linear relaxation.

  This relaxation admits no additional parameters.
  """

  def __init__(
      self,
      primitive_mapper: Dict[Primitive, Callable[..., Tuple[LinFun, LinFun]]],
      default_relaxer: Optional[LinearBoundsRelaxer] = None):
    self._primitive_mapper = primitive_mapper
    self._default_relaxer = default_relaxer

  def linearize_primitive(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *inps: Union[Bound, Tensor],
      **params) -> Tuple[LinFun, LinFun]:
    if primitive in self._primitive_mapper:
      return self._primitive_mapper[primitive](index, *inps, **params)
    elif self._default_relaxer:
      return self._default_relaxer.linearize_primitive(
          index, primitive, *inps, **params)
    else:
      raise ValueError(f'Unsupported primitive to relax: {primitive}.')


class ParameterizedNodeRelaxation(metaclass=abc.ABCMeta):
  """Computes upper/lower linearisations using optimisable parameters."""

  @abc.abstractproperty
  def arity(self) -> int:
    """Returns the number of input argument that this relaxation expects."""

  @abc.abstractmethod
  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    """Returns linearised relaxation for given optimisable parameters."""

  @abc.abstractmethod
  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    """Returns initial values of optimisable parameters."""

  @abc.abstractmethod
  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    """Projects optimisable parameters to their valid domain."""


class ParameterizedLinearBoundsRelaxer(metaclass=abc.ABCMeta):
  """Relaxer mapping each primitive to an optimisable linear relaxation."""

  @abc.abstractmethod
  def parameterized_linearizer(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *input_shapes: Sequence[int],
      **params) -> ParameterizedNodeRelaxation:
    """Obtain a parameterised family of linear relaxations of a given primitive.

    The relaxations hold when the inputs are within range of the bounds
    described by `inps`.

    Args:
      index: Index of the node in the bound propagation.
      primitive: Primitive to relax.
      *input_shapes: Shapes of the inputs of the primitive.
      **params: Parameters of the primitive.
    Returns:
      Object producing linearised lower/upper bounds given relaxation
      parameters.
    """


class OpwiseParameterizedLinearBoundsRelaxer(ParameterizedLinearBoundsRelaxer):
  """Relaxer mapping each primitive to a parameterized linear relaxation."""

  def __init__(
      self,
      primitive_mapper: Dict[
          Primitive, Callable[..., ParameterizedNodeRelaxation]],
      default_param_relaxer: Optional[ParameterizedLinearBoundsRelaxer] = None):
    self._primitive_mapper = primitive_mapper
    self._default_parameterized_relaxer = default_param_relaxer

  def parameterized_linearizer(
      self,
      index: graph_traversal.Index,
      primitive: Primitive,
      *input_shapes: Sequence[int],
      **params) -> ParameterizedNodeRelaxation:
    if primitive in self._primitive_mapper:
      return self._primitive_mapper[primitive](index, *input_shapes, **params)
    elif self._default_parameterized_relaxer:
      return self._default_parameterized_relaxer.parameterized_linearizer(
          index, primitive, *input_shapes, **params)
    else:
      raise ValueError(f'Unsupported primitive to relax: {primitive}.')


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
    return self._node_relaxations[index].linearize(
        self._relax_params[index], *inps)


class ParameterizedLinFun(metaclass=abc.ABCMeta):
  """Linearisation of lower OR upper bound using optimisable parameters."""

  @abc.abstractproperty
  def arity(self) -> int:
    """Returns the number of input argument that the linear function expects."""

  @abc.abstractmethod
  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> LinFun:
    """Returns linearised half-bound for given optimisable parameters."""

  @abc.abstractmethod
  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    """Returns initial values of optimisable parameters."""

  @abc.abstractmethod
  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    """Projects optimisable parameters to their valid domain."""


class LowerUpperRelaxation(ParameterizedNodeRelaxation):
  """Adapts two parameterised half-bounds into a parameterised relaxation."""

  def __init__(
      self, lower: ParameterizedLinFun, upper: ParameterizedLinFun):
    super().__init__()
    self._lower = lower
    self._upper = upper

  @property
  def arity(self) -> int:
    return self._lower.arity

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    lower_relax_params, upper_relax_params = relax_params
    return (
        self._lower.linearize(lower_relax_params, *inps),
        self._upper.linearize(upper_relax_params, *inps))

  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    return (
        self._lower.initial_params(*inps),
        self._upper.initial_params(*inps))

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    lower_relax_params, upper_relax_params = relax_params
    return (
        self._lower.project_params(lower_relax_params),
        self._upper.project_params(upper_relax_params))


class _NoParamRelaxation(ParameterizedNodeRelaxation):
  """Adapts a simple relaxer function into a zero-parameter relaxer function."""

  def __init__(
      self,
      relaxer: Callable[..., Tuple[LinFun, LinFun]],
      *input_shapes: Sequence[int],
      **params):
    self._relaxer = relaxer
    self._input_shapes = input_shapes
    self._params = params

  @property
  def arity(self):
    return len(self._input_shapes)

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    return self._relaxer(*inps, **self._params)

  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    return ()

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return relax_params


no_params = lambda relaxer: functools.partial(_NoParamRelaxation, relaxer)


def linearized(fn: Callable[..., Tensor], *primals: Tensor) -> LinFun:
  """Returns linear function that is tangent to `fn` at given primal point."""
  val, deriv = jax.linearize(fn, *primals)
  return lambda *xs: val + deriv(*[x - p for x, p in zip(xs, primals)])


class SupportingHyperplane(ParameterizedLinFun):
  """Linearisation of primitive, parameterised by primal point."""

  def __init__(
      self,
      fn: TensorFun,
      *input_shapes: Sequence[int]):
    super().__init__()
    self._fn = fn
    self._input_shapes = input_shapes

  @property
  def arity(self) -> int:
    return len(self._input_shapes)

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> LinFun:
    primals = [
        alpha * inp.upper + (1.-alpha) * inp.lower
        if isinstance(inp, Bound) else inp
        for inp, alpha in zip(inps, relax_params)]
    return linearized(self._fn, *primals)

  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    # If an input is [known to be] a fixed tensor, don't allocate a
    # relaxation parameter for it.
    return [
        .5 * jnp.ones(shape=inp_shape)
        if inp is None or isinstance(inp, Bound) else None
        for inp, inp_shape in zip(inps, self._input_shapes)]

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return jax.tree_map(lambda x: jnp.clip(x, 0., 1.), relax_params)


def eltwise_linfun_from_coeff(slope: Tensor, offset: Tensor) -> LinFun:
  return lambda x: slope * x + offset


class ElementwiseChord(ParameterizedLinFun):
  """Chord between input bounds of an element-wise primitive."""

  arity = 1

  def __init__(
      self,
      fn: TensorFun,
      input_shape: Sequence[int]):
    super().__init__()
    self._fn = fn
    self._input_shape = input_shape

  def linearize(
      self,
      relax_params: Nest[Tensor],
      inp: Union[Bound, Tensor],
  ) -> LinFun:
    inp_lower, inp_upper = inp.lower, inp.upper
    outp_lower, outp_upper = self._fn(inp_lower), self._fn(inp_upper)

    has_interval = inp_upper != inp_lower
    denom = jnp.where(
        has_interval, inp_upper - inp_lower,
        jnp.ones_like(inp_lower))
    slope = jnp.where(
        has_interval, (outp_upper - outp_lower) / denom,
        jnp.zeros_like(inp_lower))
    offset = jnp.where(
        has_interval, (outp_lower * inp_upper - outp_upper * inp_lower) / denom,
        outp_lower)
    return eltwise_linfun_from_coeff(slope, offset)

  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    return ()

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return relax_params


def elementwise_convex_fn_relaxer(
    primitive: Primitive,
    input_shape: Sequence[int],
    **params) -> ParameterizedNodeRelaxation:
  fn = functools.partial(primitive.bind, **params)
  lower = SupportingHyperplane(fn, input_shape)
  upper = ElementwiseChord(fn, input_shape)
  return LowerUpperRelaxation(lower, upper)


def elementwise_concave_fn_relaxer(
    primitive: Primitive,
    input_shape: Sequence[int],
    **params) -> ParameterizedNodeRelaxation:
  fn = functools.partial(primitive.bind, **params)
  lower = ElementwiseChord(fn, input_shape)
  upper = SupportingHyperplane(fn, input_shape)
  return LowerUpperRelaxation(lower, upper)


class _SmoothConvexRelaxation(LowerUpperRelaxation):
  """Linear relaxation from supporting hyperplanes of a convex relaxation."""

  def __init__(
      self,
      convex_relaxer: Callable[..., Tuple[TensorFun, TensorFun]],
      *input_shapes: Sequence[int],
      **params):
    # Create a skeleton `ParameterizedLinFun` for initialisation and projection
    # relaxation parameters. This doesn't need the lower/upper bound functions.
    skeleton = SupportingHyperplane((lambda *_: None), *input_shapes)
    super().__init__(skeleton, skeleton)

    self._convex_relaxer = convex_relaxer
    self._input_shapes = input_shapes
    self._params = params

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    # First obtain convex lower and concave upper bounds.
    # Do so at this late stage because they depend on the current input bounds.
    lb_fun, ub_fun = self._convex_relaxer(*inps, **self._params)

    lower = SupportingHyperplane(lb_fun, *self._input_shapes)
    upper = SupportingHyperplane(ub_fun, *self._input_shapes)

    lower_relax_params, upper_relax_params = relax_params
    return (
        lower.linearize(lower_relax_params, *inps),
        upper.linearize(upper_relax_params, *inps))


def linear_from_smooth_convex(
    convex_relaxer: Callable[..., Tuple[TensorFun, TensorFun]],
) -> Callable[..., ParameterizedNodeRelaxation]:
  return functools.partial(_SmoothConvexRelaxation, convex_relaxer)


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


class ReluSubgradient(ParameterizedLinFun):
  """Lower bound of a ReLU-like function.

  This applies to any two-piece linear function passing through the origin.
  """

  arity = 1

  def __init__(
      self,
      input_shape: Sequence[int],
      neg_slope: float = 0.,
      pos_slope: float = 1.):
    super().__init__()
    self._input_shape = input_shape
    self._neg_slope = neg_slope
    self._pos_slope = pos_slope

  def linearize(
      self,
      relax_params: Nest[Tensor],
      inp: Union[Bound, Tensor],
  ) -> LinFun:
    inp_lower, inp_upper = inp.lower, inp.upper

    lower_slope = jnp.where(inp_lower < 0., self._neg_slope, self._pos_slope)
    upper_slope = jnp.where(inp_upper > 0., self._pos_slope, self._neg_slope)
    slope = relax_params * upper_slope + (1.-relax_params) * lower_slope
    offset = jnp.zeros_like(inp_lower)

    return eltwise_linfun_from_coeff(slope, offset)

  def initial_params(
      self,
      inp: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    if inp is None:
      return .5 * jnp.ones(shape=self._input_shape, dtype=jnp.float32)
    else:
      # Initialise close to pos_slope if input interval has more positive mass
      # than negative, and vice versa.
      # This is a softened version of CROWN.
      return jax.nn.sigmoid(inp.lower + inp.upper)

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return jnp.clip(relax_params, 0., 1.)


def _parameterized_relu_relaxer(
    primitive: Primitive,
    input_shape: Sequence[int],
    neg_slope: float = 0.,
    pos_slope: float = 1.,
    **params) -> ParameterizedNodeRelaxation:
  """Parameterised relaxation for ReLU-like functions.

  This applies to any two-piece linear function passing through the origin.

  Args:
    primitive: ReLU-like primitive to relax.
    input_shape: Shapes of the input of the primitive.
    neg_slope: Gradient of the primitive for negative inputs.
    pos_slope: Gradient of the primitive for positive inputs.
    **params: Parameters of the primitive.
  Returns:
    Object producing linearised lower/upper bounds given relaxation
    parameters.
  """
  fn = functools.partial(primitive.bind, **params)
  subgradient = ReluSubgradient(input_shape, neg_slope, pos_slope)
  chord = ElementwiseChord(fn, input_shape)
  if neg_slope <= pos_slope:
    return LowerUpperRelaxation(lower=subgradient, upper=chord)
  else:
    # E.g. leaky ReLU with negative slope >1. This is concave.
    return LowerUpperRelaxation(lower=chord, upper=subgradient)


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
  lb_fun, ub_fun = activation_relaxation.convex_fn_relaxation(lax.abs_p, inp)

  # Linearizing in the mid point of the lower bound means that we are going to
  # pick either x or -x as the linear lower bound, depending on which one
  # represents the larger region, which is a reasonably good default.
  mid_point = 0.5 * (inp.lower + inp.upper)
  lb_linfun = linearized(lb_fun, mid_point)

  # We know that the upper bound is already linear, so we can simply use it.
  return lb_linfun, ub_fun


_parameterized_abs_relaxer = functools.partial(
    _parameterized_relu_relaxer, neg_slope=-1., pos_slope=1.)


def _fixed_leaky_relu_relaxer(
    inp: Bound, *, negative_slope: float) -> Tuple[LinFun, LinFun]:
  """Obtains the parameters of a linear relaxation of the LeakyReLU function.

  Args:
    inp: Bound on the inputs of the leaky relu.
    negative_slope: Slope for the negative inputs.
  Returns:
    lb_linfun, ub_linfun.
  """
  lb_fun, ub_fun = activation_relaxation.leaky_relu_relaxation(
      inp, negative_slope=negative_slope)

  mid_point = 0.5 * (inp.lower + inp.upper)
  return linearized(lb_fun, mid_point), linearized(ub_fun, mid_point)


def _parameterized_leaky_relu_relaxer(
    primitive: Primitive,
    input_shape: Sequence[int],
    *,
    negative_slope: float,
) -> ParameterizedNodeRelaxation:
  return _parameterized_relu_relaxer(
      primitive, input_shape, neg_slope=negative_slope,
      negative_slope=negative_slope)  # pass `negative_slope` again as **params


def _fixed_sigmoid_relaxer(inp: Bound) -> Tuple[LinFun, LinFun]:
  """Obtains the parameters of a linear relaxation of the sigmoid function.

  Args:
    inp: Bound on the inputs of the sigmoid.
  Returns:
    lb_linfun, ub_linfun.
  """
  lb_fun, ub_fun = activation_relaxation.sigmoid_relaxation(inp)

  mid_point = 0.5 * (inp.lower + inp.upper)
  return linearized(lb_fun, mid_point), linearized(ub_fun, mid_point)


def _rvt_posbilinear_relaxer(
    x: Bound,
    y: Bound,
    **params) -> Tuple[LinFun, LinFun]:
  """Obtains the parameters of a linear relaxation of a bilinear function.

  Rather than using all 4 of the McCormick inequalities,
  https://arxiv.org/pdf/2002.06622.pdf use only two:
  For x in [x_l, x_u] and y in [y_l, y_u], the bound imposed are:
    x·y >= x·y_l + x_l·y - x_l·y_l
    x·y <= x·y_u + x_l·y - x_l·y_u

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

  fn = functools.partial(synthetic_primitives.posbilinear_p.bind, **params)

  def lb_fun(x: Tensor, y: Tensor):
    return fn(x, y_lower) + fn(x_lower, y) - fn(x_lower, y_lower)

  def ub_fun(x: Tensor, y: Tensor):
    return fn(x, y_upper) + fn(x_lower, y) - fn(x_lower, y_upper)

  # These functions are linear by construction.
  return lb_fun, ub_fun


class ParameterizedPosbilinearRelaxer(ParameterizedNodeRelaxation):
  """Parameterized linear relaxation of a bilinear function.

  This uses pairwise interpolations of the McCormick inequalities.
  For x in [x_l, x_u] and y in [y_l, y_u], the bound imposed are:
    x·y >= x·y_l + x_l·y - x_l·y_l
    x·y >= x·y_u + x_h·y - x_h·y_u
    x·y <= x·y_u + x_l·y - x_l·y_u
    x·y <= x·y_l + x_u·y - x_l·y_u
  """

  arity = 2

  def __init__(
      self,
      x_shape: Sequence[int],
      y_shape: Sequence[int],
      **params):
    super().__init__()
    self._x_shape = x_shape
    self._y_shape = y_shape
    self._params = params

  def linearize(
      self,
      relax_params: Nest[Tensor],
      x: Union[Bound, Tensor],
      y: Union[Bound, Tensor]) -> Tuple[LinFun, LinFun]:
    assert isinstance(x, Bound)
    assert isinstance(y, Bound)
    x_lower, x_upper = x.lower, x.upper
    y_lower, y_upper = y.lower, y.upper

    fn = functools.partial(
        synthetic_primitives.posbilinear_p.bind, **self._params)
    lower_relax_params, upper_relax_params = relax_params

    def lb_fun(x: Tensor, y: Tensor):
      lb0 = fn(x, y_lower) + fn(x_lower, y) - fn(x_lower, y_lower)
      lb1 = fn(x, y_upper) + fn(x_upper, y) - fn(x_upper, y_upper)
      return lower_relax_params * lb0 + (1.-lower_relax_params) * lb1

    def ub_fun(x: Tensor, y: Tensor):
      ub0 = fn(x, y_upper) + fn(x_lower, y) - fn(x_lower, y_upper)
      ub1 = fn(x, y_lower) + fn(x_upper, y) - fn(x_upper, y_lower)
      return upper_relax_params * ub0 + (1.-upper_relax_params) * ub1

    # These functions are linear by construction.
    return lb_fun, ub_fun

  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    fn = functools.partial(
        synthetic_primitives.posbilinear_p.bind, **self._params)
    # Include a lower and an upper interpolation parameter for each output.
    output_shape = jax.eval_shape(
        fn, jnp.zeros(self._x_shape), jnp.zeros(self._y_shape)).shape
    return (
        .5 * jnp.ones(shape=output_shape, dtype=jnp.float32),
        .5 * jnp.ones(shape=output_shape, dtype=jnp.float32))

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    lower_relax_params, upper_relax_params = relax_params
    return (
        jnp.clip(lower_relax_params, 0., 1.),
        jnp.clip(upper_relax_params, 0., 1.))


_parameterized_posbilinear_relaxer = functools.partial(
    ParameterizedPosbilinearRelaxer)


_crown_mapper = {
    synthetic_primitives.leaky_relu_p: _fixed_leaky_relu_relaxer,
    synthetic_primitives.relu_p: _crown_relu_relaxer,
    synthetic_primitives.sigmoid_p: _fixed_sigmoid_relaxer,
    lax.abs_p: _fixed_abs_relaxer,
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
    synthetic_primitives.posreciprocal_p: _rvt_posreciprocal_relaxer,
}
_crown_mapper = {prim: utils.simple_propagation(relax)
                 for prim, relax in _crown_mapper.items()}
crown_rvt_relaxer = OpwiseLinearBoundsRelaxer(_crown_mapper)

_fastlin_mapper = {
    synthetic_primitives.leaky_relu_p: _fixed_leaky_relu_relaxer,
    synthetic_primitives.relu_p: _fastlin_relu_relaxer,
    synthetic_primitives.sigmoid_p: _fixed_sigmoid_relaxer,
    lax.abs_p: _fixed_abs_relaxer,
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posreciprocal_p: _rvt_posreciprocal_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
}
_fastlin_mapper = {prim: utils.simple_propagation(relax)
                   for prim, relax in _fastlin_mapper.items()}
fastlin_rvt_relaxer = OpwiseLinearBoundsRelaxer(_fastlin_mapper)

_parameterized_mapper = {
    synthetic_primitives.leaky_relu_p: functools.partial(
        _parameterized_leaky_relu_relaxer, synthetic_primitives.leaky_relu_p),
    synthetic_primitives.relu_p: functools.partial(
        _parameterized_relu_relaxer, synthetic_primitives.relu_p),
    synthetic_primitives.sigmoid_p: linear_from_smooth_convex(
        activation_relaxation.sigmoid_relaxation),
    lax.abs_p: functools.partial(_parameterized_abs_relaxer, lax.abs_p),
    synthetic_primitives.softplus_p: functools.partial(
        elementwise_convex_fn_relaxer, synthetic_primitives.softplus_p),
    lax.exp_p: functools.partial(elementwise_convex_fn_relaxer, lax.exp_p),
    synthetic_primitives.posreciprocal_p: functools.partial(
        elementwise_concave_fn_relaxer, synthetic_primitives.posreciprocal_p),
    synthetic_primitives.posbilinear_p: _parameterized_posbilinear_relaxer,
}
_parameterized_mapper = {prim: utils.simple_propagation(relax)
                         for prim, relax in _parameterized_mapper.items()}
parameterized_relaxer = OpwiseParameterizedLinearBoundsRelaxer(
    _parameterized_mapper)
