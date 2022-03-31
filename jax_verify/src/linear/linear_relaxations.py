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
from jax_verify.src import mccormick
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils

Bound = bound_propagation.Bound
Primitive = bound_propagation.Primitive
Tensor = bound_propagation.Tensor
Nest = bound_propagation.Nest
LinFun = Callable[..., Tensor]
TensorFun = activation_relaxation.TensorFunction


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
      params = synthetic_primitives.filter_jaxverify_kwargs(params)
      return self._primitive_mapper[primitive](*inps, **params)
    elif self._default_relaxer:
      return self._default_relaxer.linearize_primitive(
          index, primitive, *inps, **params)
    else:
      raise ValueError(f'Unsupported primitive to relax: {primitive}.')


class ParameterizedNodeRelaxation(metaclass=abc.ABCMeta):
  """Computes upper/lower linearisations using optimisable parameters."""

  @abc.abstractproperty
  def arity(self) -> int:
    """Returns the number of input arguments expected by this relaxation."""

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
      params = synthetic_primitives.filter_jaxverify_kwargs(params)
      return self._primitive_mapper[primitive](*input_shapes, **params)
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
    """Returns the number of input arguments expected by the linear function."""

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


def midpoint_relaxer(
    primitive: Primitive,
    *inputs: Union[Tensor, Bound],
    convex: bool = False,
    **params) -> Tuple[LinFun, LinFun]:
  """Obtains relaxation by linearising convex relaxation about the midpoint.

  In the case of a ReLU, this relaxes the ReLU with the adaptive choice of
  lower bounds as described for CROWN-ada in https://arxiv.org/abs/1811.00866.

  Args:
    primitive: Primitive to relax.
    *inputs: All inputs to the primitive, bounds or Tensor.
    convex: Whether the primitive is known to be convex.
    **params: Parameters of the primitive.
  Returns:
    lb_linfun, ub_linfun: Linear lower and upper bound functions.
  """
  activation = activation_relaxation.relaxation_fns[primitive]
  lb_fun, ub_fun = activation.relaxation_fn(*inputs, **params)

  mid_points = [
      0.5 * (x.lower + x.upper) if isinstance(x, Bound) else x for x in inputs]

  lb_lin_fun = linearized(lb_fun, *mid_points)
  ub_lin_fun = ub_fun if convex else linearized(ub_fun, *mid_points)
  return lb_lin_fun, ub_lin_fun


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
  slope += jnp.where(
      relu_amb,
      inp_upper / utils.safe_pos(inp_upper - inp_lower),
      jnp.zeros_like(inp_lower))
  ub_offset = jnp.where(
      relu_amb,
      -slope * inp_lower,
      jnp.zeros_like(inp_lower))
  lb_offset = jnp.zeros_like(inp_lower)

  return (eltwise_linfun_from_coeff(slope, lb_offset),
          eltwise_linfun_from_coeff(slope, ub_offset))


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
  unsafe_to_relax = (upper - lower)**2 < utils.EPSILON
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
  lb_fun, _, ub_fun, _ = (
      mccormick.posbilinear_mccormick_relaxations(
          functools.partial(synthetic_primitives.posbilinear_p.bind, **params),
          x.lower, x.upper, y.lower, y.upper))

  # These functions are linear by construction.
  return lb_fun, ub_fun


def _maybe_interp(
    funs: Sequence[TensorFun],
    interp_params: Nest[Tensor],
    *args: Tensor
) -> Tensor:
  """Interpolates between `funs` according to the given interpolation params."""
  if len(funs) == 1:
    # Interpolation is trivial.
    fun, = funs
    return fun(*args)
  else:
    # Assume two pieces, so we can interpolate with a single parameter.
    assert len(funs) == 2
    fun0, fun1 = funs
    return (1.-interp_params) * fun0(*args) + interp_params * fun1(*args)


class ParameterizedPiecewiseLinearSubgradient(ParameterizedNodeRelaxation):
  """Relaxation of a piecewise-linear function.

  This implementation currently assumes exactly two pieces.
  """

  def __init__(
      self,
      primitive: synthetic_primitives.PrimitiveLike,
      piecewise_linear_relaxation_fn: Callable[..., Tuple[
          Sequence[TensorFun], Sequence[TensorFun]]],
      *input_shapes: Sequence[int],
      soft_init: bool = True,
      **params):
    super().__init__()
    self._primitive = primitive
    self._piecewise_linear_relaxation_fn = piecewise_linear_relaxation_fn
    self._input_shapes = input_shapes
    self._soft_init = soft_init
    self._params = params

  @property
  def arity(self):
    return len(self._input_shapes)

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inputs: Union[Bound, Tensor],
  ) -> Tuple[LinFun, LinFun]:
    lb_funs, ub_funs = self._piecewise_linear_relaxation_fn(*inputs,
                                                            **self._params)
    lb_relax_params, ub_relax_params = relax_params
    return (
        functools.partial(_maybe_interp, lb_funs, lb_relax_params),
        functools.partial(_maybe_interp, ub_funs, ub_relax_params))

  def initial_params(
      self,
      *inps: Optional[Union[Bound, Tensor]],
  ) -> Nest[Tensor]:
    if len(inps) == 1 and self._soft_init:
      soft_init_inp, = inps
    else:
      soft_init_inp = None

    if soft_init_inp is not None:
      # When interpolating between linear bound pieces (xb_fun0, xb_fun1),
      # initialise close to xb_fun1 if input interval has more positive mass
      # than negative, and close to xb_fun0 otherwise.
      # This assumes that xb_fun0 is the effective piece for lower inputs
      # and xb_fun1 is the effective piece for upper inputs.
      # This is the case for ReLU for example, where (lb_fun0, lb_fun1)
      # is (x:->0, x:->x). This amounts to a softened version of CROWN.
      params = jnp.clip(
          soft_init_inp.upper
          / utils.safe_pos(soft_init_inp.upper - soft_init_inp.lower),
          0., 1.)
    else:
      # Include an interpolation parameter for each output.
      output_shape = jax.eval_shape(
          self._primitive.bind,
          *[jnp.zeros(shape) for shape in self._input_shapes],
          **self._params).shape
      params = .5 * jnp.ones(shape=output_shape, dtype=jnp.float32)

    # Determine the number of linear pieces of the lower and upper bounds.
    lb_funs, ub_funs = self._piecewise_linear_relaxation_fn(
        *[bound_propagation.IntervalBound(jnp.zeros(shape), jnp.zeros(shape))
          for shape in self._input_shapes],
        **self._params)

    return (
        () if len(lb_funs) <= 1 else params,
        () if len(ub_funs) <= 1 else params)

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return jax.tree_map(lambda x: jnp.clip(x, 0., 1.), relax_params)


_crown_mapper = {
    primitive: functools.partial(
        midpoint_relaxer, primitive, convex=activation.convex)
    for primitive, activation in activation_relaxation.relaxation_fns.items()}
_crown_mapper.update({
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
})
crown_rvt_relaxer = OpwiseLinearBoundsRelaxer(_crown_mapper)

_fastlin_mapper = {
    primitive: functools.partial(
        midpoint_relaxer, primitive, convex=activation.convex)
    for primitive, activation in activation_relaxation.relaxation_fns.items()}
_fastlin_mapper.update({
    synthetic_primitives.relu_p: _fastlin_relu_relaxer,
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
})
fastlin_rvt_relaxer = OpwiseLinearBoundsRelaxer(_fastlin_mapper)


def _make_parameterized_relaxer(
    primitive: synthetic_primitives.PrimitiveLike,
    activation: activation_relaxation.ActivationRelaxation,
) -> Callable[..., ParameterizedNodeRelaxation]:
  """Makes a linear relaxation based on the given convex relatation."""
  if activation.piecewise_linear_relaxation_fn:
    return functools.partial(
        ParameterizedPiecewiseLinearSubgradient,
        primitive,
        activation.piecewise_linear_relaxation_fn,
        soft_init=activation.pos_neg_linear)
  elif activation.convex:
    return functools.partial(elementwise_convex_fn_relaxer, primitive)
  else:
    return linear_from_smooth_convex(activation.relaxation_fn)


_parameterized_mapper = {
    primitive: _make_parameterized_relaxer(primitive, activation)
    for primitive, activation in activation_relaxation.relaxation_fns.items()}
parameterized_relaxer = OpwiseParameterizedLinearBoundsRelaxer(
    _parameterized_mapper)
