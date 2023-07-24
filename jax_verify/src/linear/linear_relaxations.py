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

"""Utils for Linear Bounds.
"""
import abc
import functools
from typing import Callable, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import mccormick
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils
from jax_verify.src.types import ArgsKwargsCallable, Index, Nest, Primitive, Tensor, TensorFun  # pylint: disable=g-multiple-import
import numpy as np

LinFun = TensorFun
# Representation of a linear function used in a relaxation.
# Can be a linear jax function (LinFun), or can be a LinearExpression.
LinearRelax = TypeVar('LinearRelax')


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

  @property
  def dtype(self):
    return self.offset.dtype

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

  def __neg__(self):
    return LinearExpression(-self.lin_coeffs, -self.offset)

  def __truediv__(self, other):
    return LinearExpression(self.lin_coeffs / other, self.offset / other)

  def transpose(self, in_shape: Tuple[int]) -> 'LinearExpression':
    """Transposes the LinearExpression.

    Convert the linear expression that are in the format of:
    [flattenned_in_shape, *out_shape]
    to the alternative format, which is
    [flattenned_out_shape, *unflattened_in_shape].

    Args:
      in_shape: Shape of the inputs to the linear functions. The product of the
        dimension needs to match the leading dimension of the lincoeffs.

    Returns:
      Expression in the transposed format.
    """
    out_shape = self.offset.shape

    flattened_out_shape = np.prod(out_shape)
    coeffs_all_flattened = jnp.reshape(self.lin_coeffs,
                                       (-1, flattened_out_shape))
    transposed_flattened_coeffs = jnp.transpose(coeffs_all_flattened, (1, 0))
    new_lin_coeffs = jnp.reshape(transposed_flattened_coeffs,
                                 (flattened_out_shape, *in_shape))
    new_offset = jnp.reshape(self.offset, (flattened_out_shape,))
    return LinearExpression(new_lin_coeffs, new_offset)


ConcretizationFn = Callable[[LinearExpression, graph_traversal.GraphInput],
                            Tensor]


def concretize_linear_expression(
    linexp: LinearExpression,
    input_bound: graph_traversal.InputBound) -> Tensor:
  """Compute the lower bound value of a linear expression.

  Args:
    linexp: Coefficients of linear functions. The leading batch dimension
      corresponds to different output neurons that need to be concretized. Shape
      is [nb_linfun, *input_bound.shape]
    input_bound: Bound on the activations of that layer. Its shape should match
      the coefficients of the linear functions to concretize. Shape is
      [*input_bound.shape]

  Returns:
    bound: A concretized bound on the value of the functions represented by
      linexp. Shape is [nb_linfun]
  """
  return concretize_linear_function_interval_bounds(linexp, input_bound)


def concretize_linear_function_interval_bounds(
    linexp: LinearExpression,
    input_bound: graph_traversal.InputBound) -> Tensor:
  """Compute the lower bound of a linear function under interval constraints."""
  act_lower = jnp.expand_dims(input_bound.lower, 0)
  act_upper = jnp.expand_dims(input_bound.upper, 0)

  dims_to_reduce = tuple(range(1, act_lower.ndim))

  return linexp.offset + jnp.sum(
      jnp.minimum(linexp.lin_coeffs, 0.) * act_upper +
      jnp.maximum(linexp.lin_coeffs, 0.) * act_lower, dims_to_reduce)


class LinearBoundsRelaxer(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def linearize_primitive(
      self,
      index: Index,
      primitive: Primitive,
      *inps: bound_propagation.LayerInput,
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
      primitive_mapper: Mapping[
          Primitive,
          ArgsKwargsCallable[
              bound_propagation.LayerInput, Tuple[LinFun, LinFun]]],
      default_relaxer: Optional[LinearBoundsRelaxer] = None):
    self._primitive_mapper = primitive_mapper
    self._default_relaxer = default_relaxer

  def linearize_primitive(
      self,
      index: Index,
      primitive: Primitive,
      *inps: bound_propagation.LayerInput,
      **params) -> Tuple[LinFun, LinFun]:
    if primitive in self._primitive_mapper:
      params = synthetic_primitives.filter_jaxverify_kwargs(params)
      return self._primitive_mapper[primitive](*inps, **params)
    elif self._default_relaxer:
      return self._default_relaxer.linearize_primitive(
          index, primitive, *inps, **params)
    else:
      raise ValueError(f'Unsupported primitive to relax: {primitive}.')


class ParameterizedNodeRelaxation(Generic[LinearRelax], metaclass=abc.ABCMeta):
  """Computes upper/lower linearisations using optimisable parameters."""

  @property
  @abc.abstractmethod
  def arity(self) -> int:
    """Returns the number of input arguments expected by this relaxation."""

  @abc.abstractmethod
  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: bound_propagation.LayerInput,
  ) -> Tuple[LinearRelax, LinearRelax]:
    """Returns linearised relaxation for given optimisable parameters."""

  @abc.abstractmethod
  def initial_params(
      self,
      *inps: Optional[bound_propagation.LayerInput],
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
      index: Index,
      primitive: Primitive,
      *input_info: Tuple[Sequence[int], bool],
      **params) -> ParameterizedNodeRelaxation[LinFun]:
    """Obtain a parameterised family of linear relaxations of a given primitive.

    Args:
      index: Index of the node in the bound propagation.
      primitive: Primitive to relax.
      *input_info: Tuples of (shape, is_bound) of the inputs of the primitive.
      **params: Parameters of the primitive.
    Returns:
      Object producing linearised lower/upper bounds given relaxation
      parameters.
    """


class OpwiseParameterizedLinearBoundsRelaxer(ParameterizedLinearBoundsRelaxer):
  """Relaxer mapping each primitive to a parameterized linear relaxation."""

  def __init__(
      self,
      primitive_mapper: Mapping[
          Primitive,
          ArgsKwargsCallable[Sequence[int],
                             ParameterizedNodeRelaxation[LinFun]]],
      default_param_relaxer: Optional[ParameterizedLinearBoundsRelaxer] = None):
    self._primitive_mapper = primitive_mapper
    self._default_parameterized_relaxer = default_param_relaxer

  def parameterized_linearizer(
      self,
      index: Index,
      primitive: Primitive,
      *input_info: Tuple[Sequence[int], bool],
      **params) -> ParameterizedNodeRelaxation[LinFun]:
    if primitive in self._primitive_mapper:
      params = synthetic_primitives.filter_jaxverify_kwargs(params)
      return self._primitive_mapper[primitive](*input_info, **params)
    elif self._default_parameterized_relaxer:
      return self._default_parameterized_relaxer.parameterized_linearizer(
          index, primitive, *input_info, **params)
    else:
      raise ValueError(f'Unsupported primitive to relax: {primitive}.')


class ParameterizedLinearBoundsGlobalRelaxer(metaclass=abc.ABCMeta):
  """Relaxer mapping each primitive to an optimisable linear propagator."""

  @abc.abstractmethod
  def parameterized_global_linearizer(
      self,
      index: Index,
      primitive: Primitive,
      network_input_spec: bound_propagation.Bound,
      *input_shapes: Sequence[int],
      **params) -> ParameterizedNodeRelaxation[LinearExpression]:
    """Obtain a parameterised family of linear bound propagator.

    Compared to the non-global version, this requires the additional
    network_input_spec parameters.

    Args:
      index: Index of the node in the bound propagation.
      primitive: Primitive to relax.
      network_input_spec: Bound over the input of the network.
      *input_shapes: Shapes of the inputs of the primitive.
      **params: Parameters of the primitive.

    Returns:
      Object producing linearised lower/upper bounds given relaxation
      parameters.
    """


class OpwiseParameterizedLinearBoundsGlobalRelaxer(
    ParameterizedLinearBoundsGlobalRelaxer):
  """Relaxer mapping each primitive to a parameterized linear relaxation."""

  def __init__(
      self,
      primitive_mapper: Mapping[
          Primitive,
          Callable[..., ParameterizedNodeRelaxation[LinearExpression]]],
      default_param_relaxer: Optional[ParameterizedLinearBoundsGlobalRelaxer
                                      ] = None):
    self._primitive_mapper = primitive_mapper
    self._default_parameterized_relaxer = default_param_relaxer

  def parameterized_global_linearizer(
      self,
      index: Index,
      primitive: Primitive,
      network_input_spec: bound_propagation.Bound,
      *input_shapes: Sequence[int],
      **params) -> ParameterizedNodeRelaxation[LinearExpression]:
    if primitive in self._primitive_mapper:
      params = synthetic_primitives.filter_jaxverify_kwargs(params)
      return self._primitive_mapper[primitive](network_input_spec,
                                               *input_shapes, **params)
    elif self._default_parameterized_relaxer:
      return (
          self._default_parameterized_relaxer.parameterized_global_linearizer(
              index, primitive, network_input_spec, *input_shapes, **params))
    else:
      raise ValueError(f'Unsupported primitive to relax: {primitive}.')


class BindRelaxerParams(LinearBoundsRelaxer):
  """Relaxer formed from binding parameters into a parameterised relaxer."""

  def __init__(
      self,
      node_relaxations: Mapping[Index, ParameterizedNodeRelaxation[LinFun]],
      relax_params: Mapping[Index, Tensor],
  ):
    self._node_relaxations = node_relaxations
    self._relax_params = relax_params

  def linearize_primitive(
      self,
      index: Index,
      primitive: Primitive,
      *inps: bound_propagation.LayerInput,
      **params) -> Tuple[LinFun, LinFun]:
    return self._node_relaxations[index].linearize(
        self._relax_params[index], *inps)


class ParameterizedLinFun(Generic[LinearRelax], metaclass=abc.ABCMeta):
  """Linearisation of lower OR upper bound using optimisable parameters."""

  @property
  @abc.abstractmethod
  def arity(self) -> int:
    """Returns the number of input arguments expected by the linear function."""

  @abc.abstractmethod
  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: bound_propagation.LayerInput,
  ) -> LinearRelax:
    """Returns linearised half-bound for given optimisable parameters."""

  @abc.abstractmethod
  def initial_params(
      self,
      *inps: Optional[bound_propagation.LayerInput],
  ) -> Nest[Tensor]:
    """Returns initial values of optimisable parameters."""

  @abc.abstractmethod
  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    """Projects optimisable parameters to their valid domain."""


class LowerUpperRelaxation(ParameterizedNodeRelaxation[LinearRelax]):
  """Adapts two parameterised half-bounds into a parameterised relaxation."""

  def __init__(self, lower: ParameterizedLinFun[LinearRelax],
               upper: ParameterizedLinFun[LinearRelax]):
    super().__init__()
    self._lower = lower
    self._upper = upper

  @property
  def arity(self) -> int:
    return self._lower.arity

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: bound_propagation.LayerInput,
  ) -> Tuple[LinearRelax, LinearRelax]:
    lower_relax_params, upper_relax_params = relax_params
    return (
        self._lower.linearize(lower_relax_params, *inps),
        self._upper.linearize(upper_relax_params, *inps))

  def initial_params(
      self,
      *inps: Optional[bound_propagation.LayerInput],
  ) -> Nest[Tensor]:
    return (
        self._lower.initial_params(*inps),
        self._upper.initial_params(*inps))

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    lower_relax_params, upper_relax_params = relax_params
    return (
        self._lower.project_params(lower_relax_params),
        self._upper.project_params(upper_relax_params))


class _NoParamRelaxation(ParameterizedNodeRelaxation[LinearRelax]):
  """Adapts a simple relaxer function into a zero-parameter relaxer function."""

  def __init__(
      self,
      relaxer: ArgsKwargsCallable[
          bound_propagation.LayerInput, Tuple[LinFun, LinFun]],
      *input_info: Tuple[Sequence[int], bool],
      **params):
    self._relaxer = relaxer
    self._input_info = input_info
    self._params = params

  @property
  def arity(self):
    return len(self._input_info)

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: bound_propagation.LayerInput,
  ) -> Tuple[LinearRelax, LinearRelax]:
    return self._relaxer(*inps, **self._params)

  def initial_params(
      self,
      *inps: Optional[bound_propagation.LayerInput],
  ) -> Nest[Tensor]:
    return ()

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return relax_params


no_params = lambda relaxer: functools.partial(_NoParamRelaxation, relaxer)


def linearized(fn: TensorFun, *primals: Tensor) -> LinFun:
  """Returns linear function that is tangent to `fn` at given primal point."""
  val, deriv = jax.linearize(fn, *primals)
  return lambda *xs: val + deriv(*[x - p for x, p in zip(xs, primals)])


class SupportingHyperplane(ParameterizedLinFun[LinFun]):
  """Linearisation of primitive, parameterised by primal point."""

  def __init__(
      self,
      fn: TensorFun,
      *input_info: Tuple[Sequence[int], bool]):
    super().__init__()
    self._fn = fn
    self._input_info = input_info

  @property
  def arity(self) -> int:
    return len(self._input_info)

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: bound_propagation.LayerInput,
  ) -> LinFun:
    primals = [
        alpha * inp.upper + (1.-alpha) * inp.lower
        if isinstance(inp, bound_propagation.Bound) else inp
        for inp, alpha in zip(inps, relax_params)]
    return linearized(self._fn, *primals)

  def initial_params(
      self,
      *inps: Optional[bound_propagation.LayerInput],
  ) -> Nest[Tensor]:
    # If an input is [known to be] a fixed tensor, don't allocate a
    # relaxation parameter for it.
    return [  # pytype: disable=bad-return-type  # jax-ndarray
        .5 * jnp.ones(shape=inp_shape)
        if inp is None or isinstance(inp, bound_propagation.Bound) else None
        for inp, (inp_shape, _) in zip(inps, self._input_info)]

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return jax.tree_map(lambda x: jnp.clip(x, 0., 1.), relax_params)


def eltwise_linfun_from_coeff(slope: Tensor, offset: Tensor) -> LinFun:
  return lambda x: slope * x + offset


class ElementwiseChord(ParameterizedLinFun[LinFun]):
  """Chord between input bounds of an element-wise primitive."""

  arity = 1

  def __init__(
      self,
      fn: TensorFun,
      input_info: Tuple[Sequence[int], bool]):
    super().__init__()
    self._fn = fn
    self._input_info = input_info

  def linearize(
      self,
      relax_params: Nest[Tensor],
      inp: bound_propagation.LayerInput,
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
      *inps: Optional[bound_propagation.LayerInput],
  ) -> Nest[Tensor]:
    return ()

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return relax_params


def elementwise_convex_fn_relaxer(
    primitive: Primitive,
    input_info: Tuple[Sequence[int], bool],
    **params) -> ParameterizedNodeRelaxation[LinFun]:
  fn = functools.partial(primitive.bind, **params)
  lower = SupportingHyperplane(fn, input_info)
  upper = ElementwiseChord(fn, input_info)
  return LowerUpperRelaxation(lower, upper)


def elementwise_concave_fn_relaxer(
    primitive: Primitive,
    input_info: Tuple[Sequence[int], bool],
    **params) -> ParameterizedNodeRelaxation[LinFun]:
  fn = functools.partial(primitive.bind, **params)
  lower = ElementwiseChord(fn, input_info)
  upper = SupportingHyperplane(fn, input_info)
  return LowerUpperRelaxation(lower, upper)


class _SmoothConvexRelaxation(LowerUpperRelaxation):
  """Linear relaxation from supporting hyperplanes of a convex relaxation."""

  def __init__(
      self,
      convex_relaxer: activation_relaxation.RelaxationFn,
      *input_info: Tuple[Sequence[int], bool],
      **params):
    # Create a skeleton `ParameterizedLinFun` for initialisation and projection
    # relaxation parameters. This doesn't need the lower/upper bound functions.
    skeleton = SupportingHyperplane((lambda *_: None), *input_info)
    super().__init__(skeleton, skeleton)

    self._convex_relaxer = convex_relaxer
    self._input_info = input_info
    self._params = params

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inps: bound_propagation.LayerInput,
  ) -> Tuple[LinFun, LinFun]:
    # First obtain convex lower and concave upper bounds.
    # Do so at this late stage because they depend on the current input bounds.
    lb_fun, ub_fun = self._convex_relaxer(*inps, **self._params)

    lower = SupportingHyperplane(lb_fun, *self._input_info)
    upper = SupportingHyperplane(ub_fun, *self._input_info)

    lower_relax_params, upper_relax_params = relax_params
    return (
        lower.linearize(lower_relax_params, *inps),
        upper.linearize(upper_relax_params, *inps))


def linear_from_smooth_convex(
    convex_relaxer: activation_relaxation.RelaxationFn,
) -> ArgsKwargsCallable[Sequence[int], ParameterizedNodeRelaxation]:
  return functools.partial(_SmoothConvexRelaxation, convex_relaxer)


def midpoint_relaxer(
    primitive: Primitive,
    *inputs: bound_propagation.LayerInput,
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
      0.5 * (x.lower + x.upper)
      if isinstance(x, bound_propagation.Bound) else x for x in inputs]

  lb_lin_fun = linearized(lb_fun, *mid_points)
  ub_lin_fun = ub_fun if convex else linearized(ub_fun, *mid_points)
  return lb_lin_fun, ub_lin_fun


def _fastlin_relu_relaxer(
    inp: bound_propagation.Bound) -> Tuple[LinFun, LinFun]:
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


def _rvt_exp_relaxer(inp: bound_propagation.Bound) -> Tuple[LinFun, LinFun]:
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
    x: bound_propagation.Bound,
    y: bound_propagation.Bound,
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


def _maybe_interp_funs(
    funs: Sequence[TensorFun],
    interp_params: Nest[Tensor],
    *args: Tensor,
) -> Tensor:
  """Interpolates between `funs` according to the given interpolation params."""
  vals = [fun(*args) for fun in funs]
  return utils.maybe_interp(vals, interp_params)


class ParameterizedPiecewiseLinearSubgradient(
    ParameterizedNodeRelaxation[LinFun]):
  """Relaxation of a piecewise-linear function.

  This implementation currently assumes exactly two pieces.
  """

  def __init__(
      self,
      primitive: Primitive,
      piecewise_linear_relaxation_fn: (
          activation_relaxation.PiecewiseLinearRelaxationFn),
      *input_info: Tuple[Sequence[int], bool],
      soft_init: bool = True,
      **params):
    super().__init__()
    self._primitive = primitive
    self._piecewise_linear_relaxation_fn = piecewise_linear_relaxation_fn
    self._input_info = input_info
    self._soft_init = soft_init
    self._params = params

  @property
  def arity(self):
    return len(self._input_info)

  def linearize(
      self,
      relax_params: Nest[Tensor],
      *inputs: bound_propagation.LayerInput,
  ) -> Tuple[LinFun, LinFun]:
    lb_funs, ub_funs = self._piecewise_linear_relaxation_fn(*inputs,
                                                            **self._params)
    lb_relax_params, ub_relax_params = relax_params
    return (
        functools.partial(_maybe_interp_funs, lb_funs, lb_relax_params),
        functools.partial(_maybe_interp_funs, ub_funs, ub_relax_params))

  def initial_params(
      self,
      *inps: Optional[bound_propagation.LayerInput],
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
          *[jnp.zeros(shape) for shape, _ in self._input_info],
          **self._params).shape
      params = .5 * jnp.ones(shape=output_shape, dtype=jnp.float32)

    # Determine the number of linear pieces of the lower and upper bounds.
    lb_funs, ub_funs = self._piecewise_linear_relaxation_fn(
        *[
            bound_propagation.IntervalBound(jnp.zeros(shape), jnp.zeros(shape))
            if is_bound else jnp.zeros(shape)
            for shape, is_bound in self._input_info
        ], **self._params)

    return (() if len(lb_funs) <= 1 else params,
            () if len(ub_funs) <= 1 else params)

  def project_params(self, relax_params: Nest[Tensor]) -> Nest[Tensor]:
    return jax.tree_map(lambda x: jnp.clip(x, 0., 1.), relax_params)


_crown_mapper: Mapping[
    Primitive,
    ArgsKwargsCallable[bound_propagation.LayerInput, Tuple[LinFun, LinFun]],
] = {
    **{prim: functools.partial(midpoint_relaxer, prim, convex=activation.convex)
       for prim, activation in activation_relaxation.relaxation_fns.items()},
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
}
crown_rvt_relaxer = OpwiseLinearBoundsRelaxer(_crown_mapper)

_fastlin_mapper: Mapping[
    Primitive,
    ArgsKwargsCallable[bound_propagation.LayerInput, Tuple[LinFun, LinFun]],
] = {
    **{prim: functools.partial(midpoint_relaxer, prim, convex=activation.convex)
       for prim, activation in activation_relaxation.relaxation_fns.items()},
    synthetic_primitives.relu_p: _fastlin_relu_relaxer,
    lax.exp_p: _rvt_exp_relaxer,
    synthetic_primitives.posbilinear_p: _rvt_posbilinear_relaxer,
}
fastlin_rvt_relaxer = OpwiseLinearBoundsRelaxer(_fastlin_mapper)


def _make_parameterized_relaxer(
    primitive: Primitive,
    activation: activation_relaxation.ActivationRelaxation,
) -> ArgsKwargsCallable[Sequence[int], ParameterizedNodeRelaxation]:
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


_parameterized_mapper: Mapping[
    Primitive,
    ArgsKwargsCallable[Sequence[int], ParameterizedNodeRelaxation],
] = {
    primitive: _make_parameterized_relaxer(primitive, activation)
    for primitive, activation in activation_relaxation.relaxation_fns.items()}
parameterized_relaxer = OpwiseParameterizedLinearBoundsRelaxer(
    _parameterized_mapper)
