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

"""Implementation of Forward propagation of linear bounds.
"""
import functools
from typing import Dict, Iterator, Union, Sequence, Tuple

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import ibp
from jax_verify.src import intersection
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils
from jax_verify.src.linear import linear_bound_utils
import numpy as np

Bound = bound_propagation.Bound
Index = graph_traversal.Index
Tensor = jnp.ndarray
Primitive = graph_traversal.Primitive
TransformContext = graph_traversal.TransformContext
LinFun = linear_bound_utils.LinFun
LinearExpression = linear_bound_utils.LinearExpression

EPSILON = 1e-5


class RefBound:
  """Wrapper around a bound so that we can use it as a key."""

  def __init__(self, index: Index, bound: Bound):
    self._index = index
    self.bound = bound
    self.nb_coeffs = np.prod(bound.lower.shape)

  def __hash__(self):
    return hash(self._index)

  def __eq__(self, other):
    return self._index == other._index


class LinearFunction:
  """Represent a pair of linear functions that encompass feasible activations.

  We store the linear functions as LinearExpressions objects in `lower_lin` and
  `upper_lin`, and also maintain a reference to the initial bounds on the input
  to be able to concretize the bounds when needed.
  """

  def __init__(self,
               lower_linexp: LinearExpression,
               upper_linexp: LinearExpression,
               reference_bound: RefBound):
    self.lower_lin = lower_linexp
    self.upper_lin = upper_linexp
    self.reference_bound = reference_bound

  @property
  def shape(self):
    return self.lower_lin.shape

  @property
  def nb_coeffs(self):
    return self.reference_bound.nb_coeffs

  def concretize(self) -> Tuple[Tensor, Tensor]:
    """Concretize the linear functions to obtain scalar bounds."""
    nb_act = len(self.shape)
    broad_shape = (-1,) + (1,)*nb_act
    flat_ref_lb = jnp.reshape(self.reference_bound.bound.lower, broad_shape)
    flat_ref_ub = jnp.reshape(self.reference_bound.bound.upper, broad_shape)

    concrete_lb = (
        (jnp.maximum(self.lower_lin.lin_coeffs, 0.) * flat_ref_lb).sum(axis=0) +
        (jnp.minimum(self.lower_lin.lin_coeffs, 0.) * flat_ref_ub).sum(axis=0) +
        self.lower_lin.offset
    )
    concrete_ub = (
        (jnp.maximum(self.upper_lin.lin_coeffs, 0.) * flat_ref_ub).sum(axis=0) +
        (jnp.minimum(self.upper_lin.lin_coeffs, 0.) * flat_ref_lb).sum(axis=0) +
        self.upper_lin.offset
    )
    return concrete_lb, concrete_ub

  def __add__(self, other: Union['LinearFunction', Tensor]) -> 'LinearFunction':
    if isinstance(other, LinearFunction):
      if self.reference_bound != other.reference_bound:
        raise ValueError('Adding linear functions referring to '
                         'different inputs.')
      return LinearFunction(self.lower_lin + other.lower_lin,
                            self.upper_lin + other.upper_lin,
                            self.reference_bound)
    else:
      return LinearFunction(self.lower_lin + other,
                            self.upper_lin + other,
                            self.reference_bound)

  def __radd__(self, other: Union['LinearFunction', Tensor]
               ) -> 'LinearFunction':
    return self.__add__(other)

  def __sub__(self, other: Union['LinearFunction', Tensor]) -> 'LinearFunction':
    if isinstance(other, LinearFunction):
      if self.reference_bound != other.reference_bound:
        raise ValueError('Substracting linear functions referring to '
                         'different inputs.')
      return LinearFunction(self.lower_lin - other.upper_lin,
                            self.upper_lin - other.lower_lin,
                            self.reference_bound)
    else:
      return LinearFunction(self.lower_lin - other,
                            self.upper_lin - other,
                            self.reference_bound)

  def __rsub__(self, other: Union['LinearFunction', Tensor]
               ) -> 'LinearFunction':
    if isinstance(other, LinearFunction):
      if self.reference_bound != other.reference_bound:
        raise ValueError('Substracting linear functions referring to '
                         'different inputs.')
      return LinearFunction(other.lower_lin - self.upper_lin,
                            other.upper_lin - self.lower_lin,
                            self.reference_bound)
    else:
      return LinearFunction(other - self.upper_lin,
                            other - self.lower_lin,
                            self.reference_bound)


class LinearBound(bound_propagation.Bound):
  """Linear bound over activations.

  This is composed of several linear functions because the networks might have
  several inputs.
  """

  def __init__(self, linear_functions: Sequence[LinearFunction]):
    self._refbound_to_linfun: Dict[RefBound, LinearFunction] = {}
    for function in linear_functions:
      ref_bound = function.reference_bound
      if ref_bound in self._refbound_to_linfun:
        self._refbound_to_linfun[ref_bound] += function
      else:
        self._refbound_to_linfun[ref_bound] = function
    self._concretized = None
    self._shape = next(iter(linear_functions)).shape

  @property
  def lower(self):
    if self._concretized is None:
      self.concretize()
    return self._concretized.lower

  @property
  def upper(self):
    if self._concretized is None:
      self.concretize()
    return self._concretized.upper

  @property
  def shape(self):
    return self._shape

  def concretize(self):
    if self._concretized is not None:
      return self._concretized
    lb = jnp.zeros(())
    ub = jnp.zeros(())
    for lin_fun in self._refbound_to_linfun.values():
      lin_fun_lb, lin_fun_ub = lin_fun.concretize()
      lb = lb + lin_fun_lb
      ub = ub + lin_fun_ub
    self._concretized = ibp.IntervalBound(lb, ub)
    return self._concretized

  def set_concretized(self, interval_bound):
    self._concretized = interval_bound

  @staticmethod
  def initial_linear_bound(index, lower_bound, upper_bound):
    input_dim = np.prod(lower_bound.shape)
    lin_coeffs = jnp.reshape(jnp.eye(input_dim),
                             (input_dim, *lower_bound.shape))
    offsets = jnp.zeros_like(lower_bound)
    identity_lin = LinearExpression(lin_coeffs, offsets)

    reference_bound = ibp.IntervalBound(lower_bound, upper_bound)
    lin_function = LinearFunction(identity_lin, identity_lin,
                                  RefBound(index, reference_bound))

    lin_bound = LinearBound([lin_function])
    lin_bound.set_concretized(ibp.IntervalBound(lower_bound, upper_bound))

    return lin_bound

  def get_linfun(self, ref_bound: RefBound) -> LinearFunction:
    """Get the coefficients of this linear bound with regards to the RefBound."""
    if ref_bound in self._refbound_to_linfun:
      return self._refbound_to_linfun[ref_bound]
    else:
      # This LinearBound does not depend on the input requested. Equivalently,
      # it depends on it with all-coefficients and offsets equal to 0.
      lincoeffs_shape = (ref_bound.nb_coeffs, *self.shape)
      zero_lincoeffs = jnp.zeros(lincoeffs_shape)
      zero_offsets = jnp.zeros(self.shape)
      zero_linexpr = LinearExpression(zero_lincoeffs, zero_offsets)
      return LinearFunction(zero_linexpr, zero_linexpr, ref_bound)

  def linear_functions(self) -> Iterator[LinearFunction]:
    for lin_fun in self._refbound_to_linfun.values():
      yield lin_fun

  def reference_bounds(self) -> Iterator[RefBound]:
    for ref_bound in self._refbound_to_linfun:
      yield ref_bound


def _get_linop_parts(
    lin_op: LinFun,
    invals: Sequence[LinearBound],
    lin_is_positive: bool = False,
    lin_is_negative: bool = False,
) -> Tuple[Tuple[LinFun, LinFun], Tuple[LinFun, LinFun], Tensor]:
  """Extract the functions that implement positive/negative parts of lin_op.

  For convenience, we will also return their vmapped version.

  Args:
    lin_op: Linear function to decompose into a positive and negative part.
    invals: Inputs to the function.
    lin_is_positive: Optional argument to guarantee that the linear function
      has only positive coefficients.
    lin_is_negative: Optional argument to guarantee that the linear function
      has only negative coefficients.
  Returns:
    pos_part_op, neg_part_op: Linear function implementing the positive/negative
      part of lin_op
    pos_part_vop, neg_part_vop: Vmapped version of those functions.
    offset: Constant part of the lin_op
  """

  zero_inps = [jnp.zeros(inval.shape) for inval in invals]
  offset = lin_op(*zero_inps)

  # In case we are told that the linear function has only positive, or only
  # negative coefficients, we do not need to do do parameter identification.
  # We also set up a dummy function doing no computation in the part that does
  # not exist.
  # If we have no guarantees, we have to identify the positive and negative part
  # of the linear function.
  if lin_is_positive:
    assert not lin_is_negative
    pos_part_op = lambda *args: lin_op(*args) - offset
    neg_part_op = lambda *_: jnp.zeros(())
    pos_part_vop = jax.vmap(pos_part_op, in_axes=0, out_axes=0)
    neg_part_vop = neg_part_op
  elif lin_is_negative:
    pos_part_op = lambda *_: jnp.zeros(())
    neg_part_op = lambda *args: lin_op(*args) - offset
    pos_part_vop = pos_part_op
    neg_part_vop = jax.vmap(neg_part_op, in_axes=0, out_axes=0)
  else:
    jac_fun = jax.jacfwd(lin_op, argnums=tuple(range(len(zero_inps))))
    jacobians = jac_fun(*zero_inps)

    def pos_part_op(*args):
      pospart = jnp.zeros(())
      for arg, jac in zip(args, jacobians):
        pos_jac = jnp.maximum(jac, 0.)
        pospart += (arg * pos_jac).sum(axis=tuple(range(-arg.ndim, 0)))
      return pospart
    def neg_part_op(*args):
      negpart = jnp.zeros(())
      for arg, jac in zip(args, jacobians):
        neg_jac = jnp.minimum(jac, 0.)
        negpart += (arg * neg_jac).sum(axis=tuple(range(-arg.ndim, 0)))
      return negpart

    pos_part_vop = jax.vmap(pos_part_op, in_axes=0, out_axes=0)
    neg_part_vop = jax.vmap(neg_part_op, in_axes=0, out_axes=0)

  return (pos_part_op, neg_part_op), (pos_part_vop, neg_part_vop), offset


def _forward_propagate_linear_bounds(lb_lin_op: LinFun,
                                     ub_lin_op: LinFun,
                                     invals: Sequence[LinearBound],
                                     lin_is_positive: bool = False,
                                     lin_is_negative: bool = False
                                     ) -> LinearBound:
  """Propagate linear bounds through a primitive relaxed to its linear bounds.

  We assume that we have linear bounds on the inputs of the function.
  The lin_is_positive/lin_is_negative arguments are optional but will
  help making the propagation more efficient if we have some information
  about the linear function that we need to propagate through.

  Args:
    lb_lin_op: Linear function, with only bound arguments that is a lower bound
      on the function we want to propagate through. All the constant inputs and
      parameters should have been bound.
    ub_lin_op: Linear function, with only bound arguments that is an upper bound
      on the function we want to propagate through. All the constant inputs and
      parameters should have been bound.
    invals: List of bounds that are inputs to lb_lin_op / ub_lin_op
    lin_is_positive: Optional argument, set to True if the linear functions
      are guaranteed to have only positive coefficients.
    lin_is_negative: Optional argument, set to True if the linear functions
      are guaranteed to have only negative coefficients.
  Returns:
    out_lin_bound: LinearBound on the output of the linear function.
  """
  ((lb_pospart_op, lb_negpart_op),
   (lb_pospart_vop, lb_negpart_vop),
   lb_offset) = _get_linop_parts(lb_lin_op, invals,
                                 lin_is_positive, lin_is_negative)
  ((ub_pospart_op, ub_negpart_op),
   (ub_pospart_vop, ub_negpart_vop),
   ub_offset) = _get_linop_parts(ub_lin_op, invals,
                                 lin_is_positive, lin_is_negative)
  all_ref_bound = set()
  for arg in invals:
    for ref_bound in arg.reference_bounds():
      all_ref_bound.add(ref_bound)

  out_linfuns = []
  lb_offset_shared = lb_offset / len(all_ref_bound)
  ub_offset_shared = ub_offset / len(all_ref_bound)
  for ref_bound in all_ref_bound:
    in_linfuns = []
    for arg in invals:
      in_linfuns.append(arg.get_linfun(ref_bound))

    lower_lincoeffs = (lb_pospart_vop(*(linfun.lower_lin.lin_coeffs
                                        for linfun in in_linfuns))
                       + lb_negpart_vop(*(linfun.upper_lin.lin_coeffs
                                          for linfun in in_linfuns)))
    lower_offset = (lb_pospart_op(*(linfun.lower_lin.offset
                                    for linfun in in_linfuns))
                    + lb_negpart_op(*(linfun.upper_lin.offset
                                      for linfun in in_linfuns))
                    + lb_offset_shared)
    lower_linexpr = LinearExpression(lower_lincoeffs, lower_offset)

    upper_lincoeffs = (ub_pospart_vop(*(linfun.upper_lin.lin_coeffs
                                        for linfun in in_linfuns))
                       + ub_negpart_vop(*(linfun.lower_lin.lin_coeffs
                                          for linfun in in_linfuns)))
    upper_offset = (ub_pospart_op(*(linfun.upper_lin.offset
                                    for linfun in in_linfuns))
                    + ub_negpart_op(*(linfun.lower_lin.offset
                                      for linfun in in_linfuns))
                    + ub_offset_shared)
    upper_linexpr = LinearExpression(upper_lincoeffs, upper_offset)

    out_linfun = LinearFunction(lower_linexpr, upper_linexpr, ref_bound)
    out_linfuns.append(out_linfun)

  return LinearBound(out_linfuns)


def _fastlin_bilinearwithparam_op(primitive: Primitive,
                                  lhs: Union[LinearBound, Tensor],
                                  rhs: Union[LinearBound, Tensor],
                                  **kwargs) -> LinearBound:
  """Propagation of Linear bounds through an affine operation.

  This operation is implemented by one of the bilinear primitives so we know
  exactly how to do the bound propagation without having to materialize the
  jacobian to obtain weights.

  Args:
    primitive: Linear function to pass through.
    lhs: Either parameters or LinearBound.
    rhs: Either parameters or LinearBound.
    **kwargs: Dict with the parameters of the linear operation.
  Returns:
    out_bounds: LinearBound
  """
  # Detect which order things are in, so that we can do forward propagation
  # simply by calling `fun_call(bound_arg, param_arg)`, whatever the ordering
  # initially was.
  if isinstance(lhs, bound_propagation.Bound):
    assert not isinstance(rhs, bound_propagation.Bound)
    bound_arg = lhs
    param_arg = rhs
    fun_call = functools.partial(primitive.bind, **kwargs)
  else:
    assert isinstance(rhs, bound_propagation.Bound)
    bound_arg = rhs
    param_arg = lhs
    fun_call = lambda b_arg, p_arg: primitive.bind(p_arg, b_arg, **kwargs)

  vmap_funcall = jax.vmap(fun_call, in_axes=(1, None), out_axes=1)

  # Extract the parameters for the bound propagation.
  abs_params = jnp.abs(param_arg)

  # Get access to the LinearBound, in case it is wrapped in an
  # IntersectionBound.
  unwrapped_bound = bound_arg.unwrap()
  # Iterate over the different linear functions that the bound is composed of.
  out_linfuns = []
  for lin_fun in unwrapped_bound.linear_functions():
    range_lin = (lin_fun.upper_lin - lin_fun.lower_lin) / 2
    mean_lin = (lin_fun.upper_lin + lin_fun.lower_lin) / 2
    ref_bound = lin_fun.reference_bound

    out_range_lin_coeffs = vmap_funcall(range_lin.lin_coeffs, abs_params)
    out_range_offset = fun_call(range_lin.offset, abs_params)

    out_mean_lin_coeffs = vmap_funcall(mean_lin.lin_coeffs, param_arg)
    out_mean_offset = fun_call(mean_lin.offset, param_arg)

    out_lowerlinexp = LinearExpression(
        out_mean_lin_coeffs - out_range_lin_coeffs,
        out_mean_offset - out_range_offset)
    out_upperlinexp = LinearExpression(
        out_mean_lin_coeffs + out_range_lin_coeffs,
        out_mean_offset + out_range_offset)

    out_linfun = LinearFunction(out_lowerlinexp, out_upperlinexp, ref_bound)
    out_linfuns.append(out_linfun)

  return LinearBound(out_linfuns)


@bound_propagation.unwrapping
def _fastlin_add(lhs, rhs):
  return lhs + rhs


@bound_propagation.unwrapping
def _fastlin_sub(lhs, rhs):
  return lhs - rhs


def _fastlin_div(lhs, rhs):
  """Propagation of Linear bounds through Elementwise division.

  We don't support the propagation of bounds through the denominator.

  Args:
    lhs: Numerator of the division.
    rhs: Denominator of the division.
  Returns:
    out_bounds: Bound on the output of the division.
  """
  if isinstance(rhs, bound_propagation.Bound):
    raise ValueError('Bound propagation through the denominator unsupported.')
  return _fastlin_bilinearwithparam_op(lax.mul_p, lhs, 1./rhs)


def forward_fastlin_bound_propagation(function, *bounds):
  """Performs forward linear bound propagation.

  This is using the relu relaxation of Fastlin.
  (https://arxiv.org/abs/1804.09699)

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  output_bound, _ = bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(forward_fastlin_transform),
      function, *bounds)
  return output_bound


def forward_crown_bound_propagation(function, *bounds):
  """Performs forward linear bound propagation.

  This is using the relu relaxation of CROWN.
  (https://arxiv.org/abs/1811.00866)

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  forward_crown_transform = ForwardLinearBoundTransform(
      linear_bound_utils.crown_rvt_relaxer)
  output_bound, _ = bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(forward_crown_transform),
      function, *bounds)
  return output_bound


def ibpforwardfastlin_bound_propagation(function, *bounds):
  """Obtains the best of IBP and ForwardFastlin bounds.

  Args:
    function: Function performing computation to obtain bounds for. Takes as
      only argument the network inputs.
    *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
  Returns:
    output_bound: Bounds on the output of the function obtained by FastLin
  """
  output_bound, _ = bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(
          intersection.IntersectionBoundTransform(ibp.bound_transform,
                                                  forward_fastlin_transform)),
      function, *bounds)
  return output_bound


class ForwardLinearBoundTransform(
    graph_traversal.GraphTransform[LinearBound]):
  """Propagate Linear bounds forward through the network."""

  def __init__(self, relaxer: linear_bound_utils.LinearBoundsRelaxer,
               linear_elision: bool = False):
    self.relaxer = relaxer
    self._linear_elision = linear_elision

  def should_handle_as_subgraph(self, primitive: Primitive) -> bool:
    if primitive is synthetic_primitives.linear_p:
      # If we do linear elision, we do not want to treat linear_p as a subgraph.
      # (we want to treat it as a single layer.)
      return not self._linear_elision
    else:
      return super().should_handle_as_subgraph(primitive)

  def input_transform(self, context: TransformContext,
                      lower: Tensor,
                      upper: Tensor) -> LinearBound:
    return LinearBound.initial_linear_bound(context.index, lower, upper)

  def primitive_transform(self, context: TransformContext,
                          primitive: Primitive,
                          *args: Union[Bound, Tensor],
                          **params) -> LinearBound:
    # We specialise bilinear and reshape operation because we can compute them
    # in a more lightweight manner, without having to resort to identifying
    # parameters.
    if primitive in bound_propagation.BILINEAR_PRIMITIVES:
      return _fastlin_bilinearwithparam_op(primitive, *args, **params)
    elif (primitive in bound_propagation.AFFINE_PRIMITIVES
          or primitive in bound_propagation.RESHAPE_PRIMITIVES
          or (primitive is lax.div_p and isinstance(args[1], Tensor))):
      is_positive = primitive in POSLINEAR_PRIMITIVES
      safe_params = utils.filter_jaxverify_kwargs(params)
      lin_fun = utils.bind_nonbound_args(primitive.bind, *args, **safe_params)
      lin_bound_inputs = [arg.unwrap() for arg in args
                          if isinstance(arg, Bound)]
      return _forward_propagate_linear_bounds(lin_fun, lin_fun,
                                              lin_bound_inputs,
                                              lin_is_positive=is_positive)
    else:
      # This is not an affine primitive. We need to go through a relaxation.
      # Obtain the linear bounds.
      lb_linfun, ub_linfun = self.relaxer.linearize_primitive(
          context.index, primitive, *args, **params)
      lin_bound_inputs = [arg.unwrap() for arg in args
                          if isinstance(arg, Bound)]
      is_positive = primitive in POSITIVE_RELAXATION_PRIMITIVES
      is_negative = primitive in NEGATIVE_RELAXATION_PRIMITIVES

      return _forward_propagate_linear_bounds(
          lb_linfun, ub_linfun, lin_bound_inputs,
          lin_is_positive=is_positive, lin_is_negative=is_negative)

POSLINEAR_PRIMITIVES = [
    lax.scatter_add_p,
    lax.add_p,
    lax.reduce_sum_p,
] + bound_propagation.RESHAPE_PRIMITIVES

POSITIVE_RELAXATION_PRIMITIVES = [
    synthetic_primitives.relu_p,
    lax.exp_p,
]

NEGATIVE_RELAXATION_PRIMITIVES = [
    synthetic_primitives.posreciprocal_p,
]

forward_fastlin_transform = ForwardLinearBoundTransform(
    linear_bound_utils.fastlin_rvt_relaxer)
