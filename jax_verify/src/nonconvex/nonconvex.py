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

"""Build the nonconvex reformulation of the convex approximation of the network.

This is accomplished by traversing the JaxPR representation of the computation
and translating the computational graph.

This is based on the paper "An efficient nonconvex reformulation of stagewise
convex optimization problems" and the implementation was inspired by the
tensorflow version at:
/l/d/r/robust_verified/verification/ibp/verification/
nonconvex_optimizable_bounds.py
"""

import abc
import functools
from typing import Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union
from absl import logging

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import ibp
from jax_verify.src import intersection
from jax_verify.src import mccormick
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils

NnCvx = TypeVar('NnCvx', bound='NonConvexBound')
Tensor = jnp.ndarray
Index = bound_propagation.Index
TransformContext = bound_propagation.TransformContext
# Mapping of a position in the computation to a set of parameters.
# These can be variables, gradients, or coefficients.
ParamSet = Dict[Index, Tensor]
PrimitiveInput = Union[Tensor, 'NonConvexBound']
Nest = bound_propagation.Nest


def _sum_fn(fn, *args, **kwargs):
  out = fn(*args, **kwargs)
  summand = out[0] if isinstance(out, tuple) else out
  return summand.sum(), out


def _sum_over_acts(var: Tensor) -> Tensor:
  return var.sum(axis=tuple(range(1, var.ndim)))


class NonConvexBound(bound_propagation.Bound, metaclass=abc.ABCMeta):
  """Represent a bound that can be optimized.

  This is the object that gets propagated through the network.

  The important elements it exposes are the following:

  - variables: Specifies what is the shape of the parameters that need to be
      provided such that a bound can be computed.
  - dual: A function that takes as input variables as specified by `variables`
      and a set of linear function over the activations, in the form of a dict
      mapping activation index to (batch_dim , nb_opt, *act_dims) tensor, and
      that returns the value of the primal objectives when those variables are
      used, as well as a dual bound on those linear objectives. Different
      options for dual computation are available, each implemented in a separate
      sub-class.
  - primal_fn: Similar function, but does not compute the dual objective. It
      also has an additional `dummy_inps` inputs. When gradients with regards
      to the activation needs to be obtained, these can obtained by providing
      "0 activations" through this parameters, which would be added to the
      network activations. Differentiating with regards to these parameters will
      allow to obtain derivatives with regards to activations.
  Making use of these elements has already been implemented in the
  `BoundOptimizer` classes.

  Important bits of machinery shared by all subclasses:
  - _eval_fn is a function that takes as inputs tensors as specified by
    `variables` and evaluate the activations.
  - concretizer is an instance of `Concretizer`, that exposes a `get_bounds`
    method. This is used during the bound propagation when concrete bounds are
    required (such as when defining the convex hull relaxation of a ReLU.).
    The choice of concretizer impacts of those intermediate concrete bounds are
    computed. This can be done by relying on a fallback method such as IBP, or
    by optimization.

  In addition, the subclass are responsible for encoding some additional
  mechanisms required for the computation of the dual they implement.
  """

  def __init__(self,
               index: Index,
               shape: Tuple[int, ...],
               previous_bounds: Dict[Index, 'NonConvexBound'],
               eval_fn: Callable[[ParamSet, ParamSet, ParamSet], Tensor],
               variables: Dict[Index, Tuple[int, ...]],
               concretized_bounds: Optional[bound_propagation.Bound] = None
               ):
    """Shared computation for the creation of NonConvexBound.

    Args:
      index: Unique index representing the position of this activation in the
        computation graph.
      shape: Shape of the activations that this bound represent.
      previous_bounds: Dict mapping index of activation to the bound
        representing it. We need it to be able to obtain the contributions of
        previous layers to the Lagrangian.
      eval_fn: Function to evaluate the bound computation problem in the primal
      variables: Dict mapping index of activation to the shape of variables
        required to optimize them.
      concretized_bounds: (Optional) Precomputed bounds for this NonConvexBound.
    """
    self.index = index
    self._shape = shape
    self.previous_bounds = previous_bounds
    self.previous_bounds[index] = self
    self._eval_fn = eval_fn
    self.variables = variables
    self._concretized_bounds = concretized_bounds

    def primal(var_set: ParamSet,
               objectives: ParamSet,
               dummy_inps: Optional[ParamSet] = None
               ) -> Tuple[Tensor, ParamSet]:
      """Evaluate the primal objective of the problem.

      dummy_inps are inputs which are always zeros, and that we will add to
      every intermediate activations of the network for which we need gradients.
      This way, computing the gradients with regards to those "inputs" allows
      us to compute gradients with regards to intermediate activations, as
      required for the definition of the dual variables.

      Args:
        var_set: Dictionary mapping the position in network to a tensor
          containing the primal variables.
        objectives: Dict mapping the position in network to the coefficient of
          the linear objective function over the activations.
        dummy_inps: Dictionary mapping the position in network to a zero
          tensor.
      Returns:
         primals: All primal objectives.
      """
      acts = {}
      self.evaluate(var_set, dummy_inps, acts)
      primals = self._objective_fn(acts, objectives)
      return primals, acts

    self.primal_fn = primal
    self.primal_sumfn = functools.partial(_sum_fn, primal)

  def _objective_fn(self, acts, objectives):
    """Default objective function is a dot product with the activations."""
    primal_objs = sum(_sum_over_acts(acts[index] * act_objectives)
                      for index, act_objectives in objectives.items())
    return primal_objs

  @property
  def shape(self):
    return self._shape

  @property
  def lower(self) -> Tensor:
    if self._concretized_bounds is None:
      logging.warning('.lower called on a non-concretized bound.'
                      'Returning spurious bounds.')
      return -float('inf') * jnp.ones(self.shape)
    return self._concretized_bounds.lower

  @property
  def upper(self) -> Tensor:
    if self._concretized_bounds is None:
      logging.warning('.upper called on a non-concretized bound.'
                      'Returning spurious bounds.')
      return float('inf') * jnp.ones(self.shape)
    return self._concretized_bounds.upper

  def evaluate(self,
               var_set: ParamSet,
               dummy_inps: Optional[ParamSet] = None,
               acts: Optional[ParamSet] = None) -> Tensor:
    if acts is None:
      acts = {}
    if dummy_inps is None:
      dummy_inps = {}
    if self.index in acts:
      return acts[self.index]
    else:
      val = self._eval_fn(var_set, dummy_inps, acts)
      if self.index in dummy_inps:
        val = val + dummy_inps[self.index]
      acts[self.index] = val
      return val

  @abc.abstractmethod
  def dual(self, var_set: ParamSet, objectives: ParamSet) -> Tensor:
    """Compute the dual, using dual variables derived from primals in var_set.

    Returns both primal and dual, so that this can be used to compute dual gap.

    Args:
      var_set: Relaxation variables to use to compute the primal activations and
        derive the duals from.
      objectives: Dict mapping the position in network to the coefficient of
        the linear objective function over the activations.
    """

  @classmethod
  @abc.abstractmethod
  def get_initial_bound_constructor(
      cls: Type[NnCvx],
      index: Index,
      lb: Tensor,
      ub: Tensor) -> Callable[..., NnCvx]:
    """Class specific initialization for the input bounds of the network."""
    raise NotImplementedError('Initial bound constructor not implemented')

  @classmethod
  @abc.abstractmethod
  def get_linear_activation_constructor(
      cls: Type[NnCvx],
      index: Index,
      vlin_fun: Callable[..., Tensor],
      in_vals: Tuple[Tensor, ...]) -> Callable[..., NnCvx]:
    """Class specific initialization for the output of a linear function."""
    raise NotImplementedError('Linear activation constructor not implemented')

  @classmethod
  @abc.abstractmethod
  def get_nonlinearity_activation_constructor(
      cls: Type[NnCvx],
      index: Index,
      inp: NnCvx,
      act_type: str,
      lb_fun: Callable[[Tensor], Tensor],
      ub_fun: Callable[[Tensor], Tensor]) -> Callable[..., NnCvx]:
    """Class specific initialization for the output of a non-linearity."""
    raise NotImplementedError('Nonlinearity activation constructor not'
                              'implemented')

  @abc.abstractmethod
  def requires_concretizing(self, primitive) -> bool:
    """Returns whether the bounds need to be concretized.

    Args:
      primitive: Primitive where this bound is going to be fed into.
                 None indicates it is an output of the network.
    Returns:
      requires_concretizing: Indicate whether concretizing is required.
    """
    raise NotImplementedError('Specification of when concretization is required'
                              'is not implemented.')

  def _compute_dualvars_nonconvexgrad(self,
                                      var_set: ParamSet,
                                      objectives: ParamSet
                                      ) -> Tuple[ParamSet, ParamSet]:
    """Obtain dual vars based on derivatives of the nonconvex reformulation.

    Compute the gradients of all the primals objectives, (done by taking their
    sum), with regards to the dummy inputs of each activation.
    This differentiation is for the function as expressed in the nonconvex
    formulation, where each activation is a function of the previous
    activations.

    Args:
      var_set: Interpolation coefficients to derive the activations from.
      objectives: Arguments for the objective function.
    Returns:
      dual_vars: Dual variables corresponding to the derivatives of the primals
        wrt to the network activation, in the nonconvex reformulation.
      acts: Activations of the network.
    """
    grad_fun = jax.grad(self.primal_sumfn, argnums=2, has_aux=True)
    dummy_acts = {key: 0*val for key, val in var_set.items()}
    dual_vars, (_, acts) = grad_fun(var_set, objectives, dummy_acts)

    return dual_vars, acts

  def _compute_dualvars_convexgrad(self,
                                   var_set: ParamSet,
                                   objectives: ParamSet
                                   ) -> Tuple[ParamSet, ParamSet]:
    """Obtain dual vars based on derivatives of the objective function.

    Compute the gradients of all the primal objectives with regards to each
    activation.
    This differentiation is for the objective function expressed in the convex
    sense, where all the activation are considered variables rather than
    functions.

    Args:
      var_set: Interpolation coefficients to derive the activations from.
      objectives: Arguments for the objective function.
    Returns:
      dual_vars: Dual variables corresponding to the derivatives of the primals
        wrt to the network activation, in the convex formulation.
      acts: Activations of the network.
    """
    acts = {}
    self.evaluate(var_set, {}, acts)
    primal_gradfun_wrt_act = utils.batch_value_and_grad(
        self._objective_fn, (0,))
    _, dual_vars = primal_gradfun_wrt_act(acts, objectives)
    return dual_vars, acts

  def concretize(self, concretizer: 'Concretizer'):
    self._concretized_bounds = concretizer.get_bounds(self)

  @classmethod
  def initial_nonconvex_bound(
      cls: Type[NnCvx],
      index: Index,
      lower_bound: Tensor,
      upper_bound: Tensor) -> NnCvx:
    shape = lower_bound.shape
    variables = {index: lower_bound.shape}

    lb = jnp.expand_dims(lower_bound, axis=0)
    ub = jnp.expand_dims(upper_bound, axis=0)

    previous_bounds = {}

    def eval_fn(var_set, *_):
      val = lb + (ub - lb) * var_set[index]
      return val

    bound_ctor = cls.get_initial_bound_constructor(index, lb, ub)

    return bound_ctor(index, shape, previous_bounds,
                      eval_fn, variables, ibp.IntervalBound(lower_bound,
                                                            upper_bound))


class ConstrainedNonConvexBound(NonConvexBound, metaclass=abc.ABCMeta):
  """This special case of NonConvexBound supports `imposed_bounds` constraints.

  The assumption is that, before any evaluation of the primal or dual, the
  `set_imposed_bounds` function is called. As a result, these should be
  created through a `ConstrainedNonConvexTransform`

  * `lower` and `upper` return those by default before being concretized.
  * Concretizing will result in the concretized bounds being the tightest
    between the bounds obtained by concretizing and the imposed ones.
  * Imposing bounds will also constrain any existing concretized bounds.
  * Evaluating an activation represented by this bound will return the
    evaluation projected into the admissible bounds.
  """

  def __init__(self,
               index: Index,
               shape: Tuple[int, ...],
               previous_bounds: Dict[Index, 'ConstrainedNonConvexBound'],
               eval_fn: Callable[[ParamSet, ParamSet, ParamSet], Tensor],
               variables: Dict[Index, Tuple[int, ...]],
               concretized_bounds: Optional[bound_propagation.Bound] = None):
    super().__init__(index, shape, previous_bounds,
                     eval_fn, variables, concretized_bounds)
    self._imposed_bounds = None

  def is_constrained(self) -> bool:
    return self._imposed_bounds is not None

  def set_imposed_bounds(self, imposed_bounds: bound_propagation.Bound):
    self._imposed_bounds = imposed_bounds
    if self._concretized_bounds is not None:
      self._concretized_bounds = intersection.IntersectionBound(
          self._concretized_bounds, self._imposed_bounds)

  def evaluate(self,
               var_set: ParamSet,
               dummy_inps: Optional[ParamSet] = None,
               acts: Optional[ParamSet] = None) -> Tensor:
    """Activation implied by `var_set`, projected onto the bounds."""
    unconstrained_eval = super().evaluate(var_set, dummy_inps, acts)
    if not self.is_constrained():
      return unconstrained_eval

    brd_lower = jnp.expand_dims(self.lower, 0)
    brd_upper = jnp.expand_dims(self.upper, 0)
    if dummy_inps and (self.index in dummy_inps):
      # The dummy inp was added to the unconstrained eval, but we need to make
      # sure that it's present even in the constrained version.
      dummy_inp = dummy_inps[self.index]
      constrained_eval = jnp.clip(unconstrained_eval,
                                  brd_lower + dummy_inp, brd_upper + dummy_inp)
    else:
      constrained_eval = jnp.clip(unconstrained_eval, brd_lower, brd_upper)
    if acts:
      acts[self.index] = constrained_eval
    return constrained_eval

  @property
  def lower(self) -> Tensor:
    if self._imposed_bounds is not None and self._concretized_bounds is None:
      return self._imposed_bounds.lower
    return super().lower

  @property
  def upper(self) -> Tensor:
    if self._imposed_bounds is not None and self._concretized_bounds is None:
      return self._imposed_bounds.upper
    return super().upper

  def concretize(self, concretizer: 'Concretizer'):
    super().concretize(concretizer)
    if self._imposed_bounds is not None:
      # Ensure that the concretized bounds respect the imposed bounds.
      self._concretized_bound = intersection.IntersectionBound(
          self._concretized_bounds, self._imposed_bounds)


class Concretizer(abc.ABC):
  """Abstract class to define the API of concretizer.

  The role of Concretizer is to give access to concrete bounds to define
  relaxation while propagating NonConvexBound which are solver based.
  """

  @abc.abstractmethod
  def accept_input(
      self,
      context: TransformContext,
      lower_bound: Tensor, upper_bound: Tensor):
    """Update the concretizer based on the initial input bound."""

  @abc.abstractmethod
  def accept_primitive(
      self,
      context: TransformContext,
      primitive: bound_propagation.Primitive,
      *in_vals: PrimitiveInput,
      **params):
    """Update the concretizer based on the resulting bound."""

  @abc.abstractmethod
  def get_bounds(
      self, nonconvex_bound: NonConvexBound) -> bound_propagation.Bound:
    """Return a concretized bound."""


class BaseBoundConcretizer(Concretizer):
  """Concretizer based on performing a parallel propagation with another method.

  This should be initialized with the `input_transform` and the
  `primitive_transform` that are usually passed to `bound_propagation`.
  """

  def __init__(self, bound_transform: bound_propagation.BoundTransform):
    self._bound_transform = bound_transform
    self._base_bounds = {}

  def accept_input(
      self,
      context: TransformContext,
      lower_bound: Tensor, upper_bound: Tensor):
    self._base_bounds[context.index] = self._bound_transform.input_transform(
        context, lower_bound, upper_bound)

  def accept_primitive(
      self,
      context: TransformContext,
      primitive: bound_propagation.Primitive,
      *in_vals: PrimitiveInput,
      **params):
    base_in_vals = [
        self._base_bounds[inp.index] if isinstance(inp, NonConvexBound) else inp
        for inp in in_vals]
    self._base_bounds[context.index] = (
        self._bound_transform.equation_transform(
            context, primitive, *base_in_vals, **params))

  def get_bounds(self, nonconvex_bound: NonConvexBound
                 ) -> bound_propagation.Bound:
    return self._base_bounds[nonconvex_bound.index]


def eval_if_nonconvexbound(inp: Union[NonConvexBound, Tensor],
                           var_set: ParamSet,
                           dummy_inps: Optional[ParamSet],
                           activations: Optional[ParamSet]) -> Tensor:
  if isinstance(inp, NonConvexBound):
    return inp.evaluate(var_set, dummy_inps, activations)
  else:
    return inp


def _nonconvex_linear_op(primitive: bound_propagation.Primitive,
                         bound_cls: Type[NnCvx],
                         index: Index,
                         *in_vals: PrimitiveInput, **kwargs) -> NnCvx:
  """Propagation of NonConvex bounds through a linear operation.

  Args:
    primitive: Primitive that this linear operation implement.
    bound_cls: Bound class to use.
    index: Unique integer identifying position
    *in_vals: Input of the bound propagation in the forward pass
    **kwargs: Dict with the parameters of the linear operation
  Returns:
    out_bounds: nonconvex bounds on the operation's output
  """
  in_axes_to_vmap = [0 if isinstance(inp, NonConvexBound) else None
                     for inp in in_vals]

  kwarged_lin_fun = lambda args: primitive.bind(*args, **kwargs)
  vlin_fun = jax.vmap(kwarged_lin_fun, [in_axes_to_vmap], 0)

  bound_parents = [inp for inp in in_vals if isinstance(inp, NonConvexBound)]
  # We first merge the requirements of the inputs
  variables = {}
  previous_bounds = {}
  for parent in bound_parents:
    variables.update(parent.variables)
    previous_bounds.update(parent.previous_bounds)

  # Compute the shape of the output
  placeholder_invals = []
  for inp in in_vals:
    if isinstance(inp, NonConvexBound):
      placeholder_invals.append(jax.core.ShapedArray(inp.shape, jnp.float32))
    else:
      placeholder_invals.append(inp)
  output_shape = jax.eval_shape(kwarged_lin_fun, placeholder_invals).shape

  def eval_fn(var_set: ParamSet,
              dummy_inps: Optional[ParamSet],
              activations: Optional[ParamSet]) -> Tensor:
    """Evaluate the value of the activation in the relaxation.

    Note: This fills up `activations` by side-effect.

    Args:
      var_set: Variables for the relaxed problems.
      dummy_inps: Variables to add so that we can obtain gradients with regards
        to intermediate activations.
      activations: Cache for already computed activations.
    Returns:
      out: Tensor with the value of the activation.
    """
    inps = [eval_if_nonconvexbound(inp, var_set, dummy_inps, activations)
            for inp in in_vals]
    out = vlin_fun(inps)
    return out

  variables[index] = output_shape

  new_bound_ctor = bound_cls.get_linear_activation_constructor(
      index, vlin_fun, in_vals)

  return new_bound_ctor(index, output_shape, previous_bounds,
                        eval_fn, variables)


def _nonconvex_div(bound_cls: Type[NnCvx],
                   index: Index,
                   lhs: PrimitiveInput,
                   rhs: PrimitiveInput) -> NnCvx:
  """Propagation of NonConvex bounds bounds through Elementwise division.

  We don't support the propagation of bounds through the denominator.

  Args:
    bound_cls: Bound class to use.
    index: Unique integer identifying position in the computational graph.
    lhs: Numerator of the division.
    rhs: Denominator of the division.
  Returns:
    out_bounds: Bound on the output of the division.
  """
  if isinstance(rhs, bound_propagation.Bound):
    raise ValueError('Bound propagation through the denominator unsupported.')
  return _nonconvex_linear_op(lax.mul_p, bound_cls, index, lhs, 1. / rhs)


def _nonconvex_softplus(bound_cls: Type[NnCvx],
                        index: Index,
                        inp: NnCvx) -> NnCvx:
  """Propagation of NonConvex bounds through a Softplus.

  Args:
    bound_cls: Bound class to use.
    index: Index of the computation.
    inp: Input to the softplus
  Returns:
    out_bounds: nonconvex bounds on the operation's output
  """
  # Get the input bounds necessary for the relaxation
  inp_lb = inp.lower
  inp_ub = inp.upper

  out_lb = jax.nn.softplus(inp_lb)
  out_ub = jax.nn.softplus(inp_ub)

  slope = (out_ub - out_lb) / jnp.maximum(inp_ub - inp_lb, 1e-12)
  offset = out_lb - slope * inp_lb
  broad_slope = jnp.expand_dims(slope, 0)
  broad_offset = jnp.expand_dims(offset, 0)

  # Define the lower and upper bound functions
  lb_fun = jax.nn.softplus
  ub_fun = lambda x: broad_slope * x + broad_offset

  return _activation_convex_relaxation(
      bound_cls, index, [inp], 'Softplus', lb_fun, ub_fun,
      ibp.IntervalBound(jax.nn.softplus(inp_lb), jax.nn.softplus(inp_ub)))


def _nonconvex_relu(bound_cls: Type[NnCvx],
                    index: Index,
                    inp: NnCvx) -> NnCvx:
  """Propagation of NonConvex bounds through a max.

  Args:
    bound_cls: Bound class to use.
    index: Index of the computation.
    inp: Input to the ReLU.
  Returns:
    out_bounds: nonconvex bounds on the operation's output
  """
  # Get the input bounds necessary for the relaxation
  inp_lb = inp.lower
  inp_ub = inp.upper

  # Get the upper bound
  relu_on = (inp_lb >= 0.)
  relu_amb = jnp.logical_and(inp_lb < 0., inp_ub >= 0.)
  slope = relu_on.astype(jnp.float32)
  slope += jnp.where(relu_amb,
                     inp_ub / jnp.maximum(inp_ub - inp_lb, 1e-12),
                     jnp.zeros_like(inp_lb))
  offset = jnp.where(relu_amb, -slope * inp_lb,
                     jnp.zeros_like(inp_lb))
  broad_slope = jnp.expand_dims(slope, 0)
  broad_offset = jnp.expand_dims(offset, 0)

  # Define the lower and upper bound functions
  lb_fun = lambda x: lax.max(x, 0.)
  ub_fun = lambda x: broad_slope * x + broad_offset

  return _activation_convex_relaxation(
      bound_cls, index, [inp], 'ReLU', lb_fun, ub_fun,
      ibp.IntervalBound(lax.max(inp_lb, 0.), lax.max(inp_ub, 0.)))


def _activation_convex_relaxation(
    bound_cls: Type[NnCvx],
    index: Index,
    inputs: List[NnCvx],
    act_type: str,
    lb_fun: Callable[..., Tensor],
    ub_fun: Callable[..., Tensor],
    precomputed_bound: Optional[bound_propagation.Bound]) -> NnCvx:
  """Builds the NonConvexBound object corresponding to after non-linearities.

  Args:
    bound_cls: Bound class to use.
    index: Index of the computation.
    inputs: Inputs of the non-linearity.
    act_type: Type of activation
    lb_fun: Function to evaluate the upper bound of the activation for an input.
    ub_fun: Function to evaluate the upper bound of the activation for an input.
    precomputed_bound: Bound on the NonConvexBound to generate.
  Returns:
    out_bounds: NonConvexBound for the output of the non-linearity
  """

  bound_parents = [inp for inp in inputs if isinstance(inp, NonConvexBound)]
  # We first merge the requirements of the inputs
  variables = {}
  previous_bounds = {}
  for parent in bound_parents:
    variables.update(parent.variables)
    previous_bounds.update(parent.previous_bounds)
  inputs_lb = [jnp.expand_dims(inp.lower, 0) for inp in inputs]
  output_shape_with_target = jax.eval_shape(lb_fun, *inputs_lb).shape
  output_shape = output_shape_with_target[1:]
  variables[index] = output_shape
  def eval_fn(var_set, dummy_inps, activations):
    """Evaluate the value of the primal."""
    inp_eval = [inp.evaluate(var_set, dummy_inps, activations) for inp in
                inputs]

    lb_val = lb_fun(*inp_eval)
    ub_val = ub_fun(*inp_eval)

    theta = var_set[index]
    out_val = lb_val + theta * (ub_val - lb_val)

    return out_val

  shape = output_shape
  new_bound_ctor = bound_cls.get_nonlinearity_activation_constructor(
      index, act_type, lb_fun, ub_fun, *inputs)
  return new_bound_ctor(index, shape, previous_bounds,
                        eval_fn, variables, precomputed_bound)


def _nonconvex_dotproductattention(
    bound_cls: Type[NnCvx],
    index: Index,
    inp: NnCvx,
    weights_query: Tuple[Tensor, Tensor],
    weights_key: Tuple[Tensor, Tensor],
) -> NnCvx:
  """Builds the NonConvexBound object corresponding to dot product.

  The weights are in the form: [num_heads, emb_dim, inp_dim]
  The bias are in the form: [num_heads, emb_dim]
  Bounds on the inputs are [batch_size, num_words, inp_dim]

  When evaluating the bounds, there will be an additional target dimension, so
    Inputs should evaluate to [nb_targets, batch_size, num_words, inp_dim]
    Output is going to be [nb_targets, batch_size,
                           num_heads, num_words, num_words]

  Compute bounds on (inp @ wq + bq)' @ (inp @ wk + bk).
  Args:
    bound_cls: Bound class to use.
    index: Index of input.
    inp: Input of the non-linearity.
    weights_query: Query weights (wq, bq)
    weights_key: Key weights (wk, bk)
  Returns:
    out_bounds: NonConvexBound for the output of the non-linearity
  """
  # Get the input bounds necessary for the relaxation
  inp_lb = inp.lower
  inp_ub = inp.upper
  wq, bq = weights_query
  wk, bk = weights_key
  assert wq.shape[-1] == inp_lb.shape[-1]
  assert wk.shape[-1] == inp_lb.shape[-1]
  assert len(wq.shape) == 3
  assert len(wk.shape) == 3

  inp_dim = wq.shape[2]

  # We are going to define the function at the scalar level, and progressively
  # add batching dimensions.

  # In order to avoid materializing too large of a matrix, we are going to
  # iterate over the "feature-to-feature" matching of the inputs which
  # gets summed over. We use a scan loop to accumulate the contribution,
  # assuming that we're computing it *from one word to another word*

  # Here, attention matrix is [inp_dim, inp_dim],
  # and all others inputs are vector of size [inp_dim]
  # Output is a tuple of scalars
  def word_pair_attention(attention_matrix, lb_a, ub_a, x_a, lb_b, ub_b, x_b):

    def accumulate_attention_by_inputfeat(carry, cross_inp_indx):
      feat_x, feat_y = jnp.divmod(cross_inp_indx, inp_dim)
      att_mat_elt = attention_matrix[feat_x, feat_y]
      pos_att_mat_elt = jnp.maximum(att_mat_elt, 0.)
      neg_att_mat_elt = jnp.minimum(att_mat_elt, 0.)

      x_a_feat = x_a[feat_x]
      lb_a_feat = lb_a[feat_x]
      ub_a_feat = ub_a[feat_x]

      x_b_feat = x_b[feat_y]
      lb_b_feat = lb_b[feat_y]
      ub_b_feat = ub_b[feat_y]

      (mc_l, mc_u) = mccormick.mccormick_outer_product(
          x_a_feat, x_b_feat,
          lb_a_feat, ub_a_feat,
          lb_b_feat, ub_b_feat)
      mc_l = mc_l.squeeze()
      mc_u = mc_u.squeeze()
      (carry_l, carry_u) = carry

      new_carry = (carry_l + pos_att_mat_elt * mc_l + neg_att_mat_elt * mc_u,
                   carry_u + neg_att_mat_elt * mc_l + pos_att_mat_elt * mc_u)
      return new_carry, None

    inp_cross_indxs = jnp.arange(inp_dim**2)
    ini_attention = (jnp.array(0.), jnp.array(0.))
    attention, _ = jax.lax.scan(accumulate_attention_by_inputfeat,
                                ini_attention, inp_cross_indxs)
    return attention

  # We are now going to vmap over the words dimension for
  #  - the inputs on the left (lb_a, ub_a, x_a)
  #  - the inputs on the right (lb_b, ub_b, x_b)
  # Attention_matrix is not batched over, and each of the inputs ignore the
  # batching of the other input.
  # You can think of these two batching as:
  #   * First, going from a function computing the attention score for a pair
  #     of words to a function computing attention scores from one word to all
  #     other words.
  #   * Second, from that function to a function computing attention score from
  #     all words to all other words.
  # The args are:
  #   - attention_matrix is [inp_dim, inp_dim]
  #   - all other inputs are now [nb_words, inp_dim]
  # Outputs is a tuple of shape (nb_words, nb_words)
  all_words_attention = jax.vmap(jax.vmap(word_pair_attention,
                                          in_axes=(None,) + (0,)*3 + (None,)*3,
                                          out_axes=0),
                                 in_axes=(None,) + (None,)*3 + (0,)*3,
                                 out_axes=0)

  # We are now going to compute the actual attention score, not just the
  # quadratic part.
  # Note that we use the previously defined function duplicating the arguments.
  # We also add the linear and constant terms.
  # All computations are done for a single head.
  #   wq, wk are [emb_dim, inp_dim]
  #   bq, bq are [emb_dim]
  #   x_lb, x_ub, x are [nb_words, inp_dim]
  # Outputs is a tuple of shape (nb_words, nb_words)
  def per_head_attention(wq, bq, wk, bk, x_lb, x_ub, x):
    attention_matrix = jnp.dot(wq.T, wk)
    # attention_matrix is [inp_dim, inp_dim]

    wq_bk = jnp.dot(wq.T, bk)
    wk_bq = jnp.dot(wk.T, bq)
    # These are [inp_dim] vectors

    bk_bq = jnp.dot(bq.T, bk)
    # bk_bq is a scalar.

    quad_term_lower, quad_term_upper = all_words_attention(attention_matrix,
                                                           x_lb, x_ub, x,
                                                           x_lb, x_ub, x)

    # Each of the component of the linear term is a [nb_words] vector.
    lin_term = (jnp.expand_dims(jnp.dot(x, wq_bk), 0)
                + jnp.expand_dims(jnp.dot(x, wk_bq), 1))
    # This is a scalar.
    constant_term = bk_bq

    return (quad_term_lower + lin_term + constant_term,
            quad_term_upper + lin_term + constant_term)

  # We add a batching dimension that corresponds to each head of the self
  # attention for all the weights and biases.
  #   wq, wk are [num_heads, emb_dim, inp_dim]
  #   bq, bk are [num_heads, emb_dim]
  #   x_lb, x_ub, x are [nb_words, inp_dim]
  # Outputs is a tuple of shape (num_heads, nb_words, nb_words)
  all_words_all_heads_attention = jax.vmap(per_head_attention,
                                           in_axes=(0,)*4 + (None, None, None),
                                           out_axes=0)

  # We add the batch dimension over the samples, to x and the bounds.
  #   wq, wk are [num_heads, emb_dim, inp_dim]
  #   bq, bk are [num_heads, emb_dim]
  #   x_lb, x_ub, x are [batch_idx, nb_words, inp_dim]
  # Outputs is a tuple of shape (batch_idx, num_heads, nb_words, nb_words)
  batched_awah_attention = jax.vmap(all_words_all_heads_attention,
                                    in_axes=(None,)*4 + (0,)*3,
                                    out_axes=0)

  # We add a target dimension for the different optimization targets, only on x.
  #   wq, wk are [num_heads, emb_dim, inp_dim]
  #   bq, bk are [num_heads, emb_dim]
  #   x_lb, x_ub are [batch_idx, nb_words, inp_dim]
  #   x is [target_idx, batch_idx, nb_words, inp_dim]
  # Outputs is a tuple of shape (target_idx, batch_idx,
  #                              num_heads, nb_words, nb_words)
  batched_per_target_awah_attention = jax.vmap(batched_awah_attention,
                                               in_axes=(None,)*6 + (0,),
                                               out_axes=0)

  # We fold in all the arguments that are parameters or input bounds, so as to
  # just get the function evaluating the bounds.
  folded_arguments = functools.partial(batched_per_target_awah_attention,
                                       wq, bq, wk, bk, inp_lb, inp_ub)

  lb_fun = lambda x: jnp.transpose(folded_arguments(x)[0], (0, 1, 2, 4, 3))
  ub_fun = lambda x: jnp.transpose(folded_arguments(x)[1], (0, 1, 2, 4, 3))

  return _activation_convex_relaxation(
      bound_cls, index, [inp], 'DotProductSA', lb_fun, ub_fun,
      None)


def _nonconvex_posbilinear(bound_cls, index, inp_a, inp_b, **params):
  """Builds the NonConvexBound object corresponding to dot product.

  Compute bounds on inp_a @ inp_b
  Args:
    bound_cls: Bound class to use.
    index: Index of output.
    inp_a: Argument of matrix product
    inp_b: Argument of matrix product
    **params: Parameters of the matrix product.
  Returns:
    out_bounds: NonConvexBound for the output of the non-linearity
  """
  lb_fun, ub_fun = activation_relaxation.posbilinear_relaxation(
      inp_a.lower, inp_a.upper, inp_b.lower, inp_b.upper,
      **params)

  lb_fun = jax.vmap(lb_fun, in_axes=0, out_axes=0)
  ub_fun = jax.vmap(ub_fun, in_axes=0, out_axes=0)

  return _activation_convex_relaxation(
      bound_cls, index, [inp_a, inp_b], 'BilinearValues', lb_fun, ub_fun,
      None)


_linear_op_primitives = (
    bound_propagation.AFFINE_PRIMITIVES +
    bound_propagation.RESHAPE_PRIMITIVES
)
_nonconvex_primitive_transform = {
    primitive: functools.partial(_nonconvex_linear_op, primitive)
    for primitive in _linear_op_primitives}
_nonconvex_primitive_transform.update({
    lax.div_p: _nonconvex_div,
    synthetic_primitives.relu_p: _nonconvex_relu,
    synthetic_primitives.softplus_p: _nonconvex_softplus,
    synthetic_primitives.posbilinear_p: _nonconvex_posbilinear,
})


def build_nonconvex_formulation(
    bound_cls: Type[NnCvx],
    concretizer_ctor: Callable[[], Concretizer],
    function: Callable[..., Nest[Tensor]],
    *bounds: List[graph_traversal.GraphInput],
    graph_simplifier=synthetic_primitives.default_simplifier
    ) -> Nest[NnCvx]:
  """Builds the optimizable objective.

  Args:
    bound_cls: Bound class to use. This determines what dual formulation will
      be computed.
    concretizer_ctor: Constructor for the concretizer to use to obtain
      intermediate bounds.
    function: Function performing computation to obtain bounds for. Takes as
      only arguments the network inputs.
    *bounds: Bounds on the inputs of the function.
    graph_simplifier: What graph simplifier to use.
  Returns:
    output_bounds: NonConvex bounds that can be optimized with a solver.
  """
  concretizer = concretizer_ctor()
  bound_transform = NonConvexTransform(bound_cls, concretizer)
  output_bounds, _ = bound_propagation.bound_propagation(
      bound_propagation.ForwardPropagationAlgorithm(bound_transform),
      function, *bounds, graph_simplifier=graph_simplifier)
  # Always concretize the returned bounds so that `lower` and `upper` are
  # accessible
  for out_bound in jax.tree_util.tree_leaves(output_bounds):
    if out_bound.requires_concretizing(None):
      out_bound.concretize(concretizer)

  return output_bounds


class NonConvexTransform(Generic[NnCvx],
                         bound_propagation.GraphTransform[NnCvx]):
  """Graph Transform to build a NonConvex Relaxation, which can be optimized."""

  def __init__(self,
               bound_cls: Type[NnCvx],
               concretizer: Concretizer):
    self._bound_cls = bound_cls
    self._concretizer = concretizer

  def input_transform(
      self,
      context: TransformContext,
      lower_bound: Tensor,
      upper_bound: Tensor,
  ) -> NnCvx:
    self._concretizer.accept_input(context, lower_bound, upper_bound)
    return self._bound_cls.initial_nonconvex_bound(
        context.index, lower_bound, upper_bound)

  def primitive_transform(
      self,
      context: TransformContext,
      primitive: bound_propagation.Primitive,
      *args: PrimitiveInput,
      **params,
  ) -> NnCvx:
    self._concretizer.accept_primitive(context, primitive, *args, **params)
    for arg in args:
      if (isinstance(arg, NonConvexBound) and
          arg.requires_concretizing(primitive)):
        arg.concretize(self._concretizer)
    params = utils.filter_jaxverify_kwargs(params)
    new_bound = _nonconvex_primitive_transform[primitive](
        self._bound_cls, context.index, *args, **params)
    return new_bound


class ConstrainedNonConvexTransform(
    NonConvexTransform[ConstrainedNonConvexBound]):
  """Graph Transform performing parallel boundprop and imposing bounds."""

  def __init__(self,
               bound_cls: Type[ConstrainedNonConvexBound],
               concretizer: Concretizer,
               imposed_boundprop: bound_propagation.BoundTransform):
    super().__init__(bound_cls, concretizer)
    self._imposed_concretizer = BaseBoundConcretizer(imposed_boundprop)

  def _impose_bounds(self, unc_bound: ConstrainedNonConvexBound):
    """Set imposed bounds, as obtained from the imposed_boundprop."""
    bounds_to_impose = self._imposed_concretizer.get_bounds(unc_bound)
    unc_bound.set_imposed_bounds(bounds_to_impose)

  def input_transform(
      self,
      context: TransformContext,
      lower_bound: Tensor,
      upper_bound: Tensor,
  ) -> ConstrainedNonConvexBound:
    self._imposed_concretizer.accept_input(context, lower_bound, upper_bound)
    bound = super().input_transform(context, lower_bound, upper_bound)
    self._impose_bounds(bound)
    return bound

  def primitive_transform(
      self,
      context: TransformContext,
      primitive: bound_propagation.Primitive,
      *args: PrimitiveInput,
      **params,
  ) -> ConstrainedNonConvexBound:
    self._imposed_concretizer.accept_primitive(context, primitive,
                                               *args, **params)
    bound = super().primitive_transform(context, primitive, *args, **params)
    self._impose_bounds(bound)
    return bound
