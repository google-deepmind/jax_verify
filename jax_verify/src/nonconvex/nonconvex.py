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
from typing import Callable, Generic, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union
from absl import logging

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import intersection
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils
from jax_verify.src.types import Index, Nest, Primitive, Tensor, TensorFun  # pylint: disable=g-multiple-import

NnCvx = TypeVar('NnCvx', bound='NonConvexBound')
# Mapping of a position in the computation to a set of parameters.
# These can be variables, gradients, or coefficients.
ParamSet = MutableMapping[Index, Tensor]


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
  - concretizer is an instance of `Concretizer`, that exposes a `concrete_bound`
    method. This is used during the bound propagation when concrete bounds are
    required (such as when defining the convex hull relaxation of a ReLU.).
    The choice of concretizer impacts of those intermediate concrete bounds are
    computed. This can be done by relying on a fallback method such as IBP, or
    by optimization.

  In addition, the subclass are responsible for encoding some additional
  mechanisms required for the computation of the dual they implement.
  """

  def __init__(
      self,
      index: Index,
      shape: Tuple[int, ...],
      previous_bounds: MutableMapping[Index, 'NonConvexBound'],
      eval_fn: Callable[[ParamSet, ParamSet, ParamSet], Tensor],
      variables: Mapping[Index, Tuple[int, ...]],
      concretized_bounds: Optional[bound_propagation.Bound] = None):
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
  def dtype(self):
    return jnp.float32

  @property
  def lower(self) -> Tensor:
    if self._concretized_bounds is None:
      logging.warning('.lower called on a non-concretized bound.'
                      'Returning spurious bounds.')
      return -float('inf') * jnp.ones(self.shape, self.dtype)
    return self._concretized_bounds.lower

  @property
  def upper(self) -> Tensor:
    if self._concretized_bounds is None:
      logging.warning('.upper called on a non-concretized bound.'
                      'Returning spurious bounds.')
      return float('inf') * jnp.ones(self.shape, self.dtype)
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
      act_type: Primitive,
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

  def concretize(self, concretizer: 'Concretizer', graph, env):
    self._concretized_bounds = concretizer.concrete_bound(graph, env, self)

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

    return bound_ctor(
        index, shape, previous_bounds, eval_fn, variables,
        bound_propagation.IntervalBound(lower_bound, upper_bound))


class ConstrainedNonConvexBound(NonConvexBound, metaclass=abc.ABCMeta):
  """This special case of NonConvexBound supports `imposed_bounds` constraints.

  The assumption is that, before any evaluation of the primal or dual, the
  `set_imposed_bounds` function is called. As a result, these should be
  created through a `_ConstrainedNonConvexTransform`

  * `lower` and `upper` return those by default before being concretized.
  * Concretizing will result in the concretized bounds being the tightest
    between the bounds obtained by concretizing and the imposed ones.
  * Imposing bounds will also constrain any existing concretized bounds.
  * Evaluating an activation represented by this bound will return the
    evaluation projected into the admissible bounds.
  """

  def __init__(
      self,
      index: Index,
      shape: Tuple[int, ...],
      previous_bounds: MutableMapping[Index, 'ConstrainedNonConvexBound'],
      eval_fn: Callable[[ParamSet, ParamSet, ParamSet], Tensor],
      variables: Mapping[Index, Tuple[int, ...]],
      concretized_bounds: Optional[bound_propagation.Bound] = None):
    super().__init__(index, shape, previous_bounds,
                     eval_fn, variables, concretized_bounds)
    self._imposed_bounds = None

  def is_constrained(self) -> bool:
    return self._imposed_bounds is not None

  def imposed_bounds(self) -> bound_propagation.Bound:
    if self._imposed_bounds is None:
      raise ValueError('No imposed bounds')
    return self._imposed_bounds

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

  def concretize(self, concretizer: 'Concretizer', graph, env):
    super().concretize(concretizer, graph, env)
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
  def concrete_bound(
      self,
      graph: graph_traversal.PropagationGraph,
      env: Mapping[jax.core.Var, bound_propagation.LayerInput],
      nonconvex_bound: NonConvexBound) -> bound_propagation.Bound:
    """Returns a concretized bound."""


class BaseBoundConcretizer(Concretizer):
  """Concretizer based on performing a parallel propagation with another method.

  This should be constructed with an environment resulting from forward
  propagation of another bound propagation method.
  """

  def concrete_bound(
      self,
      graph: graph_traversal.PropagationGraph,
      env: Mapping[jax.core.Var, bound_propagation.LayerInput],
      nonconvex_bound: NonConvexBound) -> bound_propagation.Bound:
    return env[graph.jaxpr_node(nonconvex_bound.index)]


def eval_if_nonconvexbound(
    inp: graph_traversal.LayerInput[NonConvexBound],
    var_set: ParamSet,
    dummy_inps: Optional[ParamSet],
    activations: Optional[ParamSet]) -> Tensor:
  if isinstance(inp, NonConvexBound):
    return inp.evaluate(var_set, dummy_inps, activations)
  else:
    return inp


def _nonconvex_linear_op(
    primitive: Primitive,
    bound_cls: Type[NnCvx],
    index: Index,
    *in_vals: graph_traversal.LayerInput[NonConvexBound],
    **kwargs) -> NnCvx:
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


def _nonconvex_div(
    bound_cls: Type[NnCvx],
    index: Index,
    lhs: graph_traversal.LayerInput[NonConvexBound],
    rhs: graph_traversal.LayerInput[NonConvexBound],
) -> NnCvx:
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


def _activation_convex_relaxation(
    bound_cls: Type[NnCvx],
    index: Index,
    inputs: Sequence[NnCvx],
    act_type: Primitive,
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


def _nonconvex_activation(
    act_type: Primitive,
    relaxation: Callable[..., Tuple[TensorFun, TensorFun]],
    bound_cls: Type[NnCvx],
    index: Index,
    *inps: Union[NnCvx, Tensor],
    eltwise_increasing: bool = False,
    **params) -> NnCvx:
  """Propagation of NonConvexBounds through a non-linear operation.

  Args:
    act_type: Activation type, e.g. 'Softplus' or 'ReLU'.
    relaxation: Function accepting (*inps, **params) and returning
      convex lower bound and concave upper bound functions.
    bound_cls: Bound class to use.
    index: Node index.
    *inps: Nonconvex bounds on the inputs to the operation.
    eltwise_increasing: Whether the operation is known to be element-wise
      monotonic increasing, in which case we can pre-compute its bounds.
    **params: Parameters of the operation.
  Returns:
    out_bounds: Nonconvex bounds on the operation's output.
  """
  lb_fun, ub_fun = relaxation(*inps, **params)

  if eltwise_increasing:
    # The function is assumed to be monotonically increasing, so we
    # can readily pre-compute interval bounds on its output.
    # `lb_fun` will be the original function whenever this is reached.
    precomputed_bound = bound_propagation.IntervalBound(
        lb_fun(*[inp.lower for inp in inps]),  # pytype: disable=attribute-error  # jax-ndarray
        lb_fun(*[inp.upper for inp in inps]))  # pytype: disable=attribute-error  # jax-ndarray
  else:
    precomputed_bound = None

  lb_fun = jax.vmap(lb_fun, in_axes=0, out_axes=0)
  ub_fun = jax.vmap(ub_fun, in_axes=0, out_axes=0)

  return _activation_convex_relaxation(
      bound_cls, index,
      [inp for inp in inps if isinstance(inp, NonConvexBound)],
      act_type, lb_fun, ub_fun, precomputed_bound)


def _make_activation_primitive_transform(
    primitive: Primitive,
    activation: activation_relaxation.ActivationRelaxation,
) -> Callable[..., NonConvexBound]:
  return functools.partial(
      _nonconvex_activation,
      primitive, activation.relaxation_fn,
      eltwise_increasing=activation.eltwise_increasing)


_linear_op_primitives: Sequence[Primitive] = [
    *bound_propagation.AFFINE_PRIMITIVES,
    *bound_propagation.RESHAPE_PRIMITIVES,
]
_nonconvex_primitive_transform: Mapping[
    Primitive, Callable[..., NonConvexBound],
] = {
    **{primitive: functools.partial(_nonconvex_linear_op, primitive)
       for primitive in _linear_op_primitives},
    lax.div_p: _nonconvex_div,
    **{primitive: _make_activation_primitive_transform(primitive, act)
       for primitive, act in activation_relaxation.relaxation_fns.items()},
}


class _NonConvexTransform(
    Generic[NnCvx], graph_traversal.GraphTransform[NnCvx]):
  """Graph Transform to build a NonConvex Relaxation, which can be optimized."""

  def __init__(
      self,
      bound_cls: Type[NnCvx],
      concretizer: Concretizer,
      graph: graph_traversal.PropagationGraph,
      env: Mapping[jax.core.Var, bound_propagation.LayerInput],
  ):
    self._bound_cls = bound_cls
    self._concretizer = concretizer
    self._graph = graph
    self._env = env

  def input_transform(
      self,
      context: graph_traversal.TransformContext[NnCvx],
      input_bound: graph_traversal.InputBound,
  ) -> NnCvx:
    return self._bound_cls.initial_nonconvex_bound(
        context.index, input_bound.lower, input_bound.upper)

  def primitive_transform(
      self,
      context: graph_traversal.TransformContext[NnCvx],
      primitive: Primitive,
      *args: graph_traversal.LayerInput[NnCvx],
      **params,
  ) -> NnCvx:
    for arg in args:
      if (isinstance(arg, NonConvexBound) and
          arg.requires_concretizing(primitive)):
        arg.concretize(self._concretizer, self._graph, self._env)
    params = synthetic_primitives.filter_jaxverify_kwargs(params)
    new_bound = _nonconvex_primitive_transform[primitive](
        self._bound_cls, context.index, *args, **params)
    return new_bound


class _ConstrainedNonConvexTransform(
    _NonConvexTransform[ConstrainedNonConvexBound]):
  """Graph Transform performing parallel boundprop and imposing bounds."""

  def __init__(
      self,
      bound_cls: Type[ConstrainedNonConvexBound],
      imposed_boundprop: bound_propagation.BoundTransform,
      concretizer: Concretizer,
      graph: graph_traversal.PropagationGraph,
      env: Mapping[jax.core.Var, bound_propagation.LayerInput],
  ):
    super().__init__(bound_cls, concretizer, graph, env)
    self._imposed_boundprop = imposed_boundprop

  def input_transform(
      self,
      context: graph_traversal.TransformContext[ConstrainedNonConvexBound],
      input_bound: graph_traversal.InputBound,
  ) -> ConstrainedNonConvexBound:
    bound = super().input_transform(context, input_bound)
    bound.set_imposed_bounds(self._imposed_boundprop.input_transform(
        context, input_bound))
    return bound

  def primitive_transform(
      self,
      context: graph_traversal.TransformContext[ConstrainedNonConvexBound],
      primitive: Primitive,
      *args: graph_traversal.LayerInput[ConstrainedNonConvexBound],
      **params,
  ) -> ConstrainedNonConvexBound:
    bound = super().primitive_transform(context, primitive, *args, **params)
    imposed_bound_args = [
        arg.imposed_bounds()
        if isinstance(arg, bound_propagation.Bound) else arg
        for arg in args]
    bound.set_imposed_bounds(self._imposed_boundprop.equation_transform(
        context, primitive, *imposed_bound_args, **params)[0])
    return bound


class NonConvexAlgorithm(
    Generic[NnCvx], bound_propagation.PropagationAlgorithm[NnCvx]):
  """Forward algorithm with an optional initial pass for 'base' bounds."""

  def __init__(
      self,
      nonconvex_transform_ctor: Callable[..., bound_propagation.BoundTransform],
      concretizer: Concretizer,
      base_boundprop: Optional[bound_propagation.BoundTransform] = None):
    super().__init__()
    self._nonconvex_transform_ctor = nonconvex_transform_ctor
    self._concretizer = concretizer
    self._base_boundprop = base_boundprop

  def propagate(
      self,
      graph: graph_traversal.PropagationGraph,
      bounds: Nest[graph_traversal.GraphInput],
  ) -> Tuple[
      Nest[bound_propagation.Bound],
      Mapping[jax.core.Var, bound_propagation.LayerInput],
  ]:
    if self._base_boundprop is not None:
      # Propagate the 'base' bounds in advance, for subsequent use by
      # the concretiser.
      _, base_env = bound_propagation.ForwardPropagationAlgorithm(
          self._base_boundprop).propagate(graph, bounds)
    else:
      # No 'base' boundprop method specified.
      # This is fine as long as the concretiser does not rely on base bounds.
      base_env = None

    nonconvex_transform = self._nonconvex_transform_ctor(
        self._concretizer, graph, base_env)
    output_bounds, env = bound_propagation.ForwardPropagationAlgorithm(
        nonconvex_transform).propagate(graph, bounds)

    # Always concretize the returned bounds so that `lower` and `upper` are
    # accessible
    for out_bound in jax.tree_util.tree_leaves(output_bounds):
      if out_bound.requires_concretizing(None):
        out_bound.concretize(self._concretizer, graph, base_env)

    return output_bounds, env


def nonconvex_algorithm(
    bound_cls: Type[NnCvx],
    concretizer: Concretizer,
    *,
    base_boundprop: Optional[bound_propagation.BoundTransform] = None,
    imposed_boundprop: Optional[bound_propagation.BoundTransform] = None,
) -> bound_propagation.PropagationAlgorithm[NnCvx]:
  """Builds a bound propagation algorithm for the non-convex formulation.

  Args:
    bound_cls: Bound class to use. This determines what dual formulation will
      be computed.
    concretizer: Concretizer to use to obtain intermediate bounds.
    base_boundprop: Underlying bound propagation method for obtaining concrete
      bounds.
    imposed_boundprop: Additional bounds to apply as constraints, e.g. from
      a branching decision.
  Returns:
    Propagation algorithm.
  """
  if imposed_boundprop is None:
    bound_transform_ctor = functools.partial(_NonConvexTransform, bound_cls)
  else:
    bound_transform_ctor = functools.partial(
        _ConstrainedNonConvexTransform, bound_cls, imposed_boundprop)
  return NonConvexAlgorithm(bound_transform_ctor, concretizer, base_boundprop)
