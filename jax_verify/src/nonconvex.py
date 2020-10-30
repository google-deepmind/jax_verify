# coding=utf-8
# Copyright 2020 The jax_verify Authors.
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

import jax
from jax import lax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
from jax_verify.src import utils

Tensor = jnp.ndarray


def _sum_fn(fn, *args, **kwargs):
  out = fn(*args, **kwargs)
  summand = out[0] if isinstance(out, tuple) else out
  return summand.sum(), out


def _sum_over_acts(var):
  return var.sum(axis=tuple(range(2, var.ndim)))


class NonConvexBound(bound_propagation.Bound, metaclass=abc.ABCMeta):
  """Represent a bound that can be optimized.

  This is the object that gets propagated through the network.

  The important elements it exposes are the following:

  - variables: Specifies what is the shape of the parameters that need to be
      provided such that a bound can be computed.
  - dual: A function that takes as input variables as specified by `variables`
      and a set of linear function over the activations, in the form of a
      (batch_dim , nb_opt, *act_dims) tensor, and that returns the value of the
      primal objectives when those variables are used, as well as a dual bound
      on those linear objectives. Different options for dual computation are
      available, each implementd in a separate sub-class.
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

  def __init__(self, index, shape, previous_bounds,
               eval_fn, variables, concretizer):
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
      concretizer: Instance of Concretizer, used to obtain bounds over the
        activation this object represents.

    """
    self.index = index
    self.shape = shape
    self.previous_bounds = previous_bounds
    self.previous_bounds[index] = self
    self._eval_fn = eval_fn
    self.concretizer = concretizer
    self.variables = variables
    self._concretized_bounds = None

    def primal(var_set, objectives, dummy_inps=None):
      """Evaluate the primal objective of the problem.

      dummy_inps are inputs which are always zeros, and that we will add to
      every intermediate activations of the network for which we need gradients.
      This way, computing the gradients with regards to those "inputs" allows
      us to compute gradients with regards to intermediate activations, as
      required for the definition of the dual variables.

      Args:
        var_set: Dictionary mapping the position in network to a tensor
          containing the primal variables.
        objectives: Coefficient of the objective function over the last layer
          of activations.
        dummy_inps: Dictionary mapping the position in network to a zero
          tensor.
      Returns:
         primals: All primal objectives.
      """
      acts = {}
      final_acts = self.evaluate(var_set, dummy_inps, acts)
      primals = _sum_over_acts(final_acts * objectives)
      return primals, acts
    self.primal_fn = primal
    self.primal_sumfn = functools.partial(_sum_fn, primal)

  @property
  def lower(self):
    if self._concretized_bounds is None:
      self.concretize()
    return self._concretized_bounds.lower

  @property
  def upper(self):
    if self._concretized_bounds is None:
      self.concretize()
    return self._concretized_bounds.upper

  def evaluate(self, var_set, dummy_inps=None, acts=None):
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
  def dual(self, var_set, objectives):
    """Compute the dual, using dual variables derived from primals in var_set.

    Returns both primal and dual, so that this can be used to compute dual gap.

    Args:
      var_set: Relaxation variables to use to compute the primal activations and
        derive the duals from.
      objectives: Coefficient of the objective function over the last layer
          of activations.
    """

  @classmethod
  @abc.abstractmethod
  def get_initial_bound_constructor(cls, index, lb, ub):
    """Class specific initialization for the input bounds of the network."""
    raise NotImplementedError('Initial bound constructor not implemented')

  @classmethod
  @abc.abstractmethod
  def get_linear_activation_constructor(cls, index, vlin_fun, in_vals):
    """Class specific initialization for the output of a linear function."""
    raise NotImplementedError('Linear activation constructor not implemented')

  @classmethod
  @abc.abstractmethod
  def get_nonlinearity_activation_constructor(cls, index, inp, act_type,
                                              lb_fun, ub_fun):
    """Class specific initialization for the output of a non-linearity."""
    raise NotImplementedError('Nonlinearity activation constructor not'
                              'implemented')

  def _compute_derivatives_dualvars(self, var_set, objectives):
    # Compute the gradients of all the primals (done by taking their sum),
    # with regards to the dummy inputs. This is what is giving us our dual
    # variables.
    grad_fun = jax.grad(self.primal_sumfn, argnums=2, has_aux=True)
    dummy_acts = {key: 0*val for key, val in var_set.items()}
    dual_vars, (_, acts) = grad_fun(var_set, objectives, dummy_acts)
    return dual_vars, acts

  def concretize(self):
    if self._concretized_bounds is None:
      self._concretized_bounds = self.concretizer.get_bounds(self)
    return self._concretized_bounds

  @classmethod
  def initial_nonconvex_bound(cls, concretizer_ctor, index,
                              lower_bound, upper_bound):
    shape = lower_bound.shape
    variables = {index: lower_bound.shape}

    lb = jnp.expand_dims(lower_bound, axis=1)
    ub = jnp.expand_dims(upper_bound, axis=1)

    concretizer = concretizer_ctor()
    concretizer.initialize(index, lower_bound, upper_bound)

    previous_bounds = {}

    def eval_fn(var_set, *_):
      val = lb + (ub - lb) * var_set[index]
      return val

    bound_ctor = cls.get_initial_bound_constructor(index, lb, ub)

    return bound_ctor(index, shape, previous_bounds,
                      eval_fn, variables, concretizer)


class Concretizer(abc.ABC):
  """Abstract class to define the API of concretizer.

  The role of Concretizer is to give access to concrete bounds to define
  relaxation while propagating NonConvexBound which are solver based.
  """

  @abc.abstractmethod
  def initialize(self, index, lower_bound, upper_bound):
    """Update the concretizer based on the initial input bound."""
    pass

  @abc.abstractmethod
  def propagate_concretizer(self, primitive, index, *in_vals, **params):
    """Return a concretizer for the resulting bound."""
    pass

  @abc.abstractmethod
  def get_bounds(self, nonconvex_bound):
    """Return a concretized bound."""
    pass


class OptimizingConcretizer(Concretizer):
  """Concretizer based on optimizing the intermediate bounds.

  This needs to be initialized with an optimizer, and concrete bounds will be
  obtained by solving the relaxation for the intermediate activations.
  """

  def __init__(self, optimizer):
    self._optimizer = optimizer

  def initialize(self, index, lower_bound, upper_bound):
    # There is no need to update anything.
    pass

  def propagate_concretizer(self, *_args, **_kwargs):
    # For now, the optimizer is not dependent on what propagation has been
    # achieved, so just reuse the same.
    # Potentially, in the future, we could handle adapting the hyperparameters.
    return self

  def get_bounds(self, nonconvex_bound):
    # TODO: Improve this so that if the bounds can be deduced from a
    # previous one, we don't perform full optimization.
    return self._optimizer.optimize(nonconvex_bound)


class BaseBoundConcretizer(Concretizer):
  """Concretizer based on performing a parallel propagation with another method.

  This should be initialized with the `input_transform` and the
  `primitive_transform` that are usually passed to `bound_propagation`.
  """

  def __init__(self, bound_transform):
    self._bound_transform = bound_transform
    self.base_bound = None

  def initialize(self, index, lower_bound, upper_bound):
    self.base_bound = self._bound_transform.input_transform(
        index, lower_bound, upper_bound)

  def propagate_concretizer(self, primitive, index, *in_vals, **params):
    new_concretizer = BaseBoundConcretizer(self._bound_transform)
    base_in_vals = [
        inp.concretizer.base_bound if isinstance(inp, NonConvexBound) else inp
        for inp in in_vals]
    new_base_bound = self._bound_transform.primitive_transform(
        index, primitive, *base_in_vals, **params)
    new_concretizer.base_bound = new_base_bound
    return new_concretizer

  def get_bounds(self, _):
    return self.base_bound


def eval_if_nonconvexbound(inp, var_set, dummy_inps, activations):
  if isinstance(inp, NonConvexBound):
    return inp.evaluate(var_set, dummy_inps, activations)
  else:
    return inp


def _nonconvex_linear_op(primitive, lin_fun, bound_cls, index,
                         *in_vals, **kwargs):
  """Propagation of NonConvex bounds through a linear operation.

  Args:
    primitive: Primitive that this linear operation implement.
    lin_fun: Linear function to use to implement the primitive.
    bound_cls: Bound class to use.
    index: Unique integer identifying position
    *in_vals: Input of the bound propagation in the forward pass
    **kwargs: Dict with the parameters of the linear operation
  Returns:
    out_bounds: NonConvexBound
  """
  in_axes_to_vmap = [1 if isinstance(inp, NonConvexBound) else None
                     for inp in in_vals]

  kwarged_lin_fun = lambda x: lin_fun(*x, **kwargs)
  vlin_fun = jax.vmap(kwarged_lin_fun, [in_axes_to_vmap], 1)

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

  def eval_fn(var_set, dummy_inps, activations):
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

  # Build a new concretizer out of the one of one of the input.
  new_concretizer = bound_parents[0].concretizer.propagate_concretizer(
      primitive, index, *in_vals, **kwargs)

  variables[index] = output_shape

  new_bound_ctor = bound_cls.get_linear_activation_constructor(
      index, vlin_fun, in_vals)

  return new_bound_ctor(index, output_shape, previous_bounds,
                        eval_fn, variables, new_concretizer)


def _nonconvex_div(bound_cls, index, lhs, rhs):
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
  return _nonconvex_linear_op(lax.div_p, lambda x, y: x / y, bound_cls,
                              index, lhs, rhs)


def _nonconvex_softplus(bound_cls, index, inp):
  """Propagation of NonConvex bounds through a Softplus.

  Args:
    bound_cls: Bound class to use.
    index: Index of the computation.
    inp: Input to the softplus
  Returns:
    out_bounds: NonConvexBound
  """
  new_concretizer = inp.concretizer.propagate_concretizer(
      synthetic_primitives.softplus_p, index, inp)

  # Get the input bounds necessary for the relaxation
  inp_lb = inp.lower
  inp_ub = inp.upper

  out_lb = jax.nn.softplus(inp_lb)
  out_ub = jax.nn.softplus(inp_ub)

  slope = (out_ub - out_lb) / jnp.maximum(inp_ub - inp_lb, 1e-12)
  offset = out_lb - slope * inp_lb
  broad_slope = jnp.expand_dims(slope, 1)
  broad_offset = jnp.expand_dims(offset, 1)

  # Define the lower and upper bound functions
  lb_fun = jax.nn.softplus
  ub_fun = lambda x: broad_slope * x + broad_offset

  return _activation_convex_relaxation(
      bound_cls, index, inp, new_concretizer, 'Softplus',
      lb_fun, ub_fun)


def _nonconvex_max(bound_cls, index, lhs, rhs):
  """Propagation of NonConvex bounds through a max.

  Args:
    bound_cls: Bound class to use.
    index: Index of the computation.
    lhs: First input to the max, assumed to be a ReLU input
    rhs: Second input to the max, assumed to be 0
  Returns:
    out_bounds: NonConvexBound
  """
  if not (isinstance(lhs, NonConvexBound) and rhs == 0.):
    raise NotImplementedError('Only ReLU is implemented for now.')

  new_concretizer = lhs.concretizer.propagate_concretizer(
      lax.max_p, index, lhs, rhs)

  # Get the input bounds necessary for the relaxation
  inp_lb = lhs.lower
  inp_ub = lhs.upper

  # Get the upper bound
  relu_on = (inp_lb >= 0.)
  relu_amb = jnp.logical_and(inp_lb < 0., inp_ub >= 0.)
  slope = relu_on.astype(jnp.float32)
  slope += jnp.where(relu_amb,
                     inp_ub / jnp.maximum(inp_ub - inp_lb, 1e-12),
                     jnp.zeros_like(inp_lb))
  offset = jnp.where(relu_amb, -slope * inp_lb,
                     jnp.zeros_like(inp_lb))
  broad_slope = jnp.expand_dims(slope, 1)
  broad_offset = jnp.expand_dims(offset, 1)

  # Define the lower and upper bound functions
  lb_fun = lambda x: lax.max(x, 0.)
  ub_fun = lambda x: broad_slope * x + broad_offset

  return _activation_convex_relaxation(
      bound_cls, index, lhs, new_concretizer,
      'ReLU', lb_fun, ub_fun)


def _activation_convex_relaxation(bound_cls, index, inp, concretizer, act_type,
                                  lb_fun, ub_fun):
  """Builds the NonConvexBound object corresponding to after non-linearities.

  Args:
    bound_cls: Bound class to use.
    index: Index of the computation.
    inp: Input of the non-linearity.
    concretizer: Concretizer for the NonConvexBound to generate.
    act_type: Type of activation
    lb_fun: Function to evaluate the upper bound of the activation for an input.
    ub_fun: Function to evaluate the upper bound of the activation for an input.
  Returns:
    out_bounds: NonConvexBound for the output of the non-linearity
  """

  parents_var = inp.variables
  variables = parents_var.copy()
  variables[index] = inp.shape

  def eval_fn(var_set, dummy_inps, activations):
    """Evaluate the value of the primal."""
    inp_eval = inp.evaluate(var_set, dummy_inps, activations)

    lb_val = lb_fun(inp_eval)
    ub_val = ub_fun(inp_eval)

    theta = var_set[index]
    out_val = lb_val + theta * (ub_val - lb_val)

    return out_val

  shape = inp.shape
  previous_bounds = inp.previous_bounds.copy()

  new_bound_ctor = bound_cls.get_nonlinearity_activation_constructor(
      index, inp, act_type, lb_fun, ub_fun)
  return new_bound_ctor(index, shape, previous_bounds,
                        eval_fn, variables, concretizer)


_linear_op_primitives = [
    (lax.conv_general_dilated_p, utils.wrapped_general_conv),
    (lax.add_p, lax.add),
    (lax.reshape_p, lax.reshape),
    (lax.dot_general_p, lax.dot_general),
    (lax.sub_p, lax.sub),
    (lax.mul_p, lambda x, y: x*y),
]
_nonconvex_primitive_transform = {
    primitive: functools.partial(_nonconvex_linear_op, primitive, fun)
    for primitive, fun in _linear_op_primitives}
_nonconvex_primitive_transform.update({
    lax.max_p: _nonconvex_max,
    lax.div_p: _nonconvex_div,
    synthetic_primitives.softplus_p: _nonconvex_softplus
})


def build_nonconvex_formulation(bound_cls, concretizer_ctor,
                                function, *bounds):
  """Builds the optimizable objective.

  Args:
    bound_cls: Bound class to use. This determines what dual formulation will
      be computed.
    concretizer_ctor: Constructor for the concretizer to use to obtain
      intermediate bounds.
    function: Function performing computation to obtain bounds for. Takes as
      only  arguments the network inputs.
    *bounds: jax_verify.NonConvexBound, bounds on the inputs of the function.
      These can be created using
      jax_verify.NonConvexBound.initial_nonconvex_bound.
  Returns:
    output_bound: NonConvex bound that can be optimized with a solver.
  """

  input_transform = functools.partial(bound_cls.initial_nonconvex_bound,
                                      concretizer_ctor)
  primitive_transform = {
      primitive: functools.partial(transform, bound_cls)
      for primitive, transform in _nonconvex_primitive_transform.items()
  }

  bound_transform = bound_propagation.OpwiseBoundTransform(
      input_transform, primitive_transform)
  output_bound, _ = bound_propagation.bound_propagation(
      bound_transform, function, *bounds)
  return output_bound
