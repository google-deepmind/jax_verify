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

"""Propagate the relaxation through the network.

This is accomplished by traversing the JaxPR representation of the computation
and translating the computational graph.
"""
import abc
import functools
from typing import Callable, Tuple, Dict, Union, List, Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.src import activation_relaxation
from jax_verify.src import bound_propagation
from jax_verify.src import ibp
from jax_verify.src import synthetic_primitives
import numpy as np

Tensor = jnp.ndarray


class RelaxVariable(bound_propagation.Bound):
  """Variable used to build relaxation."""

  def __init__(self, idx: bound_propagation.Index, base_bound):
    self.base_bound = base_bound
    self.name = 'rlxvar'
    for idx_cpt in idx:
      self.name += f'_{idx_cpt}'
    self.idx = idx
    self.constraints = None

  def set_constraints(self, constraints):
    self.constraints = constraints

  @property
  def lower(self):
    return self.base_bound.lower

  @property
  def upper(self):
    return self.base_bound.upper


class OptRelaxVariable(RelaxVariable):
  """Variant of RelaxVariable with lazy optimization tightening."""

  def __init__(self, base_relax_variable, optimize_transform):
    super(OptRelaxVariable, self).__init__(base_relax_variable.idx,
                                           base_relax_variable.base_bound)
    self._shape = base_relax_variable.shape
    self._opted_bounds = None
    self._optimize_transform = optimize_transform

  @property
  def shape(self):
    # Need to specify a shape method, because otherwise, a call to `.shape`
    # will attempt to read `.lower` and trigger optimization
    return self._shape

  @property
  def lower(self):
    return self._optimized_bounds.lower

  @property
  def upper(self):
    return self._optimized_bounds.upper

  @property
  def _optimized_bounds(self):
    if self._opted_bounds is None:
      self._opted_bounds = self._optimize_transform.tight_bounds(self)
    return self._opted_bounds


class BinaryVariable():
  """Binary variable."""

  def __init__(self, idx, shape):
    self.shape = shape
    self.name = f'boolvar_{idx}'
    self.idx = idx


class MIPActivationConstraint:
  """MIP constraint to encode activation."""

  def __init__(self, outvar, invar, binvar, mask, binscale, scale, bias, sense):
    """Represents: outvar =(>)(<) scale * invar + binscale * binvar + bias."""
    self.outvar = outvar
    self.invar = invar
    self.binvar = binvar
    self.binscale = binscale
    self.mask = mask
    self.scale = scale
    self.bias = bias
    self.sense = sense

  def encode_into_solver(self, solver: 'MIPSolver', index: int):
    """Encode the linear constraints into the provided solver.

    Args:
      solver: MIPSolver to create the exact constraint into.
      index: Index in the batch for which to build the variable.
    """
    biases = np.reshape(self.bias[index, ...], [-1])
    slopes = np.reshape(self.scale[index, ...], [-1])
    binslopes = np.reshape(self.binscale[index, ...], [-1])
    mask = np.reshape(self.mask[index, ...], [-1])
    for act_index, (binslope, slope, bias) in enumerate(
        zip(binslopes, slopes, biases)):
      if mask[act_index]:
        solver.create_mip_activation_solver_constraint(
            self, act_index, binslope=binslope.item(),
            slope=slope.item(), bias=bias.item())


class LinearConstraint:
  """Linear constraint, to be encoded into a solver."""

  def __init__(self, vars_and_coeffs, bias, sense):
    self._vars_and_coeffs = vars_and_coeffs
    self._bias = bias
    self.sense = sense
    self.sample_dependent = bool(self._bias.shape)

  def bias(self, index: int):
    """Get the bias corresponding to the minibatch sample `index`.

    If bias has a dimension, it means it is sample dependent. Otherwise,
    the bias is the same for all samples.

    Args:
      index: Index in the batch for which to build the variable.
    Returns:
      bias: value of the bias.
    """
    return self._bias[index] if self.sample_dependent else self._bias

  def vars_and_coeffs(self, index: int):
    """Get the variable and coefficients corresponding to the sample `index`.

    If coeffs has a dimension, the coefficients are sample dependent. Otherwise,
    the coefficients are the same for all samples.

    Args:
      index: Index in the batch for which to build the variable.
    Returns:
      vars_and_coeffs: vars_and_coeffs list where the coefficients are the one
        corresponding to the sample `index`
    """
    if self.sample_dependent:
      return [(var, (cpts, coeffs[index]))
              for (var, (cpts, coeffs)) in self._vars_and_coeffs]
    else:
      return self._vars_and_coeffs

  def encode_into_solver(self, solver: 'RelaxationSolver', index: int):
    """Encode the linear constraints into the provided solver.

    Args:
      solver: RelaxationSolver to create the linear constraint into.
      index: Index in the batch for which to build the variable.
    """
    solver.create_linear_solver_constraint(self, index)


class RelaxActivationConstraint:
  """Linear constraint involved in the relaxation of an activation."""

  def __init__(self, outvar, invar, mask, scale, bias, sense):
    """Represents the constraint outvar =(>)(<) scale * invar + bias."""
    self.outvar = outvar
    self.invar = invar
    self.mask = mask
    self.scale = scale
    self.bias = bias
    self.sense = sense

  def encode_into_solver(self, solver: 'RelaxationSolver', index: int):
    """Encode the linear constraints into the provided solver.

    Args:
      solver: RelaxationSolver to create the linear constraint into.
      index: Index in the batch for which to build the variable.
    """
    biases = np.reshape(self.bias[index, ...], [-1])
    slopes = np.reshape(self.scale[index, ...], [-1])
    mask = np.reshape(self.mask[index, ...], [-1])
    for act_index, (slope, bias) in enumerate(zip(slopes, biases)):
      if mask[act_index]:
        solver.create_activation_solver_constraint(
            self, act_index, slope.item(), bias.item())


class RelaxationSolver(metaclass=abc.ABCMeta):
  """Abstract solver for the relaxation."""

  def maybe_create_solver_variable(
      self,
      var: RelaxVariable,
      index: int):
    """Create a new solver variable for var if it has not been created yet.

    Args:
      var: Variable generated by the relaxation bound propagation.
      index: Index in the batch for which to build the variable.
    """
    if not self._variable_already_created(var):
      self._create_solver_relax_variable(var, index)

  @abc.abstractmethod
  def _create_solver_relax_variable(
      self,
      var: RelaxVariable,
      index: int):
    """Create a new bound-constrained variable based on a RelaxVariable.

    Args:
      var: Variable generated by the relaxation bound propagation.
      index: Index in the batch for which to build the variable.
    """

  @abc.abstractmethod
  def _variable_already_created(
      self,
      var: Union[RelaxVariable, BinaryVariable],
      ) -> bool:
    """Check whether the solver has already created a variable for var.

    Args:
      var: Variable generated by the relaxation bound propagation.
    """

  @abc.abstractmethod
  def create_linear_solver_constraint(
      self,
      constraint: LinearConstraint,
      index: int):
    """Create a new solver linear constraint.

    Args:
      constraint: Constraint generated by the relaxation bound propagation.
      index: Index in the batch for which to build the variable.
    """

  @abc.abstractmethod
  def create_activation_solver_constraint(
      self,
      constraint: RelaxActivationConstraint,
      act_index: int,
      slope: float,
      bias: float):
    """Create the linear constraint involved in the activation relaxation.

    Args:
      constraint: Constraint generated by the relaxation bound propagation.
      act_index: Index of the activation to encode (in the variables involved
        in constraint)
      slope : Slope coefficients of the linear inequality.
      bias: Bias of the linear inequality
    """

  @abc.abstractmethod
  def minimize_objective(
      self,
      var_name: str,
      objective: Tensor,
      objective_bias: float,
      time_limit_millis: Optional[int],
  ) -> Tuple[float, Dict[str, Tensor], bool]:
    """Minimize a linear function.

    Args:
      var_name: Index of the variable to define a linear function over the
        components.
      objective: Coefficients of the linear function.
      objective_bias: Bias of the linear function.
      time_limit_millis: Maximum solve time in ms. Use None for unbounded.
    Returns:
      val: Value of the minimum.
      solution: Solution found.
      status: Status of the optimization function.
    """


class MIPSolver(RelaxationSolver):
  """Abstract solver for the MIP encoding."""

  def maybe_create_solver_variable(
      self,
      var: Union[RelaxVariable, BinaryVariable],
      index: int):
    """Create a new solver variable for var if it has not been created yet.

    Args:
      var: Variable generated by the relaxation bound propagation.
      index: Index in the batch for which to build the variable.
    """
    if not self._variable_already_created(var):
      if isinstance(var, BinaryVariable):
        self._create_solver_bool_variable(var, index)
      else:
        self._create_solver_relax_variable(var, index)

  @abc.abstractmethod
  def _create_solver_bool_variable(
      self,
      var: BinaryVariable,
      index: int):
    """Create a new bound-constrained variable based on a BinaryVariable.

    Args:
      var: Variable generated by the relaxation bound propagation.
      index: Index in the batch for which to build the variable.
    """

  @abc.abstractmethod
  def create_mip_activation_solver_constraint(
      self,
      constraint: MIPActivationConstraint,
      act_index: int,
      binslope: float,
      slope: float,
      bias: float):
    """Create the linear constraint involved in the activation relaxation.

    Args:
      constraint: Constraint generated by the relaxation bound propagation.
      act_index: Index of the activation to encode (in the variables involved
        in constraint)
      binslope: Slope coefficients for the binary variable.
      slope : Slope coefficients for the input variable.
      bias: Bias of the linear inequality
    """


def encode_relaxation(
    solver_ctor: Callable[[], RelaxationSolver],
    env: Dict[jax.core.Var, Union[RelaxVariable, Tensor]],
    index: int,
) -> RelaxationSolver:
  """Creates a solver and encodes the relaxation into it.

  Args:
    solver_ctor: Constructor for the solver.
    env: Environment created by applying boundprop with relaxation.py
    index: The index in the minibatch for which the relaxation should be
      encoded.
  Returns:
    solver: Solver containing the relaxation of the network encoded.
  """
  solver = solver_ctor()
  for key in env.keys():
    if isinstance(env[key], RelaxVariable):
      variable = env[key]
      # Create the variable in the solver.
      solver.maybe_create_solver_variable(variable, index)
      # Create the constraints in the solver.
      if variable.constraints:
        for constraint in variable.constraints:
          if isinstance(constraint, MIPActivationConstraint):
            # create manually the binary variable because it is not collected
            # automatically by the graph propagation.
            solver.maybe_create_solver_variable(constraint.binvar, index)
          constraint.encode_into_solver(solver, index)
  return solver


def solve_relaxation(
    solver_ctor: Callable[[], RelaxationSolver],
    objective: Tensor,
    objective_bias: float,
    variable_opt: RelaxVariable,
    env: Dict[jax.core.Var, Union[RelaxVariable, Tensor]],
    index: int,
    time_limit_millis: Optional[int] = None,
) -> Tuple[float, Dict[str, Tensor], bool]:
  """Solves the relaxation using the provided LP solver.

  Args:
    solver_ctor: Constructor for the solver.
    objective: Objective to optimize, given as an array of coefficients to be
      applied to the variable to form a linear objective function
    objective_bias: Bias to add to objective
    variable_opt: RelaxVariable over which the linear function to optimize
      is defined.
    env: Environment created by applying boundprop with relaxation.py
    index: The index in the minibatch for which the LP should be solved
    time_limit_millis: Time limit on solver. None if unbounded.
  Returns:
    opt: Value of the solution found.
    solution: Solution found by the solver.
    status: Whether the optimal solution has been achieved.
  """
  solver = encode_relaxation(solver_ctor, env, index)
  return solver.minimize_objective(variable_opt.name,
                                   objective, objective_bias, time_limit_millis)


def _get_linear(primitive, outval, *eqn_invars, **params):
  """Get linear expressions corresponding to an affine layer.

  Args:
    primitive: jax primitive
    outval: dummy tensor shaped according to a single example's outputs
    *eqn_invars: Arguments of the primitive, wrapped as RelaxVariables
    **params: Keyword Arguments of the primitive.

  Returns:
    For each output component, a pair `(bias, coefficients)`, where
    `coefficients` is a list of `(component, coefficient)` pairs.
  """
  def funx(x):
    if isinstance(x, RelaxVariable):
      return jnp.zeros(x.shape)
    else:
      return x
  def fungrad(i, args):
    return jnp.reshape(primitive.bind(*args, **params)[0, ...], [-1])[i]
  results = []
  # Loop over output dimensions one at a time to avoid creating a large
  # materialized tensor
  # TODO: Replace with something more efficient
  for i in range(outval.size):
    fung = functools.partial(fungrad, i)
    bias, current_grad = jax.value_and_grad(fung, allow_int=True)(
        [funx(x) for x in eqn_invars])
    coefficients = []
    for res, in_var in zip(current_grad, eqn_invars):
      if isinstance(in_var, RelaxVariable):
        components = jnp.flatnonzero(res)
        coefficients.append((components, res.ravel()[components]))
      else:
        coefficients.append(None)
    results.append((bias, coefficients))
  return results


def _relax_input(
    index: bound_propagation.Index, in_bounds: bound_propagation.Bound,
) -> RelaxVariable:
  """Generates the initial inputs for the relaxation.

  Args:
    index: Integer identifying the input node.
    in_bounds: Concrete bounds on the input node.
  Returns:
    `RelaxVariable` for the initial inputs.
  """
  # Wrap initial bound as RelaxVariable bound.
  in_variable = RelaxVariable(index, in_bounds)
  return in_variable


_order_preserving_reshapes = [lax.reshape_p, lax.squeeze_p]
_affine_primitives_list = (
    bound_propagation.AFFINE_PRIMITIVES +
    bound_propagation.RESHAPE_PRIMITIVES +
    [lax.div_p]
)


def _relax_primitive(
    index: bound_propagation.Index, out_bounds: bound_propagation.Bound,
    primitive: jax.core.Primitive,
    *args, use_mip: bool = False, **kwargs
    ) -> RelaxVariable:
  """Generates the relaxation for a given primitive op.

  Args:
    index: Integer identifying the computation node.
    out_bounds: Concrete bounds on the outputs of the primitive op.
    primitive: jax primitive.
    *args: Arguments of the primitive, wrapped as RelaxVariables
    use_mip: whether to use mixed integer programming for the activation
      constraints.
    **kwargs: Keyword Arguments of the primitive.
  Returns:
    `RelaxVariable` that contains the output of this primitive for the
    relaxation, with all the constraints linking the output to inputs.
  """
  # Create variable for output of this primitive
  out_variable = RelaxVariable(index, out_bounds)
  # Create constraints linking output and input of primitive
  constraints = []
  if primitive in _order_preserving_reshapes:
    invar = args[0]
    constraints = [RelaxActivationConstraint(outvar=out_variable,
                                             invar=invar,
                                             mask=jnp.ones(invar.shape),
                                             scale=jnp.ones(invar.shape),
                                             bias=jnp.zeros(invar.shape),
                                             sense=0)]
  elif primitive in _affine_primitives_list:
    if primitive == lax.div_p and isinstance(args[1], RelaxVariable):
      raise NotImplementedError(
          'Division with non-constant divisor is not supported')
    results = _get_linear(primitive, out_bounds.lower[0, ...],
                          *args, **kwargs)
    for i, (bias, coeffs) in enumerate(results):
      # Coefficients of the input variable(s).
      vars_and_coeffs = [
          (arg, coeff) for arg, coeff in zip(args, coeffs)
          if isinstance(arg, RelaxVariable)]
      # Equate with the output variable, by using a coefficient of -1.
      out_coeff = (np.array([i], dtype=np.int64), np.array([-1.]))
      vars_and_coeffs.append((out_variable, out_coeff))
      constraints.append(LinearConstraint(vars_and_coeffs, bias, 0))
  elif primitive in activation_relaxation.relaxation_fns:
    # Generate convex relaxation.
    safe_kwargs = synthetic_primitives.filter_jaxverify_kwargs(kwargs)
    activation = activation_relaxation.relaxation_fns[primitive]
    lb_funs, ub_funs = activation.piecewise_linear_relaxation_fn(*args,
                                                                 **safe_kwargs)
    invar, = args
    zeros = jnp.zeros_like(invar.lower)
    ones = jnp.ones_like(invar.lower)
    if activation.pos_neg_linear and (len(lb_funs) == 1 or len(ub_funs) == 1):
      # Use equality constraints if linear over the entire interval.
      ambiguous = (invar.lower < 0) & (invar.upper > 0)
      chord, = lb_funs if len(lb_funs) == 1 else ub_funs
      constraints.append(RelaxActivationConstraint(
          outvar=out_variable,
          invar=invar,
          mask=(~ambiguous),
          scale=(chord(ones) - chord(zeros)),
          bias=chord(zeros),
          sense=0))
    else:
      ambiguous = ones

    for lb_fun in lb_funs:
      # act(x) >= lb(x)
      constraints.append(RelaxActivationConstraint(
          outvar=out_variable,
          invar=invar,
          mask=ambiguous,
          scale=(lb_fun(ones) - lb_fun(zeros)),
          bias=lb_fun(zeros),
          sense=1))

    for ub_fun in ub_funs:
      # act(x) <= ub(x)
      constraints.append(RelaxActivationConstraint(
          outvar=out_variable,
          invar=invar,
          mask=ambiguous,
          scale=(ub_fun(ones) - ub_fun(zeros)),
          bias=ub_fun(zeros),
          sense=-1))

    if use_mip:
      if primitive is not synthetic_primitives.relu_p:
        raise ValueError(
            f'Only ReLU activations supported. Encountered {primitive}')
      binvar = BinaryVariable(index, out_bounds.lower.shape)
      constraints += [
          # outvar <= upper_bound * binvar
          MIPActivationConstraint(outvar=out_variable,
                                  invar=invar,
                                  binvar=binvar,
                                  binscale=invar.upper,
                                  mask=ambiguous,
                                  scale=zeros,
                                  bias=zeros,
                                  sense=-1),
          # outvar <= invar - lower_bound * (1. - binvar)
          MIPActivationConstraint(outvar=out_variable,
                                  invar=invar,
                                  binvar=binvar,
                                  binscale=invar.lower,
                                  mask=ambiguous,
                                  scale=ones,
                                  bias=-invar.lower,
                                  sense=-1),
      ]
  else:
    raise NotImplementedError(f'Unsupported primitive: {primitive}')

  out_variable.set_constraints(constraints)
  return out_variable


class RelaxationTransform(bound_propagation.GraphTransform[RelaxVariable]):
  """Transform to produce `RelaxVariable`s for each op."""

  def __init__(
      self,
      boundprop_transform: bound_propagation.BoundTransform,
      use_mip: bool = False,
  ):
    """Defines relaxation constraint propagation.

    Args:
      boundprop_transform: Basic Jax primitive ops' equivalents for
        the underlying bound propagation method.
      use_mip: whether to use mixed integer programming for the activation
        constraints.
    """
    self._boundprop_transform = boundprop_transform
    self._use_mip = use_mip

  def input_transform(self, context, input_bound):
    in_bounds = self._boundprop_transform.input_transform(
        context, input_bound)
    return _relax_input(context.index, in_bounds)

  def primitive_transform(self, context, primitive, *args, **params):
    interval_args = [arg.base_bound if isinstance(arg, RelaxVariable) else arg
                     for arg in args]
    out_bounds = self._boundprop_transform.equation_transform(
        context, primitive, *interval_args, **params)
    return _relax_primitive(
        context.index, out_bounds, primitive, *args,
        use_mip=self._use_mip, **params)


class OptimizedRelaxationTransform(
    bound_propagation.GraphTransform[OptRelaxVariable]):
  """Wraps a RelaxVariable-producing BoundTransform to add optimization."""

  def __init__(
      self,
      transform: bound_propagation.GraphTransform[RelaxVariable],
      solver_ctor: Callable[[], RelaxationSolver],
      time_limit_millis: Optional[int] = None):
    self._transform = transform
    self.solver_ctor = solver_ctor
    self.solvers: List[RelaxationSolver] = []
    self._time_limit_millis = time_limit_millis

  def tight_bounds(self, variable: RelaxVariable) -> ibp.IntervalBound:
    """Compute tighter bounds based on the LP relaxation.

    Args:
      variable: Variable as created by the base boundprop transform. This is a
        RelaxVariable that has already been encoded into the solvers.
    Returns:
      tightened_base_bound: Bounds tightened by optimizing with the LP solver.
    """
    lbs = []
    ubs = []
    for solver in self.solvers:
      nb_targets = np.prod(variable.shape[1:])
      sample_lbs = []
      sample_ubs = []
      for target_idx in range(nb_targets):
        objective = (jnp.arange(nb_targets) == target_idx).astype(jnp.float32)
        lb, _, optimal_lb = solver.minimize_objective(
            variable.name, objective, 0., self._time_limit_millis)
        assert optimal_lb
        neg_ub, _, optimal_ub = solver.minimize_objective(
            variable.name, -objective, 0., self._time_limit_millis)
        assert optimal_ub
        sample_lbs.append(lb)
        sample_ubs.append(-neg_ub)
      lbs.append(sample_lbs)
      ubs.append(sample_ubs)

    tightened_base_bound = ibp.IntervalBound(
        jnp.reshape(jnp.array(lbs), variable.shape),
        jnp.reshape(jnp.array(ubs), variable.shape))
    return tightened_base_bound

  def input_transform(
      self,
      context: bound_propagation.TransformContext,
      input_bound: bound_propagation.InputBound,
  ) -> RelaxVariable:
    in_bounds = self._transform.input_transform(
        context, input_bound)
    for minibatch_index in range(in_bounds.shape[0]):
      # Create one solver instance for each problem in the batch because they
      # will have different constraints.
      if minibatch_index >= len(self.solvers):
        self.solvers.append(self.solver_ctor())
      solver = self.solvers[minibatch_index]
      solver.maybe_create_solver_variable(in_bounds, minibatch_index)
    return in_bounds

  def primitive_transform(
      self,
      context: bound_propagation.TransformContext,
      primitive: jax.core.Primitive,
      *args: Union[RelaxVariable, Tensor],
      **params,
  ) -> RelaxVariable:
    basic_relax_var = self._transform.equation_transform(
        context, primitive, *args, **params)
    opt_relax_var = OptRelaxVariable(basic_relax_var, self)

    for minibatch_index, solver in enumerate(self.solvers):
      # Encode the new variable and the associated constraints. We encode the
      # basic variable, not the optimized one. This way, the optimization is
      # only performed if the lower / upper bounds are required and the bounds
      # on that variable are accessed "from outside".
      solver.maybe_create_solver_variable(basic_relax_var, minibatch_index)
      if basic_relax_var.constraints:
        for constraint in basic_relax_var.constraints:
          if isinstance(constraint, MIPActivationConstraint):
            # create manually the binary variable because it is not collected
            # automatically by the graph propagation.
            solver.maybe_create_solver_variable(
                constraint.binvar, minibatch_index)
          constraint.encode_into_solver(solver, minibatch_index)
    return opt_relax_var
