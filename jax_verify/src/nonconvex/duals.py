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

"""Implement the dual computations for the NonConvex Reformulation."""

import collections
import functools
from typing import Callable, Dict, Tuple, Union, List, DefaultDict, Optional

import jax
import jax.numpy as jnp

from jax_verify.src import bound_propagation
from jax_verify.src import synthetic_primitives
from jax_verify.src.nonconvex import nonconvex

Tensor = jnp.ndarray
Index = bound_propagation.Index
ParamSet = nonconvex.ParamSet
EvalFunArgs = [ParamSet, ParamSet, ParamSet]
WolfeDualFn = Callable[[ParamSet, Tensor, ParamSet, Tensor, Tensor, ParamSet],
                       Tensor]
LagrangianLevelFn = Callable[[Tensor, ParamSet], Tensor]
LagrangianBoundingFn = Callable[[Tensor, ParamSet], Tensor]
LagrangianVarTerm = Tuple[
    Union[str, bound_propagation.Primitive],
    Callable[..., Tensor]]
LagrangianDict = DefaultDict[Index, List[LagrangianVarTerm]]
LagrangianVartermsFn = Callable[[Tensor, LagrangianDict], None]


def _sum_fn(fn, *args, **kwargs):
  out = fn(*args, **kwargs)
  summand = out[0] if isinstance(out, tuple) else out
  return summand.sum(), out


def _sum_over_acts(var: Tensor) -> Tensor:
  return var.sum(axis=tuple(range(1, var.ndim)))

CVX_HULL_PRIMITIVES = (synthetic_primitives.relu_p,
                       synthetic_primitives.softplus_p,
                       synthetic_primitives.posbilinear_p,
                       )


class WolfeNonConvexBound(nonconvex.ConstrainedNonConvexBound):
  """This subclass allows the computation of the WolfeDual.

  This is done through the `wolfe_dual_fn`, which propagates the dual variables
  backwards. The quantity propagated backwards (dvar) needs to
  be split between pos_dvar (dual variable on [lb_fun() - z]) and
  neg_dvar (dual variable on [z - ub_fun()]).

  In the presence of imposed (concrete) constraints, those need to be specified
  differently. We need:
  (bound_posdvar - bound_negdvar) + (boundfun_posdvar - boundfun_negdvar)
    = dvar_cte + sum_{j>i} (boundfun_posdvar_j * lb_fun()
                             - boundfun_negdvar_j * ub_fun())
  which means that we have degrees of freedom.

  Current implementation decides based on a greedy approach, ignoring the
  downstream consequences. This could be improved.
  """

  def __init__(
      self,
      wolfe_dual_fn: WolfeDualFn,
      index: Index,
      shape: Tuple[int, ...],
      previous_bounds: Dict[Index, 'WolfeNonConvexBound'],
      eval_fn: Callable[EvalFunArgs, Tensor],
      variables: Dict[Index, Tuple[int, ...]],
      concretized_bounds: Optional[bound_propagation.Bound] = None):
    """Create a NonConvexBound that can compute the WolfeDual.

    Args:
      wolfe_dual_fn: Function performing backward propagation of bounds for the
        wolfe dual and computing the contribution of this layer to the dual.
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

    super().__init__(index, shape, previous_bounds,
                     eval_fn, variables, concretized_bounds)
    self.wolfe_dual_fn = wolfe_dual_fn

  def dual(self, var_set: ParamSet, objectives: ParamSet
           ) -> Tuple[Tensor, Tensor]:
    dual_vars, acts = self._compute_dualvars_convexgrad(var_set, objectives)
    primal = self._objective_fn(acts, objectives)

    dual_gap = 0
    index_bound_list = list(self.previous_bounds.items())
    for index, intermediate_bound in reversed(index_bound_list):
      wolfe_dual_fn = intermediate_bound.wolfe_dual_fn
      if intermediate_bound.is_constrained():
        wolfe_dual_contrib = wolfe_dual_fn(
            var_set, dual_vars[index], acts,
            intermediate_bound.lower, intermediate_bound.upper, dual_vars)
      else:
        wolfe_dual_contrib = wolfe_dual_fn(
            var_set, dual_vars[index], acts, None, None, dual_vars)
      dual_gap = dual_gap + wolfe_dual_contrib

    wolfe_dual = primal + dual_gap
    return primal, wolfe_dual

  @classmethod
  def get_initial_bound_constructor(
      cls: Callable[..., 'WolfeNonConvexBound'],
      index: Index,
      lb: Tensor,
      ub: Tensor) -> Callable[..., 'WolfeNonConvexBound']:

    def wolfe_dual_fn(var_set: ParamSet,
                      dvar: Tensor,
                      acts: ParamSet,
                      bound_lb: Tensor, bound_ub: Tensor,
                      dual_vars: ParamSet) -> Tensor:
      del var_set
      del bound_lb
      del bound_ub
      del dual_vars
      pos_dvar = jnp.maximum(dvar, 0.)
      neg_dvar = jnp.maximum(-dvar, 0.)
      x_0 = acts[index]
      dual_contrib = _sum_over_acts(pos_dvar * (lb - x_0)
                                    + neg_dvar * (x_0 - ub))
      return dual_contrib

    return functools.partial(cls, wolfe_dual_fn)

  @classmethod
  def get_linear_activation_constructor(
      cls: Callable[..., 'WolfeNonConvexBound'],
      index: Index,
      vlin_fun: Callable[..., Tensor],
      in_vals: List[Union['WolfeNonConvexBound', Tensor]],
  ) -> Callable[..., 'WolfeNonConvexBound']:

    def wolfe_dual_fn(var_set: ParamSet,
                      dvar: Tensor,
                      acts: ParamSet,
                      bound_lb: Tensor, bound_ub: Tensor,
                      dual_vars: ParamSet) -> Tensor:
      pos_dvar = jnp.maximum(dvar, 0.)
      neg_dvar = jnp.maximum(-dvar, 0.)
      all_inps = [nonconvex.eval_if_nonconvexbound(inp, var_set, None, acts)
                  for inp in in_vals]

      if bound_lb is not None:
        bound_fun_eval = vlin_fun(all_inps)
        brd_bound_lb = jnp.expand_dims(bound_lb, 0)
        brd_bound_ub = jnp.expand_dims(bound_ub, 0)

        bound_posdvar = jnp.where(brd_bound_lb >= bound_fun_eval,
                                  pos_dvar, jnp.zeros_like(pos_dvar))
        pos_dvar = pos_dvar - bound_posdvar
        bound_negdvar = jnp.where(brd_bound_ub <= bound_fun_eval,
                                  neg_dvar, jnp.zeros_like(neg_dvar))
        neg_dvar = neg_dvar - bound_negdvar

      _, backprop = jax.vjp(vlin_fun, all_inps)
      all_pp_dvars = backprop(pos_dvar)[0]
      all_nq_dvars = backprop(neg_dvar)[0]

      for i, inp_i in enumerate(in_vals):
        if not isinstance(inp_i, nonconvex.NonConvexBound):
          continue
        prev_dvar = all_pp_dvars[i] - all_nq_dvars[i]
        if inp_i.index in dual_vars:
          prev_dvar = dual_vars[inp_i.index] + prev_dvar
        dual_vars[inp_i.index] = prev_dvar

      if bound_lb is not None:
        act_out_eval = acts[index]
        bound_fun_to_out = bound_fun_eval - act_out_eval
        dual_contrib = _sum_over_acts(
            jnp.where(bound_posdvar > 0,
                      bound_posdvar * (brd_bound_lb - act_out_eval),
                      jnp.zeros_like(bound_posdvar))
            + jnp.where(bound_negdvar > 0,
                        bound_negdvar * (act_out_eval - brd_bound_ub),
                        jnp.zeros_like(bound_negdvar))
            + (pos_dvar - neg_dvar) * bound_fun_to_out)
        return dual_contrib
      else:
        # There shouldn't be a contrib term here; everything cancels out.
        return 0

    return functools.partial(cls, wolfe_dual_fn)

  @classmethod
  def get_nonlinearity_activation_constructor(
      cls: Callable[..., 'WolfeNonConvexBound'],
      index: Index,
      act_type: bound_propagation.Primitive,
      lb_fun: Callable[[Tensor], Tensor],
      ub_fun: Callable[[Tensor], Tensor],
      *inp: 'WolfeNonConvexBound',
  ) -> Callable[..., 'WolfeNonConvexBound']:
    def wolfe_dual_fn(var_set: ParamSet,
                      dvar: Tensor,
                      acts: ParamSet,
                      bound_lb: Tensor, bound_ub: Tensor,
                      dual_vars: ParamSet) -> Tensor:
      pos_dvar = jnp.maximum(dvar, 0.)
      neg_dvar = jnp.maximum(-dvar, 0.)
      inp_val = [inp_i.evaluate(var_set, {}, acts) for inp_i in inp]

      lb_val, lb_backprop = jax.vjp(lb_fun, *inp_val)
      ub_val, ub_backprop = jax.vjp(ub_fun, *inp_val)

      if bound_lb is not None:
        brd_bound_lb = jnp.expand_dims(bound_lb, 0)
        brd_bound_ub = jnp.expand_dims(bound_ub, 0)

        bound_posdvar = jnp.where(brd_bound_lb >= lb_val,
                                  pos_dvar, jnp.zeros_like(pos_dvar))
        pos_dvar = pos_dvar - bound_posdvar
        bound_negdvar = jnp.where(brd_bound_ub <= ub_val,
                                  neg_dvar, jnp.zeros_like(neg_dvar))
        neg_dvar = neg_dvar - bound_negdvar

      backprop_pos = lb_backprop(pos_dvar)
      backprop_neg = ub_backprop(neg_dvar)
      for (i, inp_i) in enumerate(inp):
        prev_dvar = backprop_pos[i] - backprop_neg[i]
        if inp_i.index in dual_vars:
          prev_dvar = dual_vars[inp_i.index] + prev_dvar
        dual_vars[inp_i.index] = prev_dvar

      out_val = acts[index]

      dual_contrib = _sum_over_acts(neg_dvar * (out_val - ub_val)
                                    + pos_dvar * (lb_val - out_val))
      if bound_lb is not None:
        dual_contrib = dual_contrib + _sum_over_acts(
            jnp.where(bound_posdvar > 0,
                      bound_posdvar * (brd_bound_lb - out_val),
                      jnp.zeros_like(bound_posdvar))
            + jnp.where(bound_negdvar > 0,
                        bound_negdvar * (out_val - brd_bound_ub),
                        jnp.zeros_like(bound_negdvar)))
      return dual_contrib

    return functools.partial(cls, wolfe_dual_fn)

  def requires_concretizing(self, consuming_primitive):
    needs_concretizing = (consuming_primitive is None
                          or consuming_primitive in CVX_HULL_PRIMITIVES)
    return (self._concretized_bounds is None) and needs_concretizing


def _initial_lagrangian_term(dvar: Tensor, lb: Tensor, ub: Tensor, x: Tensor
                             ) -> Tensor:
  pos_dvar = jnp.maximum(dvar, 0.)
  neg_dvar = jnp.maximum(-dvar, 0.)
  dual_contrib = (neg_dvar * (x - ub) + pos_dvar * (lb - x))
  return dual_contrib


class LinLagrangianNonConvexBound(nonconvex.NonConvexBound):
  """This subclass allows the computation of the Linearized Lagrangian dual.

  The lagrangian and its linearization are obtained through the
  `lagrangian_level_fn` which compute the contribution of this layer to the
  lagrangian, based on precomputed activation.
  The minimization of linear function (such as the linearized lagrangian) over
  the feasible domain is done through the `bounding_fn` function.
  """

  def __init__(self,
               lagrangian_level_fn: LagrangianLevelFn,
               bounding_fn: LagrangianBoundingFn,
               index: Index,
               shape: Tuple[int, ...],
               previous_bounds: Dict[Index, 'LinLagrangianNonConvexBound'],
               eval_fn: Callable[EvalFunArgs, Tensor],
               variables: Dict[Index, Tuple[int, ...]],
               concretized_bounds: Optional[bound_propagation.Bound] = None):
    """Create a NonConvexBound that can compute the Linearized Lagrangian dual.

    Args:
      lagrangian_level_fn: Function returning the contribution of this layer to
        the lagrangian, based on precomputed activations.
      bounding_fn: Function to perform linear minimization over the domain of
        an activation.
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
    super(LinLagrangianNonConvexBound, self).__init__(
        index, shape, previous_bounds, eval_fn,
        variables, concretized_bounds)
    self.lagrangian_level_fn = lagrangian_level_fn
    self.bounding_fn = bounding_fn

    def lagrangian(acts: ParamSet,
                   objectives: ParamSet,
                   dual_vars: Tensor,
                   ) -> Tuple[Tensor, ParamSet]:
      primals = self._objective_fn(acts, objectives)
      lagrangian = primals
      for index, intermediate_bound in self.previous_bounds.items():
        lagrangian_level_fn = intermediate_bound.lagrangian_level_fn
        dvar = dual_vars[index]
        contrib = lagrangian_level_fn(dvar, acts)
        lagrangian += contrib

      return lagrangian, primals

    self._lagrangian_fn = lagrangian
    self._lagrangian_sumfn = functools.partial(_sum_fn, lagrangian)

  def dual(self, var_set: ParamSet, objectives: ParamSet) -> Tensor:
    dual_vars, acts = self._compute_dualvars_nonconvexgrad(var_set, objectives)

    # Compute the gradients of all the lagrangians (done by taking their sum),
    # with regards to the activations.
    lag_grad_fun = jax.value_and_grad(self._lagrangian_sumfn, argnums=0,
                                      has_aux=True)
    ((_, (lagrangians, primals)),
     laggrad_wrt_acts) = lag_grad_fun(acts, objectives, dual_vars)
    lin_duals = lagrangians
    for index, intermediate_bound in self.previous_bounds.items():
      bounding_fn = intermediate_bound.bounding_fn
      lag_grad = laggrad_wrt_acts[index]
      contrib = bounding_fn(lag_grad, acts)
      lin_duals += contrib
    return primals, lin_duals

  @classmethod
  def get_initial_bound_constructor(
      cls: Callable[..., 'LinLagrangianNonConvexBound'],
      index: Index,
      lb: Tensor,
      ub: Tensor) -> Callable[..., 'LinLagrangianNonConvexBound']:
    def lagrangian_level_fn(dvar: Tensor, acts: ParamSet) -> Tensor:
      x_0 = acts[index]
      dual_contrib = _sum_over_acts(_initial_lagrangian_term(dvar, lb, ub, x_0))
      return dual_contrib

    def bounding_fn(lag_grad: Tensor, acts: ParamSet) -> Tensor:
      x_0 = acts[index]
      bound_contrib = _sum_over_acts(jnp.maximum(lag_grad, 0.) * (lb - x_0) +
                                     jnp.minimum(lag_grad, 0.) * (ub - x_0))
      return bound_contrib
    return functools.partial(cls, lagrangian_level_fn, bounding_fn)

  @classmethod
  def get_linear_activation_constructor(
      cls: Callable[..., 'LinLagrangianNonConvexBound'],
      index: Index,
      vlin_fun: Callable[..., Tensor],
      in_vals: Tuple[Union['LinLagrangianNonConvexBound', Tensor], ...]
  ) -> Callable[..., 'LinLagrangianNonConvexBound']:

    def lagrangian_level_fn(dvar: Tensor, acts: ParamSet) -> Tensor:
      act_inp_eval = [
          acts[inp.index] if isinstance(inp, nonconvex.NonConvexBound) else inp
          for inp in in_vals]
      # Because this is linear, the function is both the lower bound and the
      # upper bound.
      act_out_eval = acts[index]
      f_inp_eval = vlin_fun(act_inp_eval)
      dual_contrib = _sum_over_acts(dvar * (f_inp_eval - act_out_eval))
      return dual_contrib

    def bounding_fn(lag_grad: Tensor, acts: ParamSet) -> Tensor:
      act_out_eval = acts[index]

      # We need to minimize the dotproduct between the lagrangian and the output
      # of that linear layer. Let's take the gradient (because everything is
      # linear and then we can simply assign bounds based on sign of gradient
      # coefficients.)
      dot_lagrangian_output = lambda x: (lag_grad * vlin_fun(x)).sum()
      act_inp_eval = [
          acts[inp.index] if isinstance(inp, nonconvex.NonConvexBound) else inp
          for inp in in_vals]
      minimizing_inps = []
      grads = jax.grad(dot_lagrangian_output)(act_inp_eval)
      for inp, grad in zip(in_vals, grads):
        if isinstance(inp, nonconvex.NonConvexBound):
          broad_lb = jnp.expand_dims(inp.lower, 0)
          broad_ub = jnp.expand_dims(inp.upper, 0)
          minimizing_inps.append(jnp.where(grad >= 0, broad_lb, broad_ub))
        else:
          minimizing_inps.append(inp)

      bound_contrib = _sum_over_acts((vlin_fun(minimizing_inps) - act_out_eval)
                                     * lag_grad)
      return bound_contrib

    return functools.partial(cls, lagrangian_level_fn, bounding_fn)

  @classmethod
  def get_nonlinearity_activation_constructor(
      cls: Callable[..., 'LinLagrangianNonConvexBound'],
      index: Index,
      act_type: bound_propagation.Primitive,
      lb_fun: Callable[[Tensor], Tensor],
      ub_fun: Callable[[Tensor], Tensor],
      *inp: 'LinLagrangianNonConvexBound',
      ) -> Callable[..., 'LinLagrangianNonConvexBound']:
    assert len(inp) == 1
    inp = inp[0]
    def lagrangian_level_fn(dvar: Tensor, acts: ParamSet) -> Tensor:
      pos_dvar = jnp.maximum(dvar, 0.)
      neg_dvar = jnp.maximum(-dvar, 0.)
      act_inp_eval = acts[inp.index]
      act_out_eval = acts[index]

      lb_val = lb_fun(act_inp_eval)
      ub_val = ub_fun(act_inp_eval)

      dual_contrib = _sum_over_acts(neg_dvar * (act_out_eval - ub_val)
                                    + pos_dvar * (lb_val - act_out_eval))
      return dual_contrib

    # We consider convex monotonous activation functions, so
    # - The lower bound is exact.
    # - The lower/upper bound on the output can be obtained by forwarding
    #   through the exact function the lower/upper bound on the input.
    out_lb = lb_fun(jnp.expand_dims(inp.lower, 0))
    out_ub = lb_fun(jnp.expand_dims(inp.upper, 0))

    def bounding_fn(lag_grad: Tensor, acts: ParamSet) -> Tensor:
      act_out_eval = acts[index]

      lb_val = jnp.expand_dims(out_lb, 0)
      ub_val = jnp.expand_dims(out_ub, 0)
      bound_contrib = _sum_over_acts(
          jnp.maximum(lag_grad, 0.) * (lb_val - act_out_eval)
          + jnp.minimum(lag_grad, 0.) * (ub_val - act_out_eval))
      return bound_contrib

    return functools.partial(cls, lagrangian_level_fn, bounding_fn)

  def requires_concretizing(self, consuming_primitive):
    return self._concretized_bounds is None


class MinLagrangianNonConvexBound(nonconvex.NonConvexBound):
  """This subclass allows the computation of the primal minimized lagrangian.

  The contribution of each primal variables are collected by the
  `lagrangian_varterms_fn`. It does not directly compute the lagrangian but
  fills in a dictionary mapping variables to the terms that involve them.
  This is done so that we can reorganize the lagrangian per variable, and then
  minimize it one variable at a time.
  """

  def __init__(self,
               lagrangian_varterms_fn: LagrangianVartermsFn,
               index: Index,
               shape: Tuple[int, ...],
               previous_bounds: Dict[Index, 'MinLagrangianNonConvexBound'],
               eval_fn: Callable[EvalFunArgs, Tensor],
               variables: Dict[Index, Tuple[int, ...]],
               concretized_bounds: Optional[bound_propagation.Bound] = None):
    """Create a NonConvexBound that can compute the primal minimized Lagrangian.

    Args:
      lagrangian_varterms_fn: Function filling in a dictionary mapping each
        variable to the terms involving it in the lagrangian.
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
    super(MinLagrangianNonConvexBound, self).__init__(
        index, shape, previous_bounds, eval_fn, variables, concretized_bounds)
    self.lagrangian_varterms_fn = lagrangian_varterms_fn

  def collect_lagrangian_varterms(self,
                                  objectives: ParamSet,
                                  dual_vars: ParamSet) -> LagrangianDict:
    lagrangian_dict = collections.defaultdict(list)
    for index, intermediate_bound in self.previous_bounds.items():
      lagrangian_varterms_fn = intermediate_bound.lagrangian_varterms_fn
      dvar = dual_vars[index]
      lagrangian_varterms_fn(dvar, lagrangian_dict)
    return lagrangian_dict

  def dual(self, var_set: ParamSet, objectives: ParamSet) -> Tensor:
    dual_vars, acts = self._compute_dualvars_nonconvexgrad(var_set, objectives)
    nb_targets = objectives[self.index].shape[0]

    # Compute the primals. This is not based on the activation minimizing the
    # lagrangian (because those are not necessarily primal feasible)
    primals = self._objective_fn(acts, objectives)

    lagrangian_terms = self.collect_lagrangian_varterms(objectives, dual_vars)
    # For each item in the network, we have a list of all the terms it is
    # involved in. Let's use this to minimize the lagrangian.
    opt_acts = {}
    for index, lag_terms in lagrangian_terms.items():
      intermediate_bound = self.previous_bounds[index]
      broad_lb = jnp.repeat(jnp.expand_dims(intermediate_bound.lower, axis=0),
                            nb_targets, axis=0)
      broad_ub = jnp.repeat(jnp.expand_dims(intermediate_bound.upper, axis=0),
                            nb_targets, axis=0)
      opt_acts[index] = _optimize_lagrangian_terms(lag_terms,
                                                   broad_lb, broad_ub)

    minimized_lagrangian = self._objective_fn(opt_acts, objectives)
    for index, lag_terms in lagrangian_terms.items():
      for _, lag_term_fn in lag_terms:
        out_term = lag_term_fn(opt_acts[index])
        minimized_lagrangian = minimized_lagrangian + _sum_over_acts(out_term)

    return primals, minimized_lagrangian

  @classmethod
  def get_initial_bound_constructor(
      cls: Callable[..., 'MinLagrangianNonConvexBound'],
      index: Index,
      lb: Tensor,
      ub: Tensor) -> Callable[..., 'MinLagrangianNonConvexBound']:
    def lagrangian_varterms_fn(dvar: Tensor, lagrangian_dict: LagrangianDict):
      lagrangian_dict[index].append(
          ('Linear', functools.partial(_initial_lagrangian_term, dvar, lb, ub)))
    return functools.partial(cls, lagrangian_varterms_fn)

  @classmethod
  def get_linear_activation_constructor(
      cls: Callable[..., 'MinLagrangianNonConvexBound'],
      index: Index,
      vlin_fun: Callable[..., Tensor],
      in_vals: List[Union['MinLagrangianNonConvexBound', Tensor]],
  ) -> Callable[..., 'MinLagrangianNonConvexBound']:

    def lagrangian_varterms_fn(dvar: Tensor, lagrangian_dict: LagrangianDict):
      # There is a linear term of dvar over the outputs.
      lagrangian_dict[index].append(('Linear', lambda x: (-dvar*x)))

      # If only one of the input is a variable, we can do things in a simple
      # way. Special casing this pattern avoids a bunch of failures on TPUs.
      inp_is_bound = list(isinstance(inp, nonconvex.NonConvexBound)
                          for inp in in_vals)
      if sum(inp_is_bound) == 1:
        bound_arg_pos = inp_is_bound.index(True)
        # The linear function has only one input, so we can just use it
        # directly.
        def single_input_vlin_fun(x):
          inps = [inp if not is_bound else x
                  for inp, is_bound in zip(in_vals, inp_is_bound)]
          return dvar * vlin_fun(inps)
        lagrangian_dict[in_vals[bound_arg_pos].index].append(
            ('Linear', single_input_vlin_fun))
      else:
        # There is multiple inputs, so we need to separate the contribution of
        # each one, and assign the bias to one of them.
        inps = []
        for inp in in_vals:
          if isinstance(inp, nonconvex.NonConvexBound):
            # Add the opt dimension, and put in all the examples to 0, so that
            # we can identify the bias term.
            shape = inp.shape
            inp_shape = (dvar.shape[0],) + shape
            example_inp = jnp.zeros(inp_shape)
            inps.append(example_inp)
          else:
            inps.append(inp)
        # Get the linear term over the inputs through auto-diff
        def lag_inp_contrib(x):
          contrib = dvar * vlin_fun(x)
          contrib = _sum_over_acts(contrib)
          return contrib.sum(), contrib
        (_, bias), grads = jax.value_and_grad(lag_inp_contrib,
                                              has_aux=True)(inps)

        grad_dot_prod = lambda grad, bias, x: _sum_over_acts(grad * x) + bias
        for inp, grad in zip(in_vals, grads):
          if isinstance(inp, nonconvex.NonConvexBound):
            lagrangian_dict[inp.index].append(
                ('Linear', functools.partial(grad_dot_prod, grad, bias)))
            # Zero out the bias now that it has been included in one term.
            bias = 0. * bias

    return functools.partial(cls, lagrangian_varterms_fn)

  @classmethod
  def get_nonlinearity_activation_constructor(
      cls: Callable[..., 'MinLagrangianNonConvexBound'],
      index: Index,
      act_type: bound_propagation.Primitive,
      lb_fun: Callable[[Tensor], Tensor],
      ub_fun: Callable[[Tensor], Tensor],
      *inp: 'MinLagrangianNonConvexBound',
      ) -> Callable[..., 'MinLagrangianNonConvexBound']:
    assert len(inp) == 1
    assert act_type in _lagrangian_opt_fns
    inp = inp[0]
    def lagrangian_varterms_fn(dvar: Tensor, lagrangian_dict: LagrangianDict):
      # There is a linear term of dvar over the outputs.
      lagrangian_dict[index].append(('Linear', lambda x: (-dvar*x)))
      # For the inputs, there is a linear term through the upper bound:
      pos_dvar = jnp.maximum(dvar, 0.)
      neg_dvar = jnp.maximum(-dvar, 0.)
      negdvar_dot_ub = lambda x: (-neg_dvar * ub_fun(x))
      lagrangian_dict[inp.index].append(('Linear', negdvar_dot_ub))

      # For the inputs, there is a ReLU term through the lower bound
      lagrangian_dict[inp.index].append(
          (act_type, lambda x: (pos_dvar * lb_fun(x))))
    return functools.partial(cls, lagrangian_varterms_fn)

  def requires_concretizing(self, consuming_primitive):
    return self._concretized_bounds is None


def _optimize_lagrangian_terms(lagrangian_terms: List[LagrangianVarTerm],
                               lower_bound: Tensor,
                               upper_bound: Tensor) -> Tensor:
  """Minimize the part of the lagrangian corresponding to a given variable.

  Args:
    lagrangian_terms: A list of the terms involving that variable.
    lower_bound: A tensor with the lower bound on the variable to optimize.
    upper_bound: A tensor with the upper bound on the variable to optimize.
  Returns:
    opt_act: A tensor with the inputs minimizing the lagrangian terms for each
      optimization target.
  """
  act_term = None

  # Get the total linear term
  def linear_term(x):
    out = 0
    for act_type, lag_term_fn in lagrangian_terms:
      if act_type == 'Linear':
        out += lag_term_fn(x).sum()
    return out

  # Identify the NonLinear term if there is one
  for act_type, lag_term_fn in lagrangian_terms:
    if act_type in _lagrangian_opt_fns:
      if act_term is not None:
        raise ValueError('Variable involved in several activations.')
      act_term = act_type, lag_term_fn
    elif act_type == 'Linear':
      pass
    else:
      raise ValueError('Unexpected contribution: ' + act_type)

  # Perform the minimization
  lin_coeffs = jax.grad(linear_term)(lower_bound)
  if act_term is None:
    # This does not involve a non linearity; this is just a linear term.
    return jnp.where(lin_coeffs >= 0, lower_bound, upper_bound)
  else:
    act_type, lag_term_fn = act_term
    return _lagrangian_opt_fns[act_type](
        lin_coeffs, lag_term_fn, lower_bound, upper_bound)


def _optimize_softplus_lagrangian(lin_coeffs: Tensor,
                                  nonlin_term: Callable[[Tensor], Tensor],
                                  lower_bound: Tensor,
                                  upper_bound: Tensor) -> Tensor:
  """Compute the input minimizing a sum of a linear term and a softplus.

  To minimize a * softplus(x) + b  * x
  Either cancel gradient is feasible:
         a * (1 / (1 + exp(-x))) + b = 0
   <=>   a + b * (1 + exp(-x))       = 0
   <=>   - (a + b) / b               = exp(-x)
   <=>   x = ln(- b / (a + b))
  If b=0, this is just normal linear minimization.
  If b / (a + b) > 0, that means there is no point where the gradient
    cancels, which means that the minimum will be obtained at one of the
    extremum. We can simply do linear minimization with the gradient.
  Otherwise, the minimum is for x = ln(-b / (a+b)), clipped to valid bounds.

  Args:
    lin_coeffs: b in the previous equation.
    nonlin_term: x -> a * softplus(x)
    lower_bound: Lower bound on the input we're minimizing over.
    upper_bound: Upper bound on the input we're minimizing over.
  Returns:
    opt_act: A tensor with the inputs minimizing the function specified.
  """
  # Get the coefficients on the softplus
  dummy_inp = jnp.ones_like(lower_bound)
  softplus_coeffs = nonlin_term(dummy_inp) / jax.nn.softplus(dummy_inp)
  grad_at_lb = lin_coeffs + softplus_coeffs * jax.nn.sigmoid(lower_bound)
  # Check condition where we can disregard the 0-gradient solution
  safe_denom = jnp.where(lin_coeffs + softplus_coeffs != 0,
                         lin_coeffs + softplus_coeffs, 1e-12)
  inner_log = -lin_coeffs / safe_denom
  safe_inner_log = jnp.where(inner_log > 0,
                             inner_log, jnp.ones_like(inner_log))
  zero_grad_infeasible = jnp.any(
      jnp.stack([(lin_coeffs + jnp.zeros_like(softplus_coeffs)) == 0,
                 lin_coeffs + softplus_coeffs == 0,
                 inner_log <= 0], axis=0), axis=0)
  return jnp.where(zero_grad_infeasible,
                   jnp.where(grad_at_lb >= 0, lower_bound, upper_bound),
                   jnp.clip(jnp.log(safe_inner_log),
                            a_min=lower_bound, a_max=upper_bound))


def _optimize_relu_lagrangian(lin_coeffs: Tensor,
                              nonlin_term: Callable[[Tensor], Tensor],
                              lower_bound: Tensor,
                              upper_bound: Tensor) -> Tensor:
  """Compute the input minimizing a sum of a linear term and a ReLU.

  To minimize a * relu(x) + b * x,
  We know that the function is piecewise linear. We will stack the three
  possible solutions along axis = 0 and then keep the minimum one.

  Args:
    lin_coeffs: b in the previous equation.
    nonlin_term: x -> a * relu(x)
    lower_bound: Lower bound on the input we're minimizing over.
    upper_bound: Upper bound on the input we're minimizing over.
  Returns:
    opt_act: A tensor with the inputs minimizing the function specified.
  """
  zero_inp = jnp.zeros_like(lower_bound)
  possible_inps = jnp.stack([
      lower_bound,
      jnp.clip(zero_inp, a_min=lower_bound, a_max=upper_bound),
      upper_bound], axis=0)
  out_val = lin_coeffs * possible_inps + nonlin_term(possible_inps)
  choice = out_val.argmin(axis=0)
  return jnp.choose(choice, possible_inps, mode='clip')


_lagrangian_opt_fns = {
    synthetic_primitives.relu_p: _optimize_relu_lagrangian,
    synthetic_primitives.softplus_p: _optimize_softplus_lagrangian,
}
