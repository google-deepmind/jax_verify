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

# Lint as: python3
# pylint: disable=invalid-name
"""Neural network verification with cvxpy for correctness checks."""

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
import jax.numpy as jnp
from jax_verify.src.sdp_verify import utils
import numpy as np
import scipy


def solve_mip_mlp_elided(verif_instance):
  """Compute optimal attack loss for MLPs, via exactly solving MIP."""
  assert MIP_SOLVERS, 'No MIP solvers installed with cvxpy.'
  assert verif_instance.type == utils.VerifInstanceTypes.MLP_ELIDED
  params, bounds, obj, obj_const = (
      verif_instance.params, verif_instance.bounds, verif_instance.obj,
      verif_instance.const)
  layer_sizes = utils.mlp_layer_sizes(params)
  on_state = []
  post_activations = [cp.Variable((1, layer_sizes[0]))]
  pre_activations = []
  constraints = []

  for (i, param) in enumerate(params):
    W, b = param
    b = jnp.reshape(b, (1, b.size))
    on_state.append(cp.Variable((1, b.size), boolean=True))
    pre_activations.append(cp.Variable((1, b.size)))
    post_activations.append(cp.Variable((1, b.size)))

    # Linear relaxation of ReLU constraints
    constraints += [pre_activations[-1] == post_activations[-2]@W + b]
    constraints += [post_activations[-1] >= pre_activations[-1]]
    constraints += [post_activations[-1] >= 0]

    # If ReLU is off, post activation is non-positive. Otherwise <= ub
    constraints += [post_activations[-1] <= cp.multiply(on_state[-1],
                                                        bounds[i+1].ub)]

    # If ReLU is off, pre-activation is non-positive. Otherwise <= ub_pre
    constraints += [pre_activations[-1] <= cp.multiply(on_state[-1],
                                                       bounds[i+1].ub_pre)]

    # If ReLU is on, post-activation == pre-activation
    # Define <= here, >= constraint added above.
    constraints += [post_activations[-1]-pre_activations[-1] <=
                    cp.multiply(1-on_state[-1],
                                bounds[i+1].ub-bounds[i+1].lb_pre)]

  # Optionally, include IBP bounds to speed up MIP solving
  # Post activations are within bounds
  # i=0 case encodes input constraint
  for (i, post) in enumerate(post_activations):
    constraints += [post <= bounds[i].ub]
    constraints += [post >= bounds[i].lb]

  # # Pre activations are within bounds
  for (i, pre) in enumerate(pre_activations):
    constraints += [pre <= bounds[i+1].ub_pre]
    constraints += [pre >= bounds[i+1].lb_pre]

  # Set objective over final post-activations
  obj_cp = cp.sum(cp.multiply(obj, post_activations[-1]))

  # Define and solve problem
  problem = cp.Problem(cp.Maximize(obj_cp), constraints)
  # NB: Originally, we used cp.ECOS_BB here, but cvxpy 1.1 drops support,
  # so we just use the first available MIP solver (which is dependent on user
  # installation).
  problem.solve(solver=MIP_SOLVERS[0])

  # Report results
  info = {
      'problem': problem,
      'post': post_activations,
      'pre': pre_activations,
  }
  return obj_cp.value + obj_const, info


def solve_lp_primal_elided(verif_instance):
  """Compute optimal attack loss for MLPs against LP relaxation."""
  assert verif_instance.type == utils.VerifInstanceTypes.MLP_ELIDED
  params, bounds, obj, obj_const = (
      verif_instance.params, verif_instance.bounds, verif_instance.obj,
      verif_instance.const)
  layer_sizes = utils.mlp_layer_sizes(params)
  post_activations = [cp.Variable((1, layer_sizes[0]))]
  constraints = []

  for (i, param) in enumerate(params):
    W, b = param
    b = jnp.reshape(b, (1, b.size))
    post_activations.append(cp.Variable((1, b.size)))
    pre_act = post_activations[-2]@W + b
    post_act = post_activations[-1]

    # Linear relaxation of ReLU constraints
    constraints += [post_act >= pre_act]
    constraints += [post_act >= 0]

    # Triangle relaxation
    l = np.minimum(0., bounds[i+1].lb_pre)
    u = np.maximum(0., bounds[i+1].ub_pre)
    constraints += [cp.multiply(u, pre_act) - cp.multiply(u, l) -
                    cp.multiply(u - l, post_act) >= 0]

  # Optionally, include IBP bounds to speed up MIP solving
  # Post activations are within bounds
  # i=0 case encodes input constraint
  for (i, post) in enumerate(post_activations[:1]):
    constraints += [post <= bounds[i].ub]
    constraints += [post >= bounds[i].lb]

  # Set objective over final post-activations
  obj_cp = cp.sum(cp.multiply(obj, post_activations[-1]))

  # Define and solve problem
  problem = cp.Problem(cp.Maximize(obj_cp), constraints)
  problem.solve(solver=cp.ECOS)

  # Report results
  info = {
      'problem': problem,
      'post': post_activations,
  }
  return obj_cp.value + obj_const, info


def solve_sdp_mlp_elided(verif_instance, solver_name='SCS', verbose=False,
                         check_feasibility=False, feasibility_margin=0.0):
  """Compute exact SDP relaxation verified bound, following Raghunathan 18.

  Args:
    verif_instance: VerifInstance namedtuple
    solver_name: string, SDP solver, either 'SCS' or 'CVXOPT'
    verbose: bool, controls verbose output from SDP solver
    check_feasibility: bool, if True, try to find any verified certificate,
      rather than tightest possible lower bound
    feasibility_margin: float, when `check_feasibility=True`, verify that
      adversary cannot decrease objective below `feasibility_margin`.

  Returns:
    obj_value: either a float, the bound on objective (check_feasibility=False),
      or a bool, whether verification succeeded (check_feasibility=True)
    info: dict of other info, e.g. solver status, values found by solver
  """
  assert verif_instance.type == utils.VerifInstanceTypes.MLP_ELIDED
  params, input_bounds, bounds, obj, obj_const = (
      verif_instance.params, verif_instance.input_bounds, verif_instance.bounds,
      verif_instance.obj, verif_instance.const)
  layer_sizes = utils.mlp_layer_sizes(params)
  assert len(bounds) == len(layer_sizes) + 1
  # Matrix P, where P = vv' before SDP relaxation, and v = [1, x_0, ..., x_L]
  P = cp.Variable((1 + sum(layer_sizes), 1 + sum(layer_sizes)))

  # Matrix constraints
  constraints = [
      P == P.T,
      P >> 0,
      P[0][0] == 1.,
  ]

  cumsum_sizes = [0] + list(np.cumsum(layer_sizes))
  def _slice(i):
    """Helper method for `p_slice`."""
    if i == -1:
      return 0
    else:
      return slice(1 + cumsum_sizes[i], 1 + cumsum_sizes[i+1])

  def p_slice(i, j):
    """Symbolic indexing into matrix P.

    Args:
      i: an integer, either -1, or in [0, num_layers).
      j: an integer, either -1, or in [0, num_layers).

    Returns:
      slice object, used to index into P. In the QCQP, if P = vv', where
        v = [1, x_0, ..., x_L], then p_slice(i, j) gives submatrix corresponding
        to P[x_i x_j']. When j = -1, this returns P[x_i].
    """
    return P[_slice(i), _slice(j)]
  diag = cp.atoms.affine.diag.diag

  # Input/IBP constraints
  # TODO: Check if these are actually necessary
  if input_bounds is not None:
    constraints += [p_slice(0, -1) >= input_bounds[0]]
    constraints += [p_slice(0, -1) <= input_bounds[1]]

  for i in range(len(layer_sizes)):
    lb = bounds[i].lb[0]
    ub = bounds[i].ub[0]
    assert lb.shape == ub.shape == (layer_sizes[i],)
    assert diag(p_slice(i, i)).shape == (layer_sizes[i],)
    assert p_slice(i, -1).shape == (layer_sizes[i],)
    constraints += [diag(p_slice(i, i)) <=
                    cp.multiply(lb + ub, p_slice(i, -1)) - cp.multiply(lb, ub)]

  # Relu / weight constraints
  for i, param in enumerate(params):
    W, b = param
    constraints += [p_slice(i+1, -1) >= 0]
    constraints += [p_slice(i+1, -1) >= p_slice(i, -1)@W + b]
    # Encode constraint P[zz'] = WP[xz']. Since our networks use z=xW+b rather
    # than z=Wx+b, we use W'x = xW (which holds when x is a vector)
    constraints += [
        diag(p_slice(i+1, i+1)) ==
        diag(W.T@p_slice(i, i+1)) + cp.multiply(b, p_slice(i+1, -1))]

  # Set objective over final post-activations
  final_idx = len(layer_sizes)-1
  x_final = P[_slice(final_idx), _slice(-1)]
  obj_cp = cp.sum(cp.multiply(obj[0], x_final))

  # Define and solve problem
  if check_feasibility:
    constraints += [obj_cp + obj_const >= feasibility_margin]
    problem = cp.Problem(cp.Maximize(cp.Constant(0.)), constraints)
  else:
    problem = cp.Problem(cp.Maximize(obj_cp), constraints)
  solver = getattr(cp, solver_name)
  problem.solve(solver=solver, verbose=verbose)

  # Report results
  info = {
      'problem': problem,
      'P': P,
      'constraints': constraints,
  }
  print('status', problem.status)
  if check_feasibility:
    # If solver shows problem is infeasible for adversary, this is a certificate
    obj_value = problem.status == 'infeasible'
  else:
    obj_value = obj_cp.value + obj_const if obj_cp.value is not None else -99999
  return obj_value, info


def _violation(arr):
  return np.maximum(0, (np.max(arr)))


def _violation_leq(arr1, arr2):
  """Get violation for constraint `arr1 <= arr2`."""
  return _violation(arr1 - arr2)


def check_sdp_bounds_numpy(P, verif_instance, input_bounds=(0, 1)):
  """Check SDP solution for 1-hidden MLP satisfies constraints in numpy."""
  params, bounds, obj, const = (
      verif_instance.params, verif_instance.bounds, verif_instance.obj,
      verif_instance.const)
  layer_sizes = utils.mlp_layer_sizes(params)
  assert len(layer_sizes) == 2, 'Relu MLP with 1 hidden layer'
  assert len(params) == 1, 'Relu MLP with 1 hidden layer'
  assert P.shape == (1+sum(layer_sizes), 1+sum(layer_sizes))
  violations = {}
  # Matrix constraints
  violations['P = P.T'] = _violation(np.abs(P - P.T))
  violations['P[0][0] = 1'] = abs(P[0][0] - 1.0)
  eig_vals, _ = scipy.linalg.eigh(P)
  violations['P >= 0 (SDP)'] = _violation(-eig_vals)

  x = P[0, 1:1+layer_sizes[0]]
  z = P[0, 1+layer_sizes[0]:]
  xx = P[1:1+layer_sizes[0], 1:1+layer_sizes[0]]
  xz = P[1:1+layer_sizes[0], 1+layer_sizes[0]:]
  zz = P[1+layer_sizes[0]:, 1+layer_sizes[0]:]

  # Relu constraints
  w, b = params[0]
  violations['relu_0'] = _violation_leq(0, z)
  violations['relu_wx_b'] = _violation_leq(np.matmul(x, w) + b, z)
  violations['relu_eq'] = _violation(np.abs(
      np.diag(np.matmul(w.T, xz)) + b*z - np.diag(zz)))

  # Input bound constraints
  violations['input_lb'] = _violation_leq(input_bounds[0], x)
  violations['input_ub'] = _violation_leq(x, input_bounds[1])

  # Interval bound constraints
  for i in range(len(layer_sizes)):
    lb = bounds[i].lb[0]
    ub = bounds[i].ub[0]
    x_slice = slice(1+sum(layer_sizes[:i]), 1+sum(layer_sizes[:i+1]))
    x = P[0, x_slice]
    xx = P[x_slice, x_slice]
    violations[f'lay{i}_bound'] = _violation_leq(np.diag(xx), (lb+ub)*x - lb*ub)

  # Objective
  obj = const + np.sum(obj * z)
  return obj, violations
