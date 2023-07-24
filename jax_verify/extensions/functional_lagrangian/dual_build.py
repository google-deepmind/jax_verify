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

"""Library functions for verification of neural networks using functional lagrange multipliers."""

import abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import lagrangian_form as lag_form
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import sdp_verify
from jax_verify.extensions.sdp_verify import utils as sdp_utils
from jax_verify.src import bound_propagation
from jax_verify.src import graph_traversal
from jax_verify.src import synthetic_primitives
from jax_verify.src.types import Nest
import numpy as np
import optax

Params = verify_utils.Params
ParamsTypes = verify_utils.ParamsTypes
InnerVerifInstance = verify_utils.InnerVerifInstance
LagrangianForm = lag_form.LagrangianForm


class DualOp(bound_propagation.Bound):
  """Lagrangian dual contribution."""

  def __init__(
      self,
      name,
      base_bound: bound_propagation.Bound,
      affine_fn: Callable[[jnp.ndarray], jnp.ndarray],
      inputs: Optional[Sequence[Union['DualOp', jnp.ndarray]]] = None,
      relu_preact_name: Optional[int] = None,
  ):
    self.name = name
    self._base_bound = base_bound
    self._affine_fn = affine_fn
    self._inputs = inputs
    self._relu_preact_name = relu_preact_name

  @property
  def base_bound(self) -> bound_propagation.Bound:
    return self._base_bound

  @property
  def lower(self) -> jnp.ndarray:
    return self._base_bound.lower

  @property
  def upper(self) -> jnp.ndarray:
    return self._base_bound.upper

  @property
  def shape(self) -> Sequence[int]:
    return self._base_bound.lower.shape

  def affine(self, act_or_input):
    return self._affine_fn(act_or_input)

  @property
  def is_input(self) -> bool:
    return self._inputs is None

  @property
  def is_relu(self) -> bool:
    return self._relu_preact_name is not None

  @property
  def relu_preact_name(self) -> int:
    if self._relu_preact_name is None:
      raise ValueError('Not an activation.')
    return self._relu_preact_name

  @property
  def inputs(self) -> Sequence[Union['DualOp', jnp.ndarray]]:
    if self._inputs is None:
      raise ValueError('Input node does not have inputs')
    return self._inputs


_affine_primitives_list = [
    *bound_propagation.AFFINE_PRIMITIVES,
    *bound_propagation.RESHAPE_PRIMITIVES,
    lax.div_p,
]


class _LagrangianTransform(graph_traversal.GraphTransform[DualOp]):
  """Identifies graph nodes having Lagrangian dual contributions."""

  def __init__(self, boundprop_transform: bound_propagation.BoundTransform):
    """Defines propagation of Lagrangian dual contributions.

    Args:
      boundprop_transform: Basic Jax primitive ops' equivalents for
        the underlying bound propagation method.
    """
    self._boundprop_transform = boundprop_transform

  def input_transform(self, context, input_bound):
    in_bounds = self._boundprop_transform.input_transform(context, input_bound)
    return DualOp(context.index, in_bounds, lambda x: x, inputs=None)

  def primitive_transform(self, context, primitive, *args, **params):
    interval_args = [arg.base_bound if isinstance(arg, DualOp) else arg
                     for arg in args]
    out_bounds, = self._boundprop_transform.equation_transform(
        context, primitive, *interval_args, **params)

    if primitive in _affine_primitives_list:
      if (primitive in bound_propagation.BILINEAR_PRIMITIVES and
          isinstance(args[0], DualOp) and isinstance(args[1], DualOp)):
        raise NotImplementedError(
            'Multiplication with two non-constant inputs is not supported')
      elif primitive == lax.div_p and isinstance(args[1], DualOp):
        raise NotImplementedError(
            f'Division with non-constant divisor {args[1]} is not supported')

      # Compose this affine primitive with the inputs' own affine functions
      # in terms of the previous ReLU activation (or original network input).
      def affine_fn(act_or_input):
        return primitive.bind(*[
            arg.affine(act_or_input) if isinstance(arg, DualOp) else arg
            for arg in args], **params)
      return DualOp(context.index, out_bounds, affine_fn, inputs=args)

    elif primitive == synthetic_primitives.relu_p:
      return DualOp(
          context.index, out_bounds, lambda x: x, inputs=args,
          relu_preact_name=args[0].name)

    else:
      raise NotImplementedError(f'Unsupported primitive: {primitive}')


class InnerMaxStrategy(metaclass=abc.ABCMeta):
  """Solve inner maximisations."""
  jittable = True

  @abc.abstractmethod
  def solve_max(
      self,
      inner_dual_vars: Any,
      opt_instance: InnerVerifInstance,
      key: jnp.ndarray,
      step: int,
  ) -> jnp.ndarray:
    """Solve maximization problem of opt_instance.

    Args:
      inner_dual_vars: Dual variables for the inner maximisation.
      opt_instance: Verification instance that defines optimization problem to
        be solved.
      key: Jax PRNG key.
      step: outer optimization iteration number.

    Returns:
      max_value: final value of the objective function found.
    """

  def supports_stochastic_parameters(self):
    return False

  def build_spec(self, opt_instance: InnerVerifInstance, step: int,
                 softmax: bool = False):
    """Build objective function for the maximization problem."""
    # affine_fns are assumed to be non-batched in both inputs and ouputs
    affine_fns = opt_instance.affine_fns

    lag_form_pre = opt_instance.lagrangian_form_pre
    lag_form_post = opt_instance.lagrangian_form_post

    def forward_relu_before_affine(x):
      # we use relu before affine ordering
      # -> first op is relu unless this is the first layer
      if not opt_instance.is_first:
        x = jax.nn.relu(x)

      # forward through intermediate layers of opt_instance
      for affine_fn in affine_fns[:-1]:
        x = affine_fn(x)
        x = jax.nn.relu(x)

      # forward through last layer of opt_instance
      x = affine_fns[-1](x)
      return x

    def forward_affine_before_relu(x):
      # forward through intermediate layers of opt_instance
      for affine_fn in affine_fns[:-1]:
        x = affine_fn(x)
        x = jax.nn.relu(x)

      # forward through last layer of opt_instance, which contains activations
      # unless it is the last layer of the network
      x = affine_fns[-1](x)
      if not opt_instance.is_last:
        x = jax.nn.relu(x)
      return x

    forward = (
        forward_affine_before_relu if opt_instance.affine_before_relu
        else forward_relu_before_affine)

    def obj_first(x, duals_pre, duals_post):
      del duals_pre  # unused
      return lag_form_post.apply(forward(x), duals_post, step)

    def obj_intermediate(x, duals_pre, duals_post):
      return (lag_form_post.apply(forward(x), duals_post, step)
              - lag_form_pre.apply(x, duals_pre, step))

    def obj_last(x, duals_pre, duals_post):
      del duals_post  # unused
      if softmax:
        y = jax.nn.softmax(x)
      else:
        y = x
      return forward(y) - lag_form_pre.apply(x, duals_pre, step)

    if opt_instance.is_first:
      return obj_first
    elif opt_instance.is_last:
      return obj_last
    else:
      return obj_intermediate

  def init_duals(
      self,
      boundprop_transform: bound_propagation.BoundTransform,
      spec_type: verify_utils.SpecType,
      affine_before_relu: bool,
      spec_fn: Callable[..., jnp.ndarray],
      key: jnp.ndarray,
      lagrangian_form_per_layer: Iterable[LagrangianForm],
      *input_bounds: Nest[graph_traversal.GraphInput],
  ) -> Tuple[Dict[int, DualOp], Params, ParamsTypes]:
    """Initialize the dual parameters and their types (Inequality vs Equality).

    Args:
      boundprop_transform: Underlying bound propagation method.
      spec_type: Type of specification, adversarial robustness, uncertainty.
      affine_before_relu: whether layer ordering uses the affine layer before
          the ReLU.
      spec_fn: Specification function to bound above.
      key: PRNGKey used while initializing trainable params.
      lagrangian_form_per_layer: Sequence of LagrangianForm
        instances whose 'init_params' function initialises the parameters of
        the layer's functional Lagrangian.
      *input_bounds: Interval bounds on the inputs of `spec_fn`.
    Returns:
      env: Lagrangian computations for each contributing graph node.
      dual_params: lagrangian parameters as 'outer', dummy params as 'inner'.
      dual_params_types: constraint types (inequality vs equality) for
        'outer' and 'inner', governing whether to project.
    """
    # Analyse the graph, propagating (or applying) bounds along the way.
    _, env = bound_propagation.bound_propagation(
        bound_propagation.ForwardPropagationAlgorithm(
            _LagrangianTransform(boundprop_transform)),
        spec_fn, *input_bounds)
    env = {
        op.name: op for op in env.values()
        if isinstance(op, DualOp)}

    make_equality_constraint = lambda s: sdp_utils.DualVarTypes.EQUALITY

    # initialize outer variables and types
    lagrangian_form = {}
    lagrange_params = {}
    lagrangian_form_iter = iter(lagrangian_form_per_layer)
    for name, op in env.items():
      if op.is_relu:
        lagrangian_form[name] = next(lagrangian_form_iter)
        key, layer_key = jax.random.split(key, 2)
        lagrange_params[name] = lagrangian_form[name].init_params(
            layer_key, op.shape[1:])
      elif op.is_input or op.name == max(env):
        # special case for first and last layers
        lagrangian_form[name] = None
        lagrange_params[name] = None

    lagrange_params_types = jax.tree_map(
        make_equality_constraint, lagrange_params)

    inner_problems = _enumerate_inner_max_problems(
        env, lagrangian_form, lagrange_params, spec_type, affine_before_relu)
    # Initialize inner variables and types
    inner_params = []
    inner_params_types = []
    for inner_problem in inner_problems:
      layer_inner_params, layer_inner_params_types = (
          self.init_layer_inner_params(inner_problem))
      inner_params.append(layer_inner_params)
      inner_params_types.append(layer_inner_params_types)

    dual_params = Params(inner=inner_params, outer=lagrange_params)
    dual_params_types = ParamsTypes(
        inner=inner_params_types, outer=lagrange_params_types,
        lagrangian_form=lagrangian_form)
    return env, dual_params, dual_params_types

  @abc.abstractmethod
  def init_layer_inner_params(
      self, opt_instance: verify_utils.InnerVerifInstance) -> Tuple[Any, Any]:
    """Initialises duals and their types for a single inner maximisation.

    Args:
      opt_instance: The context (nearby bounds and outer duals) for the
        layer's inner maximisation problem.

    Returns:
      inner_params: parameters for the 'inner' optimisation.
      inner_params_types: constraint types (inequality vs equality) for
        the 'inner' optimisation, governing whether to project.
    """


def project_dual(dual_params: Params,
                 dual_params_types: ParamsTypes) -> Params:
  """Project the dual variables."""
  projected_inner_vars = sdp_verify.project_duals(dual_params.inner,
                                                  dual_params_types.inner)
  projected_outer_vars = sdp_verify.project_duals(dual_params.outer,
                                                  dual_params_types.outer)
  new_dual_params = dual_params._replace(
      inner=projected_inner_vars, outer=projected_outer_vars)
  return new_dual_params


def build_dual_fun(
    env: Dict[int, DualOp],
    lagrangian_form: Dict[int, LagrangianForm],
    inner_opt: InnerMaxStrategy,
    affine_before_relu: bool,
    spec_type: verify_utils.SpecType,
    merge_problems: Optional[Dict[int, int]] = None,
) -> Callable[[Params, jnp.ndarray, int], jnp.ndarray]:
  """Build the dual function that takes as input the inner/outer lagrangian parameters.

  Args:
    env: Lagrangian computations for each contributing graph node.
    lagrangian_form: Dictionary, keyed by layer index, of LagrangianForm
      instances whose 'apply' function accepts hidden-layer activations and
      the parameters for the functional lagrange multplier, and returns a scalar
      value.
    inner_opt: Inner optimisation strategy.
    affine_before_relu: whether layer ordering uses the affine layer before
        the ReLU.
    spec_type: Specification type, adversarial or uncertainty specification.
    merge_problems: the key of the dictionary corresponds to the index of the
      layer to begin the merge, and the associated value corresponds to the
      number of consecutive layers to be merged with it.
      For example, `{0: 2, 2: 3}` will merge together layer 0 and 1,
      as well as layers 2, 3 and 4.

  Returns:
    A function that is a (possibly proxy) upper bound on the verification
    objective, and takes as input the inner and outer dual variables, and the
    PRNG key.
  """
  def dual_loss_fun(
      dual_params: Params, key: jnp.ndarray, step: int
  ) -> jnp.ndarray:
    lagrange_params = dual_params.outer
    inner_vars_list = dual_params.inner

    inner_problems = _enumerate_inner_max_problems(
        env, lagrangian_form, lagrange_params, spec_type, affine_before_relu)

    if merge_problems:
      inner_problems = _merge_specified_instances(
          inner_problems, merge_problems)

    # accumulate loss over inner optimization problems
    loss = 0.0
    stats = {}
    for inner_problem, inner_vars in zip(inner_problems, inner_vars_list):
      key, inner_key = jax.random.split(key, 2)
      loss_inner_problem = inner_opt.solve_max(
          inner_vars, inner_problem, key=inner_key, step=step)

      assert loss_inner_problem.ndim == 1
      # assuming batch_size of 1 for now
      loss_inner_problem = jnp.reshape(loss_inner_problem, ())

      stats[f'loss_problem_{inner_problem.idx}'] = loss_inner_problem
      loss += loss_inner_problem

    stats['loss'] = loss
    return loss, stats  # pytype: disable=bad-return-type  # jnp-array

  return dual_loss_fun


def _enumerate_inner_max_problems(
    env: Dict[int, DualOp],
    lagrangian_form: Dict[int, LagrangianForm],
    lagrange_params: Dict[int, Any],
    spec_type: verify_utils.SpecType,
    affine_before_relu: bool,
) -> List[InnerVerifInstance]:
  """Enumerates the inner maximisation problems."""
  # iteratively create inner problems: each innner problem links the
  # output of a layer to the next
  inner_problems = []
  idx = 0
  for op in env.values():
    is_last = op.name == max(env)
    if op.is_relu or is_last:
      preact_op = env[op.relu_preact_name] if op.is_relu else op
      # Search for the previous ReLU.
      prev_op = preact_op
      while not (prev_op.is_input or prev_op.is_relu):
        input_ops = [io for io in prev_op.inputs if isinstance(io, DualOp)]
        if len(input_ops) != 1:
          raise NotImplementedError('Multi-input ops not currently supported.')
        prev_op = input_ops[0]
      prev_preact_op = prev_op.inputs[0] if prev_op.is_relu else None

      # Lagrange parameters for the equality constraint just before the layer
      lagrange_params_pre = lagrange_params[prev_op.name]
      # Lagrange parameters for the equality constraint just after the layer
      lagrange_params_post = lagrange_params[op.name]

      # corresponding constraints (obtained via e.g. bound propagation)
      bounds_pre = sdp_utils.IntBound(
          lb_pre=(prev_preact_op.lower if prev_preact_op is not None
                  else prev_op.lower),
          ub_pre=(prev_preact_op.upper if prev_preact_op is not None
                  else prev_op.upper),
          lb=prev_op.lower, ub=prev_op.upper)
      bounds_post = sdp_utils.IntBound(
          lb_pre=None, ub_pre=None,  # not needed
          lb=op.lower, ub=op.upper)
      lagrangian_form_pre = lagrangian_form[prev_op.name]
      lagrangian_form_post = lagrangian_form[op.name]

      # create inner optimization problem
      opt_instance = verify_utils.InnerVerifInstance(
          affine_fns=[preact_op.affine],
          bounds=[bounds_pre, bounds_post],
          is_first=(lagrange_params_pre is None), is_last=is_last,
          lagrangian_form_pre=lagrangian_form_pre,
          lagrangian_form_post=lagrangian_form_post,
          lagrange_params_post=lagrange_params_post,
          lagrange_params_pre=lagrange_params_pre,
          idx=idx,
          spec_type=spec_type, affine_before_relu=affine_before_relu)

      # if not last layer, lagrange_params_post cannot be None
      assert(opt_instance.is_last or
             opt_instance.lagrange_params_post is not None)

      inner_problems.append(opt_instance)

      idx += 1
  if spec_type == verify_utils.SpecType.UNCERTAINTY:
    # Uncertainty spec has this layer as the logits layer
    # is_last is used to treat this layer without relu when affine_before_relu
    # flag is true
    inner_problems[-2] = dataclasses.replace(inner_problems[-2], is_last=True)
  return inner_problems


def _merge_specified_instances(
    instances: Sequence[InnerVerifInstance],
    merge_specification: Dict[int, int],
) -> Sequence[InnerVerifInstance]:
  """Merge instances according to the specified list of groups to merge."""
  merged_instances = []
  idx = 0
  merge_specification = merge_specification.copy()
  while idx < len(instances):
    run_length = merge_specification.pop(idx, 1)  # default to single
    instances_to_merge = instances[idx:(idx+run_length)]
    merged_instances.append(_merge_instances(*instances_to_merge))
    idx += run_length

  if idx > len(instances):
    raise ValueError(
        f'Invalid specification (index {idx} out of {len(instances)}).')

  if merge_specification:
    raise ValueError(
        f'Unused entry in merge_specification: {merge_specification}.')

  return merged_instances


def _merge_instances(
    instance_first: InnerVerifInstance,
    *instances_rest: InnerVerifInstance,
) -> InnerVerifInstance:
  """Merge InnerVerifInstances together."""

  if not instances_rest:
    return instance_first
  else:
    instance_second, *instances_rest = instances_rest

  if (instance_first.lagrangian_form_post
      is not instance_second.lagrangian_form_pre):
    raise ValueError(
        'Cannot merge InnerVerifInstances with different Lagrangian forms.')

  merged_instance = dataclasses.replace(
      instance_first,
      affine_fns=(instance_first.affine_fns + instance_second.affine_fns),
      bounds=(instance_first.bounds[:-1] + instance_second.bounds),
      is_first=instance_first.is_first,
      is_last=instance_second.is_last,
      # the solver corresponding to the first idx is used if using mixed strat
  )

  return _merge_instances(merged_instance, *instances_rest)


def make_opt_and_num_steps(opt_config):
  """Get optax optimizer, and number of steps to run training for."""
  if opt_config.anneal_lengths:
    print('Using custom annealing schedule', opt_config.anneal_lengths)
    steps_per_anneal = [int(x) for x in opt_config.anneal_lengths.split(',')]
    assert len(steps_per_anneal) > 1, 'for no anneals, do not use this flag'
    num_steps = sum(steps_per_anneal)
    steps_per_anneal = steps_per_anneal[:-1]
    num_anneals = len(steps_per_anneal)
    anneal_steps = np.cumsum(steps_per_anneal)
  else:
    num_anneals = opt_config.num_anneals
    num_steps = opt_config.steps_per_anneal * (1 + opt_config.num_anneals)
    anneal_steps = [
        opt_config.steps_per_anneal *
        (i + 1) for i in range(opt_config.num_anneals)
    ]
  anneal_steps = jnp.array(anneal_steps)
  def lr_schedule(t):
    cur_epoch = jnp.minimum(num_anneals,
                            jnp.sum(t > anneal_steps))
    return opt_config.lr_init * jnp.float_power(opt_config.anneal_factor,
                                                cur_epoch)

  opt_class = getattr(optax, opt_config.opt_name)
  base_opt = opt_class(1., **opt_config.opt_kwargs)
  opt = optax.chain(base_opt, optax.scale_by_schedule(lr_schedule))
  return opt, num_steps
