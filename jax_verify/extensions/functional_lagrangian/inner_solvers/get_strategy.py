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

"""Inner solvers."""

from jax_verify.extensions.functional_lagrangian.inner_solvers import input_uncertainty_spec
from jax_verify.extensions.functional_lagrangian.inner_solvers import lp
from jax_verify.extensions.functional_lagrangian.inner_solvers import mixed
from jax_verify.extensions.functional_lagrangian.inner_solvers import pga
from jax_verify.extensions.functional_lagrangian.inner_solvers import uncertainty_spec


def get_strategy(config, params, mode):
  """Returns configured strategy for inner maximisation."""

  return _build_strategy_recursively(config.inner_opt.get(mode), params)


def _build_strategy_recursively(config_inner_opt, params):
  """Create inner solver strategy (potentially recursively)."""

  optim_type = config_inner_opt['optim_type']

  if optim_type == 'pga':
    strategy = pga.PgaStrategy(
        n_iter=config_inner_opt['n_iter'],
        lr=config_inner_opt['lr'],
        n_restarts=config_inner_opt['n_restarts'],
        method=config_inner_opt['method'],
        finetune_n_iter=config_inner_opt['finetune_n_iter'],
        finetune_lr=config_inner_opt['finetune_lr'],
        finetune_method=config_inner_opt['finetune_method'],
        normalize=config_inner_opt['normalize'])
  elif optim_type == 'lp':
    strategy = lp.LpStrategy()
  elif optim_type == 'probability_threshold':
    strategy = input_uncertainty_spec.ProbabilityThresholdSpecStrategy()
  elif optim_type == 'uncertainty':
    solve_max = {f.value: f for f in uncertainty_spec.MaxType
                }[config_inner_opt.get('solve_max')]
    strategy = uncertainty_spec.UncertaintySpecStrategy(
        n_iter=config_inner_opt.get('n_iter'),
        n_pieces=config_inner_opt.get('n_pieces'),
        solve_max=solve_max,
        learning_rate=config_inner_opt.get('learning_rate'),
    )
  elif optim_type == 'uncertainty_input':
    layer_type = {f.value: f for f in input_uncertainty_spec.LayerType
                 }[config_inner_opt.get('layer_type')]
    sig_max = config_inner_opt.get('sig_max')
    strategy = input_uncertainty_spec.InputUncertaintySpecStrategy(
        layer_type=layer_type, sig_max=sig_max)
  elif optim_type == 'mixed':
    solvers = [[
        _build_strategy_recursively(strat, params) for strat in strats_for_layer
    ] for strats_for_layer in config_inner_opt['mixed_strat']]
    strategy = mixed.MixedStrategy(
        solvers=solvers, solver_weights=config_inner_opt['solver_weights'])
  else:
    raise NotImplementedError(
        f'Unsupported optim type {config_inner_opt["optim_type"]}')

  if (any(p.has_bounds for p in params) and
      not strategy.supports_stochastic_parameters()):
    # this is a conservative check: we fail if *any* parameter is
    # stochastic, although it might not actually be used by strategy
    raise ValueError('Inner opt cannot handle stochastic parameters.')

  return strategy
