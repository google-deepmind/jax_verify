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

"""Verification configuration."""

import ml_collections


def get_input_uncertainty_config(num_layers=2):
  """Running mixed solver strategy."""
  config = ml_collections.ConfigDict()
  config.train = ml_collections.ConfigDict()
  config.train.optim_type = 'mixed'
  u_config = {
      'optim_type': 'uncertainty',
      'solve_max': 'exp',
      'n_iter': 20,
      'n_pieces': 30,
      'learning_rate': 1.
  }
  solver_config_input_init = {
      'optim_type': 'uncertainty_input',
      'layer_type': 'input',
      'sig_max': .1
  }
  solver_config_input_first = {
      'optim_type': 'uncertainty_input',
      'layer_type': 'first',
      'sig_max': .1
  }
  solver_config = {'optim_type': 'lp'}
  config.train.mixed_strat = (
      [[solver_config_input_init], [solver_config_input_first]] +
      [[solver_config]] * num_layers + [[u_config]])
  config.train.solver_weights = [[1.0]] * (num_layers + 3)
  u_config_eval = {
      'optim_type': 'uncertainty',
      'n_iter': 0,
      'n_pieces': 100,
      'solve_max': 'exp_bound',
  }

  config.eval = ml_collections.ConfigDict()
  config.eval.optim_type = 'mixed'
  config.eval.mixed_strat = (
      [[solver_config_input_init], [solver_config_input_first]] +
      [[solver_config]] * num_layers + [[u_config_eval]])
  config.eval.solver_weights = [[1.0]] * (num_layers + 3)
  return config


def get_dual_config():
  """Dual config."""
  config = ml_collections.ConfigDict()
  names = ['linear_exp', 'linear', 'linear', 'linear', 'linear']
  config.lagrangian_form = []
  for name in names:
    config.lagrangian_form.append(
        ml_collections.ConfigDict({
            'name': name,
            'kwargs': {},
        }))

  config.affine_before_relu = False

  return config


def get_attack_config():
  """Attack config."""
  # Config to use for adversarial attak lower bound
  config = ml_collections.ConfigDict()
  config.num_steps = 200
  config.learning_rate = 1.

  return config


def get_config():
  """Main configdict."""

  config = ml_collections.ConfigDict()

  config.assets_dir = '/tmp/jax_verify'  # directory to download data and models

  config.seed = 23
  config.use_gpu = True
  config.spec_type = 'uncertainty'
  config.labels_in_distribution = []
  config.use_best = True

  config.problem = ml_collections.ConfigDict()
  config.problem.dataset = 'emnist_CEDA'
  config.problem.dataset_idx = 0  # which example from dataset to verify?
  config.problem.target_label_idx = 4  # which class to target?
  config.problem.epsilon_unprocessed = 0.04  # radius before preprocessing
  config.problem.probability_threshold = .97
  config.problem.input_shape = (28, 28, 1)
  # Use inception_preprocessing i.e. [-1,1]-scaled inputs
  config.problem.scale_center = False
  config.problem.model_name = 'mnist_ceda'

  # check adversary cannot bring loss below feasibility_margin
  config.problem.feasibility_margin = 0.0

  config.add_input_noise = True
  config.dual = get_dual_config()
  config.attack = get_attack_config()

  # whether to block asynchronous dispatch at each iteration for precise timing
  config.block_to_time = False

  # Choose boundprop method: e.g. 'nonconvex', 'ibp', 'crown_ibp'
  config.boundprop_type = 'nonconvex'
  config.bilinear_boundprop_type = 'ibp'

  # nonconvex boundprop params, only used if config.boundprop_type = 'nonconvex'
  config.nonconvex_boundprop_steps = 100
  config.nonconvex_boundprop_nodes = 128

  config.outer_opt = ml_collections.ConfigDict()
  config.outer_opt.lr_init = 1e-4  # initial learning rate
  config.outer_opt.steps_per_anneal = 10  # steps between each anneal
  config.outer_opt.anneal_lengths = '60000, 20000, 20000'  # steps per epoch
  config.outer_opt.anneal_factor = 0.1  # learning rate anneal factor
  config.outer_opt.num_anneals = 2  # # of times to anneal learning rate
  config.outer_opt.opt_name = 'adam'  # Optix class: "adam" "sgd", "rmsprop"
  config.outer_opt.opt_kwargs = {}  # Momentum for gradient descent'

  config.inner_opt = get_input_uncertainty_config()
  return config
