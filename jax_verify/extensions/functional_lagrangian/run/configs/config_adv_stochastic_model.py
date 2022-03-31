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


def get_pga_params_config_dict():
  """Create config dict with for running PGA."""
  pga_config = ml_collections.ConfigDict()
  pga_config.optim_type = 'pga'
  pga_config.n_iter = 10000
  pga_config.lr = 0.1
  pga_config.n_restarts = 300
  pga_config.method = 'square'
  pga_config.finetune_n_iter = 50
  pga_config.finetune_lr = 0.1
  pga_config.finetune_method = 'pgd'
  pga_config.normalize = False
  return pga_config


def get_adv_softmax_config(num_layers):
  """Running mixed solver strategy."""
  # Train config
  config = ml_collections.ConfigDict()
  config.train = ml_collections.ConfigDict()
  config.train.optim_type = 'mixed'

  u_config_train = get_pga_params_config_dict()

  solver_config = {'optim_type': 'lp'}
  config.train.mixed_strat = [[solver_config]] * num_layers + [[u_config_train]]
  config.train.solver_weights = [[1.0]] * (num_layers + 1)

  # Eval config
  config.eval = ml_collections.ConfigDict()
  u_config_eval = {
      'optim_type': 'uncertainty',
      'solve_max': 'exp_bound',
  }
  config.eval.optim_type = 'mixed'
  config.eval.mixed_strat = [[solver_config]] * num_layers + [[u_config_eval]]
  config.eval.solver_weights = [[1.0]] * (num_layers + 1)
  return config


def get_attack_config():
  """Attack config."""
  # Config to use for adversarial attak lower bound
  config = ml_collections.ConfigDict()
  config.num_steps = 200
  config.learning_rate = 1.
  config.num_samples = 50

  return config


def get_dual_config():
  """Dual config."""
  # type of lagrangian functional: e.g. dense_quad, mlp
  config = ml_collections.ConfigDict()
  config.lagrangian_form = ml_collections.ConfigDict({
      'name': 'linear',
      'kwargs': {},
  })

  config.affine_before_relu = False

  return config


def get_config(model_name='mnist_mlp_1_128'):
  """Main configdict."""

  if model_name.startswith('mnist_mlp_1'):
    dataset = 'mnist'
    num_layers = 2
    num_std_for_bound = 3.0
  elif model_name.startswith('mnist_mlp_2'):
    dataset = 'mnist'
    num_layers = 3
    num_std_for_bound = 3.0
  elif model_name.startswith('mnist_cnn'):
    dataset = 'mnist'
    num_layers = 5
    num_std_for_bound = None
  elif model_name.startswith('cifar_vgg'):
    dataset = 'cifar10'
    num_layers = 6
    num_std_for_bound = None

  config = ml_collections.ConfigDict()

  config.assets_dir = '/tmp/jax_verify'  # directory to download data and models

  config.seed = 23
  config.use_gpu = True
  config.spec_type = 'adversarial_softmax'
  config.labels_in_distribution = []
  config.use_best = False  # PGA may be overly optimistic

  config.problem = ml_collections.ConfigDict()
  config.problem.dataset = dataset
  config.problem.dataset_idx = 0  # which example from dataset to verify?
  config.problem.target_label_idx = 0  # which class to target?
  config.problem.epsilon_unprocessed = 0.001  # radius before preprocessing
  config.problem.scale_center = False
  config.problem.num_std_for_bound = num_std_for_bound

  # check adversary cannot bring loss below feasibility_margin
  config.problem.feasibility_margin = 0.0

  config.problem.model_name = 'mnist_mlp_1_128'

  config.dual = get_dual_config()
  config.attack = get_attack_config()

  # whether to block asynchronous dispatch at each iteration for precise timing
  config.block_to_time = False

  # Choose boundprop method: e.g. 'nonconvex', 'ibp', 'crown_ibp'
  config.boundprop_type = 'nonconvex'
  config.bilinear_boundprop_type = 'ibp'

  # nonconvex boundprop params, only used if config.boundprop_type = 'nonconvex'
  config.nonconvex_boundprop_steps = 0
  config.nonconvex_boundprop_nodes = 128

  config.outer_opt = ml_collections.ConfigDict()
  config.outer_opt.lr_init = 1e-3  # initial learning rate
  config.outer_opt.steps_per_anneal = 1000  # steps between each anneal
  config.outer_opt.anneal_lengths = ''  # steps per epoch
  config.outer_opt.anneal_factor = 0.1  # learning rate anneal factor
  config.outer_opt.num_anneals = 3  # # of times to anneal learning rate
  config.outer_opt.opt_name = 'adam'  # Optix class: "adam" "sgd", "rmsprop"
  config.outer_opt.opt_kwargs = {}  # Momentum for gradient descent'

  config.inner_opt = get_adv_softmax_config(num_layers)
  return config
