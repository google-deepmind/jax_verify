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

# Lint as: python3
r"""Run SDP verification for adversarial robustness specification.

Example launch commands which achieve good results:

CIFAR10 CNN-Mix:
python3 run_sdp_verify.py --model_name=models/cifar10_wongsmall_eps2_mix.pkl \
  --anneal_lengths=30000,30000,30000

MNIST CNN-Adv:
python3 run_sdp_verify.py --model_name=models/mnist_wongsmall_eps_10_adv.pkl \
  --epsilon=0.1 --dataset=mnist \
  --anneal_lengths=20000,20000,20000 --opt_name=adam --anneal_factor=0.03 \
  --n_iter_lanczos=300

MNIST Adv-MLP:
python3 run_sdp_verify.py --epsilon=0.1 --dataset=mnist \
  --model_name=models/raghunathan18_pgdnn.pkl --use_exact_eig_train=True \
  --use_exact_eig_eval=True --opt_name=adam --lam_coeff=0.1 --nu_coeff=0.03 \
  --custom_kappa_coeff=10000 --anneal_lengths=10000,4000,1000 \
  --kappa_zero_after=2000
"""


import functools
import os
import pickle

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import jax_verify
from jax_verify import sdp_verify
from jax_verify.extensions.sdp_verify import boundprop_utils
from jax_verify.extensions.sdp_verify import problem
from jax_verify.extensions.sdp_verify import utils
import numpy as np

flags.DEFINE_integer('dataset_idx', 1, 'i^th example in dataset')
flags.DEFINE_integer('target_label_idx', 0, 'which class to target?')
flags.DEFINE_float('epsilon', 2. / 255, 'attack radius')
flags.DEFINE_string('dataset', 'cifar10', 'dataset, mnist or cifar')
flags.DEFINE_string('model_name', 'models/cifar10_wongsmall_eps2_mix.pkl',
                    'model name specifying Pickle file with network weights')
flags.DEFINE_boolean('inception_preprocess', False,
                     'Use inception_preprocessing i.e. [-1,1]-scaled inputs')
flags.DEFINE_string('boundprop_type', 'crown_ibp',
                    'Method for obtaining initial activation bounds. '
                    'E.g. "crown_ibp" "nonconvex" or "ibp"')
flags.DEFINE_float('lam_coeff', 1.0, 'Coeff for dual variables')
flags.DEFINE_float('nu_coeff', 0.03, 'Coeff for dual variables')
flags.DEFINE_float('custom_kappa_coeff', -1,
                   'if >0, scale LR for top-left kappa')
flags.DEFINE_string('anneal_lengths', '15,5',
                    'comma-separated integers with # of steps per epoch')

# Flags passed directly to solver
flags.DEFINE_boolean('use_exact_eig_train', False,
                     'Use exact eigendecomposition for training')
flags.DEFINE_boolean('use_exact_eig_eval', False,
                     'Use exact eigendecomposition for evaluation')
flags.DEFINE_integer('n_iter_lanczos', 200, '# of Lanczos iters per step')
flags.DEFINE_float('eval_every', 1000, 'Iterations per log.')
flags.DEFINE_float('lr_init', 1e-3, 'initial learning rate')
flags.DEFINE_float('anneal_factor', 0.1, 'learning rate anneal factor')
flags.DEFINE_string('opt_name', 'rmsprop',
                    'Optix class: "adam" "sgd" or "rmsprop"')
flags.DEFINE_float('kappa_zero_after', 1e9, 'zero kappa_{1:n} after N steps')
flags.DEFINE_float('kappa_reg_weight', -1, '-1 disables kappa regularization')
FLAGS = flags.FLAGS


def _load_dataset(dataset):
  """Loads the 10000 MNIST (CIFAR) test set examples, saved as numpy arrays."""
  assert dataset in ('mnist', 'cifar10'), 'invalid dataset name'
  with jax_verify.open_file(os.path.join(dataset, 'x_test.npy'), 'rb') as f:
    xs = np.load(f)
  with jax_verify.open_file(os.path.join(dataset, 'y_test.npy'), 'rb') as f:
    ys = np.load(f)
  return xs, ys


def _load_weights(path):
  with jax_verify.open_file(path, 'rb') as f:
    data = pickle.load(f)
  return data


def get_verif_instance(params, x, label, target_label, epsilon,
                       input_bounds=(0., 1.)):
  """Creates verif instance."""
  if FLAGS.boundprop_type == 'ibp':
    bounds = utils.boundprop(
        params, utils.init_bound(x, epsilon, input_bounds=input_bounds))
  else:
    bounds = boundprop_utils.boundprop(
        params, np.expand_dims(x, axis=0), epsilon, input_bounds,
        FLAGS.boundprop_type)
  verif_instance = utils.make_relu_robust_verif_instance(
      params, bounds, target_label=target_label, label=label,
      input_bounds=input_bounds)
  return verif_instance


def _opt_multiplier_fn(path, kappa_index, kappa_dim=None):
  """Set adaptive learning rates."""
  if FLAGS.custom_kappa_coeff > 0:
    kappa_lr_mul = FLAGS.custom_kappa_coeff
    if kappa_index in path:
      onehot = jax.nn.one_hot([0], kappa_dim)
      return onehot.at[(0, 0)].set(kappa_lr_mul)
  if 'lam' in path:
    return FLAGS.lam_coeff
  if path == (kappa_index - 1, 'nu'):
    return FLAGS.nu_coeff
  return 1.0


def verify_cnn_single_dual(verif_instance):
  """Run verification for a CNN on a single MNIST/CIFAR problem."""
  verif_instance = problem.make_sdp_verif_instance(verif_instance)
  solver_params = dict(
      use_exact_eig_train=FLAGS.use_exact_eig_train,
      use_exact_eig_eval=FLAGS.use_exact_eig_eval,
      n_iter_lanczos=FLAGS.n_iter_lanczos,
      eval_every=FLAGS.eval_every,
      opt_name=FLAGS.opt_name,
      anneal_factor=FLAGS.anneal_factor,
      lr_init=FLAGS.lr_init,
      kappa_zero_after=FLAGS.kappa_zero_after,
      kappa_reg_weight=FLAGS.kappa_reg_weight,
  )
  # Set schedule
  steps_per_anneal = [int(x) for x in FLAGS.anneal_lengths.split(',')]
  num_steps = sum(steps_per_anneal)
  solver_params['steps_per_anneal'] = steps_per_anneal[:-1] + [int(1e9)]

  # Set learning rate multipliers
  kappa_shape = verif_instance.dual_shapes[-1]
  kappa_index = len(verif_instance.dual_shapes) - 1
  assert len(kappa_shape) == 2 and kappa_shape[0] == 1
  opt_multiplier_fn = functools.partial(
      _opt_multiplier_fn, kappa_index=kappa_index, kappa_dim=kappa_shape[1])

  # Call solver
  obj_value, info = sdp_verify.solve_sdp_dual(
      verif_instance,
      num_steps=num_steps,
      verbose=True,
      opt_multiplier_fn=opt_multiplier_fn,
      **solver_params)
  info['final_dual_vars'] = jax.tree_map(np.array, info['final_dual_vars'])
  return float(obj_value), info


class PickleWriter:

  def write(self, d):
    with open('/tmp/run_sdp_verify_results.pkl', 'wb') as f:
      pickle.dump(d, f)


def main(unused_argv):
  run_verification(PickleWriter())


def run_verification(writer):
  """Run verification."""
  xs, ys = _load_dataset(FLAGS.dataset)
  dataset_idx = FLAGS.dataset_idx
  if FLAGS.dataset == 'cifar10':
    x = utils.preprocess_cifar(xs[dataset_idx])
    epsilon, input_bounds = utils.preprocessed_cifar_eps_and_input_bounds(
        shape=x.shape, epsilon=FLAGS.epsilon,
        inception_preprocess=FLAGS.inception_preprocess)
  else:
    x = xs[dataset_idx]
    epsilon = FLAGS.epsilon
    input_bounds = (0., 1.)
  true_label = ys[dataset_idx]
  target_label = FLAGS.target_label_idx
  params = _load_weights(FLAGS.model_name)
  if isinstance(params[0], dict):
    params[0]['input_shape'] = x.shape[0]

  verif_instance = get_verif_instance(
      params, x, label=true_label, target_label=target_label,
      epsilon=epsilon, input_bounds=input_bounds)

  # Report initial bound from interval bounds.
  ibp_bound = utils.ibp_bound_elided(verif_instance)
  print('IBP bound:', ibp_bound)
  if true_label == target_label:
    return

  # Run dual SDP verification.
  verified_ub, info = verify_cnn_single_dual(verif_instance)

  # Run FGSM eval.
  model_fn = lambda x: utils.predict_cnn(params, jnp.expand_dims(x, axis=0))
  x_adv = utils.fgsm_single(
      model_fn, x, true_label, target_label, input_bounds=input_bounds,
      epsilon=epsilon, num_steps=100, step_size=0.001)
  adv_objective = float(
      utils.adv_objective(model_fn, x_adv, true_label, target_label))
  print('adv_objective :', adv_objective)

  output_dict = {
      'dataset_idx': dataset_idx,
      'true_label': true_label,
      'target_label': target_label,
      'epsilon': FLAGS.epsilon,
      'verified_ub': verified_ub,
      'adv_lb': adv_objective,
      'adv_success': adv_objective > 0.0,
      'ibp_bound': ibp_bound,
  }
  output_dict.update(info)
  jax_to_np = lambda x: np.array(x) if isinstance(x, jnp.DeviceArray) else x
  output_dict = jax.tree_map(jax_to_np, output_dict)
  writer.write(output_dict)

if __name__ == '__main__':
  app.run(main)
