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

"""Run verification for feedforward ReLU networks."""

import os
import time
from typing import Any, Callable, Mapping

from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
from jax_verify.extensions.functional_lagrangian import attacks
from jax_verify.extensions.functional_lagrangian import bounding
from jax_verify.extensions.functional_lagrangian import data
from jax_verify.extensions.functional_lagrangian import dual_solve
from jax_verify.extensions.functional_lagrangian import model
from jax_verify.extensions.functional_lagrangian import verify_utils
from jax_verify.extensions.sdp_verify import utils as sdp_utils
import ml_collections
from ml_collections import config_flags

PROJECT_PATH = os.getcwd()

config_flags.DEFINE_config_file(
    'config', f'{PROJECT_PATH}/configs/config_ood_stochastic_model.py',
    'ConfigDict for the experiment.')

FLAGS = flags.FLAGS


def make_logger(log_message: str) -> Callable[[int, Mapping[str, Any]], None]:
  """Creates a logger.

  Args:
    log_message: description message for the logs.

  Returns:
    Function that accepts a step counter and measurements, and logs them.
  """

  def log_fn(step, measures):
    msg = f'[{log_message}] step={step}'
    for k, v in measures.items():
      msg += f', {k}={v}'
    logging.info(msg)

  return log_fn


def main(unused_argv):

  config = FLAGS.config

  logging.info('Config: \n %s', config)

  data_spec = data.make_data_spec(config.problem, config.assets_dir)
  spec_type = {e.value: e for e in verify_utils.SpecType}[config.spec_type]

  if spec_type == verify_utils.SpecType.UNCERTAINTY:
    if data_spec.true_label in config.labels_in_distribution:
      return
  else:
    if data_spec.true_label == data_spec.target_label:
      return
  params = model.load_model(
      root_dir=config.assets_dir,
      model_name=config.problem.model_name,
      num_std_for_bound=config.problem.get('num_std_for_bound'),
  )

  params_elided, bounds, bp_bound, bp_time = (
      bounding.make_elided_params_and_bounds(config, data_spec, spec_type,
                                             params))

  dual_state = ml_collections.ConfigDict(type_safe=False)

  def spec_fn(inputs):
    # params_elided is a list of network parameters, with the final
    # layer elided with the objective (output size is 1, and not num classes)
    return jnp.squeeze(sdp_utils.predict_cnn(params_elided, inputs), axis=-1)

  def run(mode: str):

    logger = make_logger(log_message=mode.title())

    start_time = time.time()
    prng_key = dual_solve.solve_dual(
        dual_state=dual_state,
        config=config,
        bounds=bounds,
        spec_type=spec_type,
        spec_fn=spec_fn,
        params=params_elided,
        mode=mode,
        logger=logger)
    elapsed_time = time.time() - start_time

    adv_objective = attacks.adversarial_attack(  # pytype: disable=wrong-arg-types  # jax-devicearray
        params, data_spec, spec_type, prng_key, config.attack.num_steps,
        config.attack.learning_rate, config.attack.get('num_samples', 1))

    output_dict = {
        'dataset_idx': config.problem.dataset_idx,
        'true_label': data_spec.true_label,
        'target_label': data_spec.target_label,
        'epsilon': config.problem.epsilon_unprocessed,
        'verified_ub': dual_state.loss,
        'verification_time': elapsed_time,
        'adv_lb': adv_objective,
        'adv_success': adv_objective > config.problem.feasibility_margin,
        'bp_bound': bp_bound,
        'bp_time': bp_time,
    }
    logger = make_logger(log_message=mode.title())
    logger(0, output_dict)

  run('train')
  run('eval')


if __name__ == '__main__':
  app.run(main)
