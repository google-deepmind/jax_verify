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

#!/bin/bash
# Run examples with various flags and make sure they don't crash.
# This script is used for continuous integration testing.

set -e  # Exit on any error

echo "Running run_boundprop.py"
python3 run_boundprop.py
python3 run_boundprop.py --model=cnn
python3 run_boundprop.py --boundprop_method=interval_bound_propagation
python3 run_boundprop.py --boundprop_method=ibpfastlin_bound_propagation
python3 run_boundprop.py --boundprop_method=fastlin_bound_propagation
python3 run_boundprop.py --boundprop_method=crown_bound_propagation
python3 run_boundprop.py --boundprop_method=crownibp_bound_propagation

echo "Running run_sdp_verify.py"
python3 run_sdp_verify.py --model_name=models/cifar10_wongsmall_eps2_mix.pkl \
  --anneal_lengths="3,3"
python3 run_sdp_verify.py --epsilon=0.1 --dataset=mnist \
  --model_name=models/raghunathan18_pgdnn.pkl --use_exact_eig_train=True \
  --use_exact_eig_eval=True --opt_name=adam --lam_coeff=0.1 --nu_coeff=0.03 \
  --anneal_lengths="3,3" --custom_kappa_coeff=10000 --kappa_zero_after=2

echo "Running run_lp_solver.py"
python3 run_lp_solver.py --model=toy
