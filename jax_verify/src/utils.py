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

"""Utils used for JAX neural network verification."""

import os
from jax import lax
import urllib.request

######## File Loading ########

def open_file(name, *open_args, **open_kwargs):
  """Load file, downloading to /tmp/jax_verify first if necessary."""
  local_root = '/tmp/jax_verify'
  local_path = os.path.join(local_root, name)
  if not os.path.exists(os.path.dirname(local_path)):
    os.makedirs(os.path.dirname(local_path))
  if not os.path.exists(local_path):
    gcp_bucket_url = 'https://storage.googleapis.com/deepmind-jax-verify/'
    download_url = gcp_bucket_url + name
    urllib.request.urlretrieve(download_url, local_path)
  return open(local_path, *open_args, **open_kwargs)

######### Miscellaneous #########


def collect_required_arguments(req_args, all_kwargs):
  """Extract a dictionary with the required keys from a larger dictionary.

  Args:
    req_args: List of keys to extract
    all_kwargs: Dictionary with a superset of the required keys.
  Returns:
    req_args_dict: Dictionary with all the required arguments.
  """
  return {key: all_kwargs[key] for key in req_args}


def wrapped_general_conv(lhs, rhs, **kwargs):
  """Wrapped version of convolution that drop extra arguments.

  Args:
    lhs: First input to the convolution
    rhs: Second input to the convolution
    **kwargs: Dict with the parameters of the convolution with potentially
      some spurious parameters that the `lax.conv_general_dilated` would
      reject.
  Returns:
    out: Convolution output
  """
  req_arguments = ['window_strides', 'padding', 'lhs_dilation',
                   'rhs_dilation', 'dimension_numbers', 'feature_group_count',
                   'precision']
  lax_conv_params = collect_required_arguments(req_arguments, kwargs)
  return lax.conv_general_dilated(lhs, rhs, **lax_conv_params)


def simple_propagation(fn):
  """Create a wrapper function to ignore the context argument."""
  def wrapper(context, *args, **kwargs):
    del context
    return fn(*args, **kwargs)
  return wrapper
