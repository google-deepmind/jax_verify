# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
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

import math
import os
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax_verify.src import bound_propagation
import numpy as np
import urllib.request


Tensor = bound_propagation.Tensor
Bound = bound_propagation.Bound

######## File Loading ########

def open_file(name, *open_args, root_dir='/tmp/jax_verify', **open_kwargs):
  """Load file, downloading to /tmp/jax_verify first if necessary."""
  local_path = os.path.join(root_dir, name)
  if not os.path.exists(os.path.dirname(local_path)):
    os.makedirs(os.path.dirname(local_path))
  if not os.path.exists(local_path):
    gcp_bucket_url = 'https://storage.googleapis.com/deepmind-jax-verify/'
    download_url = gcp_bucket_url + name
    urllib.request.urlretrieve(download_url, local_path)
  return open(local_path, *open_args, **open_kwargs)

######### Miscellaneous #########


def bind_nonbound_args(
    fun: Callable[..., Tensor],
    *all_in_args: Union[Bound, Tensor],
    **kwargs
) -> Callable[..., Tensor]:
  """Take a function and bind all keyword arguments and non-bound arguments."""

  def tensorbound_fun(*bound_args):
    fun_inps = []
    bound_arg_pos = 0
    for arg in all_in_args:
      if isinstance(arg, Bound):
        fun_inps.append(bound_args[bound_arg_pos])
        bound_arg_pos += 1
      else:
        fun_inps.append(arg)
    assert len(bound_args) == bound_arg_pos
    return fun(*fun_inps, **kwargs)
  return tensorbound_fun


def filter_jaxverify_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
  if 'jax_verify_keepjvargs' in kwargs and kwargs['jax_verify_keepjvargs']:
    return kwargs
  else:
    return {k: v for k, v in kwargs.items()
            if not k.startswith('jax_verify_subgraph')}


def simple_propagation(fn):
  """Create a wrapper function to ignore the context argument."""
  def wrapper(context, *args, **kwargs):
    del context
    params = filter_jaxverify_kwargs(kwargs)
    return fn(*args, **params)
  return wrapper


def batch_value_and_grad(fun, batch_dims, *args, **kwargs):
  """Equivalent to jax `value_and_grad` function but allows batched function.

  This is to go around the fact that jax.value_and_grad only supports scalar
  outputs.

  Args:
    fun: Function, operating in batch, to obtain gradients for.
    batch_dims: Dimensions to batch over.
    *args: Positional arguments for jax.value_and_grad
    **kwargs: Named arguments for jax.value_and_grad.
  Returns:
    batch_value_and_grad_fn: Function returning the value and gradients of the
      batched function
  """
  add_batch_dim = lambda x: jnp.expand_dims(x, batch_dims)
  remove_batch_dim = lambda x: x.squeeze(batch_dims)
  def nobatch_fun(*nobatch_inps):
    batch_inps = jax.tree_util.tree_multimap(add_batch_dim, nobatch_inps)
    batch_out = fun(*batch_inps)
    nobatch_out = jax.tree_util.tree_multimap(remove_batch_dim, batch_out)
    return nobatch_out
  nobatch_value_and_grad = jax.value_and_grad(nobatch_fun, *args, **kwargs)

  batch_value_and_grad_fn = nobatch_value_and_grad
  for batch_dim in batch_dims:
    batch_value_and_grad_fn = jax.vmap(batch_value_and_grad_fn,
                                       in_axes=batch_dim, out_axes=batch_dim)
  return batch_value_and_grad_fn


def objective_chunk(
    obj_shape: Sequence[int],
    chunk_index: int,
    nb_parallel_nodes: int,
):
  """Returns a one-hot tensor to select a chunk of elements from an objective.

  Args:
    obj_shape: Shape of the objective tensor to be chunked.
    chunk_index: Index of the optimization chunk to generate.
    nb_parallel_nodes: How large should the optimization chunks be. If 0,
      optimize all problems at once.
  Returns:
    One-hot tensor of shape (nb_parallel_nodes, *obj_shape) specifying,
      for each index in the chunk, an element of the objective.
  """
  total_nb_nodes_to_opt = int(np.prod(obj_shape))

  start_node = chunk_index * nb_parallel_nodes
  if (nb_parallel_nodes == 0) or (total_nb_nodes_to_opt <= nb_parallel_nodes):
    nb_nodes_to_opt = total_nb_nodes_to_opt
  else:
    nb_nodes_to_opt = nb_parallel_nodes

  # In order to be able to use the function in the while loop, we have to have
  # all tensors remain the same size so we're going to always create a tensor
  # of the same size, but will not necessarily fill all the rows.
  flat_obj = jnp.zeros((nb_nodes_to_opt, total_nb_nodes_to_opt))
  opt_idx = jnp.arange(nb_nodes_to_opt)
  node_idx = jnp.minimum(start_node + opt_idx, total_nb_nodes_to_opt-1)
  to_add = ((start_node + opt_idx) < total_nb_nodes_to_opt).astype(jnp.float32)
  flat_obj = flat_obj.at[(opt_idx, node_idx)].add(
      to_add, indices_are_sorted=True, unique_indices=False)
  obj = jnp.reshape(flat_obj, (nb_nodes_to_opt, *obj_shape))

  return obj


def chunked_bounds(
    bound_shape: Tuple[int, ...],
    max_parallel_nodes: int,
    bound_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor]:
  """Perform computation of the bounds in chunks.

  Args:
    bound_shape: Shape of the bounds to compute
    max_parallel_nodes: How many activations' bounds to compute at once.
      If zero, compute all the activations' bounds simultaneously.
    bound_fn: Function to compute bounds for a chunk, given a one-hot tensor
      of shape (nb_parallel_nodes, *obj_shape) specifying the activation
      elements to compute.
  Returns:
    Computed lower and upper bounds.
  """
  def bound_chunk(chunk_index: int) -> Tuple[Tensor, Tensor]:
    # Create the objective matrix
    obj = objective_chunk(bound_shape, chunk_index, max_parallel_nodes)
    return bound_fn(obj)

  nb_act = int(np.prod(bound_shape))
  if (max_parallel_nodes == 0) or (nb_act <= max_parallel_nodes):
    flat_lbs, flat_ubs = bound_chunk(0)
  else:
    nb_bound_chunk = math.ceil(nb_act / max_parallel_nodes)
    chunk_indices = jnp.arange(nb_bound_chunk)
    (map_lbs, map_ubs) = jax.lax.map(bound_chunk, chunk_indices)
    # Remove the padding elements
    flat_lbs = jnp.reshape(map_lbs, (-1,))[:nb_act]
    flat_ubs = jnp.reshape(map_ubs, (-1,))[:nb_act]
  lbs = jnp.reshape(flat_lbs, bound_shape)
  ubs = jnp.reshape(flat_ubs, bound_shape)
  return lbs, ubs
