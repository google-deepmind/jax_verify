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

# Lint as: python3
"""Functions for computing eigenvectors/eigenvalues (e.g. Lanczos)."""
# pylint: disable=invalid-name
# Capital letters for matrices

import time
from absl import logging
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg
import numpy as np


def safe_eigh(a, UPLO=None, symmetrize_input=True):
  # TODO: Remove when issue with CUDA eigh is resolved
  eigs, eig_vecs = lax.cond(
      jnp.linalg.norm(a) > 0.0,
      lambda t: jax.scipy.linalg.eigh(t, UPLO, symmetrize_input),
      lambda _: (jnp.zeros(a.shape[0]), jnp.eye(a.shape[0])),
      operand=a)
  return jax.lax.stop_gradient(eigs), jax.lax.stop_gradient(eig_vecs)


def lanczos_alg(matrix_vector_product,
                dim,
                order,
                rng_key,
                dynamic_unroll=True,
                use_jax=True,
                verbose=False):
  """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

  This function applies Lanczos algorithm of a given order.  This function
  does full reorthogonalization.

  WARNING: This function may take a long time to jit compile (e.g. ~3min for
  order 90 and dim 1e7).

  Args:
    matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
      Input/Output must be of shape [dim].
    dim: Matrix H is [dim, dim].
    order: An integer corresponding to the number of Lanczos steps to take.
    rng_key: The jax PRNG key.
    dynamic_unroll: bool, False does static unroll (default), True uses
      jax.fori_loop for faster JIT compile times.
    use_jax: Whether or not to do the lanczos computation in jax or numpy.
      Can save memory to use numpy.
    verbose: Print intermediate computations stats.

  Returns:
    tridiag: A tridiagonal matrix of size (order, order).
    vecs: A numpy array of size (order, dim) corresponding to the Lanczos
      vectors.
  """
  if dynamic_unroll:
    assert use_jax, 'Dynamic unroll only available with JAX.'
    return _lanczos_alg_dynamic_unroll(
        matrix_vector_product, dim, order, rng_key)
  if use_jax:
    backend = jnp
    index_update = lambda x, idx, y: x.at[idx].set(y)
  else:
    backend = np
    def _index_update(array, index, value):
      new_array = array.copy()
      new_array[index] = value
      return new_array
    index_update = _index_update
  tridiag = backend.zeros((order, order))
  vecs = backend.zeros((order, dim))

  init_vec = random.normal(rng_key, shape=(dim,))
  init_vec = init_vec / backend.linalg.norm(init_vec)
  vecs = index_update(vecs, 0, init_vec)

  beta = 0
  start = time.time()
  # TODO: Better to use lax.fori loop for faster compile?
  for i in range(order):
    if verbose:
      end = time.time()
      logging.info('Iter %d out of %d. Time: %f', i, order, end-start)
    v = vecs[i, :].reshape((dim,))
    if i == 0:
      v_old = 0
    else:
      v_old = vecs[i - 1, :].reshape((dim,))

    w = matrix_vector_product(v)
    assert (w.shape[0] == dim and len(w.shape) == 1), (
        'Output of matrix_vector_product(v) must be of shape [dim].')
    w = w - beta * v_old

    alpha = backend.dot(w, v)
    tridiag = index_update(tridiag, (i, i), alpha)
    w = w - alpha * v

    # Full Reorthogonalization. Vectorized implementation of Gram Schmidt
    coeffs = backend.dot(vecs, w)
    scaled_vecs = (vecs.transpose()*coeffs).transpose()
    scaled_vecs = backend.sum(scaled_vecs, axis=0)
    w -= scaled_vecs

    beta = backend.linalg.norm(w)

    if i + 1 < order:
      # Small beta (<1e-6) implies Lanczos has converged.
      # TODO: Refactor to not run the loop when beta < 1e-6
      beta_write = lax.cond(beta < 1e-6, beta, jnp.zeros_like, beta,
                            lambda x: x)
      w_write = lax.cond(beta < 1e-6, w, jnp.zeros_like, w / beta, lambda x: x)
      tridiag = index_update(tridiag, (i, i + 1), beta_write)
      tridiag = index_update(tridiag, (i + 1, i), beta_write)
      vecs = index_update(vecs, i + 1, w_write)

  return (tridiag, vecs)


def _lanczos_alg_dynamic_unroll(
    matrix_vector_product, dim, order, rng_key):
  """Lanczos with jax.fori_loop unroll - see docstring for lanczos_alg()."""
  backend = jnp
  index_update = lambda x, idx, y: x.at[idx].set(y)
  tridiag = backend.zeros((order, order))
  vecs = backend.zeros((order, dim))

  init_vec = random.normal(rng_key, shape=(dim,))
  init_vec = init_vec / backend.linalg.norm(init_vec)
  vecs = index_update(vecs, 0, init_vec)

  beta = 0.

  def _body_fn_update_alpha(i, vecs, tridiag, beta):
    """Duplicated code from first half of body_fn() used for final iteration."""
    v = vecs[i, :].reshape((dim,))
    v_old = vecs[i - 1, :].reshape((dim,))
    w = matrix_vector_product(v)
    assert (w.shape[0] == dim and len(w.shape) == 1), (
        'Output of matrix_vector_product(v) must be of shape [dim].')
    w = w - beta * v_old
    alpha = backend.dot(w, v)
    tridiag = index_update(tridiag, (i, i), alpha)
    return tridiag

  def body_fn(i, vals):
    """Main body used for jax.fori_loop."""
    vecs, tridiag, beta = vals
    v = vecs[i, :].reshape((dim,))
    v_old = lax.cond(i == 0, None, lambda x: jnp.zeros(dim, jnp.float32),
                     vecs, lambda vecs: vecs[i - 1, :].reshape((dim,)))
    w = matrix_vector_product(v)
    assert (w.shape[0] == dim and len(w.shape) == 1), (
        'Output of matrix_vector_product(v) must be of shape [dim].')
    w = w - beta * v_old

    alpha = backend.dot(w, v)
    tridiag = index_update(tridiag, (i, i), alpha)
    w = w - alpha * v

    # Full Reorthogonalization. Vectorized implementation of Gram Schmidt
    coeffs = backend.dot(vecs, w)
    scaled_vecs = (vecs.transpose()*coeffs).transpose()
    scaled_vecs = backend.sum(scaled_vecs, axis=0)
    w -= scaled_vecs

    beta = backend.linalg.norm(w)

    # Small beta (<1e-6) implies Lanczos has converged.
    beta_write = lax.cond(beta < 1e-6, beta, jnp.zeros_like, beta, lambda x: x)
    w_write = lax.cond(beta < 1e-6, w, jnp.zeros_like, w / beta, lambda x: x)
    tridiag = index_update(tridiag, (i, i + 1), beta_write)
    tridiag = index_update(tridiag, (i + 1, i), beta_write)
    vecs = index_update(vecs, i + 1, w_write)
    return (lax.stop_gradient(vecs), lax.stop_gradient(tridiag),
            lax.stop_gradient(beta))

  vecs, tridiag, beta_final = jax.lax.fori_loop(
      0, order - 1, body_fn, (vecs, tridiag, beta))
  # Update tridiag one last time for final iteration
  tridiag = _body_fn_update_alpha(order - 1, vecs, tridiag, beta_final)
  return (tridiag, vecs)


def _make_pos(vecx):
  return jnp.maximum(vecx, 0)


############     Lanczos, Lagrangian, dual function     ############


def max_eigenvector_lanczos(matrix_vector_product, dim, order, key, scl=-1,
                            dynamic_unroll=True, use_safe_eig_vec=True):
  """Get (soft)max eigenvector via Lanczos + Scipy eigendecomp."""
  tridiag, vecs = lanczos_alg(matrix_vector_product, dim, order, key,
                              verbose=False, dynamic_unroll=dynamic_unroll)
  eigs_triag, eig_vecs = safe_eigh(tridiag)
  if scl < 0:
    # Get max eigenvector
    eig_vec = jnp.dot(jnp.transpose(vecs), eig_vecs[:, -1])
  else:
    # Softmax weighting of max eigenvector - better gradients?
    eig_softmax = jnp.exp(scl*eigs_triag -
                          jax.scipy.special.logsumexp(scl * eigs_triag))
    eig_vec = jnp.dot(jnp.transpose(vecs), jnp.dot(eig_vecs, eig_softmax))

  if use_safe_eig_vec:
    # To handle the case when the norm of the eigen vector is ~0.
    # This can happen when triag is rank deficient. To handle this corner case,
    # sample a new eigen-vector and remove components with respect to
    # all eigen-vecs with non-zero eigen vals to get a vector in the null-space
    # of the Hessian => Eigen vector correspoding to eig-val 0.
    # TODO: Possible suspect if Lanczos starts to diverge.
    def get_eig_vec(vals):
      key, (eig_vecs, vecs, eigs_triag) = vals
      eig_vec = jnp.dot(jnp.transpose(vecs), eig_vecs[:, -1])
      random_vec = jax.random.uniform(key, shape=eig_vec.shape)
      eig_vecs = jnp.dot(jnp.transpose(vecs), eig_vecs)
      coeffs = jnp.dot(random_vec, eig_vecs)
      scaled_vecs = (eig_vecs * coeffs * (eigs_triag > 1e-7))
      scaled_vecs = jnp.sum(scaled_vecs, axis=1)
      eig_vec = random_vec - scaled_vecs
      return eig_vec

    vals = key, (eig_vecs, vecs, eigs_triag)
    eig_norm = jnp.linalg.norm(eig_vec)
    eig_vec = lax.cond(eig_norm < 1e-7, vals, get_eig_vec, eig_vec, lambda x: x)
  eig_vec = eig_vec/jnp.linalg.norm(eig_vec)
  return jax.lax.stop_gradient(eig_vec)


def min_eigenvector_lanczos(matrix_vector_product, *args, **kwargs):
  # If for matrix H, eigenvector v has eigenvalue lambda, then for matrix -H,
  # v has eigenvalue -lambda. So we find max eigenvector for -H instead.
  neg_mat_vec_product = lambda v: -matrix_vector_product(v)
  return max_eigenvector_lanczos(neg_mat_vec_product, *args, **kwargs)


def max_eigenvector_exact(matrix_vector_product, vec_dim, scl=-1,
                          report_all=False):
  """Get max eigenvector via Scipy eigendecomp."""
  @jax.jit
  def batched_Hv(v_batched):
    return jax.vmap(matrix_vector_product)(v_batched)

  H = batched_Hv(jnp.eye(vec_dim))
  H = (H + H.T)/2
  eig_vals, eig_vecs = safe_eigh(H)
  if scl < 0:
    eig_vec = eig_vecs[:, -1]
  else:
    eig_softmax_weights = jnp.exp(
        scl*eig_vals - jax.scipy.special.logsumexp(scl*eig_vals))
    eig_vec = jnp.sum(eig_vecs * jnp.expand_dims(eig_softmax_weights, axis=0),
                      axis=1)
  eig_vec = jax.lax.stop_gradient(eig_vec)
  if report_all:
    return eig_vec, (eig_vals, eig_vecs, H)
  return eig_vec


def min_eigenvector_exact(matrix_vector_product, vec_dim, scl=-1,
                          report_all=False):
  neg_mat_vec_product = lambda v: -matrix_vector_product(v)
  return max_eigenvector_exact(neg_mat_vec_product, vec_dim,
                               scl=scl, report_all=report_all)
