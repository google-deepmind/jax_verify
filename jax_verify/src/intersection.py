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

"""Mechanism to combine input bounds from multiple methods."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax_verify.src import bound_propagation


Tensor = jnp.ndarray
TransformContext = bound_propagation.TransformContext
InputBound = bound_propagation.InputBound


class IntersectionBound(bound_propagation.Bound):
  """Concretises to intersection of constituent bounds."""

  def __init__(self, *base_bounds: bound_propagation.Bound):
    self.base_bounds = base_bounds

  @property
  def lower(self) -> Tensor:
    return jnp.array([bound.lower for bound in self.base_bounds]).max(axis=0)

  @property
  def upper(self) -> Tensor:
    return jnp.array([bound.upper for bound in self.base_bounds]).min(axis=0)


class ConstrainedBound(bound_propagation.Bound):
  """Wraps a Bound with additional concrete constraints."""

  def __init__(
      self,
      base_bound: bound_propagation.Bound,
      lower: Optional[Tensor],
      upper: Optional[Tensor],
  ):
    self._base_bound = base_bound
    self._lower = lower
    self._upper = upper

  def unwrap(self) -> bound_propagation.Bound:
    return self._base_bound.unwrap()

  @property
  def lower(self) -> Tensor:
    if self._lower is None:
      return self._base_bound.lower
    else:
      return jnp.maximum(self._base_bound.lower, self._lower)

  @property
  def upper(self) -> Tensor:
    if self._upper is None:
      return self._base_bound.upper
    else:
      return jnp.minimum(self._base_bound.upper, self._upper)


class IntersectionBoundTransform(bound_propagation.BoundTransform):
  """Aggregates several bound transforms, intersecting their concrete bounds."""

  def __init__(self, *base_transforms: bound_propagation.BoundTransform):
    self._base_transforms = base_transforms

  def input_transform(
      self, context: TransformContext, input_bound: InputBound
  ) -> IntersectionBound:
    """Constructs initial input bounds for each constituent bound type.

    Args:
      context: Transform context containing node index.
      input_bound: Original concrete bounds on the input.

    Returns:
      Intersection of the constituent input bounds.
    """
    return IntersectionBound(*[
        transform.input_transform(context, input_bound)
        for transform in self._base_transforms])

  def primitive_transform(
      self, context: TransformContext,
      primitive: jax.core.Primitive, *args, **kwargs
  ) -> IntersectionBound:
    """Propagates bounds for each constituent bound type.

    Args:
      context: Transform context containing node index.
      primitive: Primitive Jax operation to transform.
      *args: Arguments of the primitive, wrapped as `IntersectionBound`s.
      **kwargs: Keyword Arguments of the primitive.

    Returns:
      Intersection of the propagated constituent output bounds.
    """
    def base_args_for_arg(arg):
      if isinstance(arg, bound_propagation.Bound):
        return [ConstrainedBound(bound, arg.lower, arg.upper)
                for bound in arg.unwrap().base_bounds]
      else:
        # Broadcast over the intersection components.
        return [arg for _ in self._base_transforms]

    base_args = [base_args_for_arg(arg) for arg in args]
    return IntersectionBound(*[
        transform.equation_transform(context, primitive, *args, **kwargs)
        for transform, *args in zip(self._base_transforms, *base_args)])

  def should_handle_as_subgraph(
            self, primitive: jax.core.Primitive) -> bool:
    # Handle as a sub-graph only if _all_ intersectands can.
    # Otherwise handle at the higher (synthetic primitive) level.
    return all(base_transform.should_handle_as_subgraph(primitive)
               for base_transform in self._base_transforms)

