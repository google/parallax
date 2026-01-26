# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implements FSDP sharding strategy."""

import jax
import jaxtyping

NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec
PyTree = jaxtyping.PyTree


def get_shardings(
    fn,
    params,
    *inputs,
    data_axis_name: str = 'data',
    model_axis_name: str = 'model',
    **kwargs,
) -> tuple[tuple[PyTree, ...], PyTree]:
  """Returns sharding assignments for FSDP Training.

  In FSDP, parameters are sharded across devices, and inputs/outputs are
  sharded along the batch dimension.

  Args:
    fn: The function to be sharded.
    params: The parameters of the function.
    *inputs: The inputs to the function.
    data_axis_name: The name of the data axis to shard inputs/outputs on.
    model_axis_name: The name of the model axis to shard parameters on.
    **kwargs: Unused.

  Returns:
    A tuple of ((params_assignments, *inputs_assignments), output_assignments).
    Assignments are pytrees with the same structure as inputs.
  """
  del kwargs

  # Shard the last dimension of parameters for FSDP.
  def _get_fsdp_param_sharding(param_value):
    """Create a PartitionSpec that shards the last axis of a tensor."""
    if param_value.ndim == 0:
      return P()  # Replicate scalars
    sharding = [None] * param_value.ndim
    sharding[-1] = model_axis_name
    return P(*sharding)

  params_assignments = jax.tree_util.tree_map(_get_fsdp_param_sharding, params)

  def get_data_sharding(x):
    """Returns a sharding spec for data, sharded along the first axis."""
    sharding = [None] * x.ndim
    if x.ndim > 0:
      sharding[0] = data_axis_name
    return sharding

  inputs_assignments = jax.tree_util.tree_map(get_data_sharding, inputs)
  output_shapes = fn(params, *inputs)
  output_assignments = jax.tree_util.tree_map(get_data_sharding, output_shapes)
  return (params_assignments, *inputs_assignments), output_assignments
