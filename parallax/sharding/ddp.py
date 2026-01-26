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

"""Data Parallel sharding strategy."""

import jax


P = jax.sharding.PartitionSpec


def get_shardings(fn, params, *inputs, data_axis_name: str = 'data', **kwargs):
  """Returns sharding assignments for Data Parallel Training.

  In DDP, all parameters are replicated across all devices. The inputs (and
  typically outputs) are sharded along the batch dimension.

  Args:
    fn: The function to be sharded.
    params: The parameters of the function.
    *inputs: The inputs to the function.
    data_axis_name: The name of the data axis to shard inputs/outputs on.
    **kwargs: Unused.

  Returns:
    A tuple of ((params_assignments, *inputs_assignments), output_assignments).
    Assignments are pytrees with the same structure as inputs.
  """
  del kwargs  # Unused for DDP.

  def get_replicated_sharding(x):
    """Returns a fully replicated sharding spec for a given tensor shape."""
    sharding = [None] * x.ndim
    return P(*sharding)

  params_assignments = jax.tree_util.tree_map(get_replicated_sharding, params)

  def get_ddp_sharding(x):
    """Returns a sharding spec for data, sharded along the first axis."""
    sharding = [None] * x.ndim
    if x.ndim > 0:
      sharding[0] = data_axis_name
    return P(*sharding)

  inputs_assignments = jax.tree_util.tree_map(get_ddp_sharding, inputs)
  output_shapes = fn(params, *inputs)
  output_assignments = jax.tree_util.tree_map(get_ddp_sharding, output_shapes)
  return (params_assignments, *inputs_assignments), output_assignments
