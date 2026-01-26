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
"""Auto-sharding."""

import collections
import itertools

import jax
import jaxtyping
from parallax.sharding import data_structures

PyTree = jaxtyping.PyTree
P = jax.sharding.PartitionSpec


def get_shardings(
    fn,
    params,
    *inputs,
    min_shard_size: int = 0,
    data_axis_name: str = 'data',
    model_axis_name: str = 'model',
) -> tuple[tuple[PyTree, ...], PyTree]:
  """Apply auto-sharding to a function.

  This function will try to assign one "model" sharding and multiple "data"
  shardings to the params, the inputs and the outputs.

  The axis that appears most often in the parameters will be assigned with the
  "model" sharding, together with the axes that can share the same sharding.
  The heuristic is weighted by the size of the dimension.
  Axes that appear in the inputs but not in the parameters will be assigned with
  the "data" sharding.

  Args:
    fn: a function.
    params: the parameters of the function.
    *inputs: the inputs of the function.
    min_shard_size: tensors smaller than this will not be sharded.
    data_axis_name: The name of the data axis.
    model_axis_name: The name of the model axis.

  Returns:
    A tuple of ((params_assignments, *inputs_assignments), output_assignments).
    Assignments are pytrees with the same structure as inputs.
  """
  # TODO(b/452968881): Take axis names from current mesh.
  mesh = jax.sharding.get_abstract_mesh()
  if len(mesh.axis_names) != 2:
    raise ValueError('Autosharding currently requires a mesh of rank 2.')

  jaxpr, abs_ret = jax.make_jaxpr(fn, return_shape=True)(params, *inputs)
  graph = _parse_jaxpr(jaxpr)
  edges = graph.get_edges()

  params_flat, params_treedef = jax.tree.flatten(params)
  _, inputs_treedef = jax.tree.flatten(inputs)
  _, outputs_treedef = jax.tree.flatten(abs_ret)

  # Assign the mostly seen axis as "model" axis.
  seen = collections.Counter()
  for var in jaxpr.jaxpr.invars[: len(params_flat)]:
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      seen.update([graph.get_root((var, i))])
      model_axis = max(seen, key=lambda x: seen[x]) if seen else None
  # Axes in inputs that are never seen in params are "data" axes.
  data_axes = []
  for var in jaxpr.jaxpr.invars[len(params_flat) :]:
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      root = graph.get_root((var, i))
      if root not in seen and root not in data_axes:
        data_axes.append(root)

  # Try to assign "model" axis to more params, if possible.
  params_assignments = []
  for var in jaxpr.jaxpr.invars[: len(params_flat)]:
    params_assignments.append([])
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      root = graph.get_root((var, i))
      if (model_axis is not None and (root, model_axis) in edges) or var.aval.shape[i] < min_shard_size:  # pytype: disable=attribute-error
        params_assignments[-1].append(None)  # conflict with model axis
      else:
        params_assignments[-1].append(
            model_axis_name if model_axis is not None else None
        )
  params_assignments = params_treedef.unflatten(params_assignments)

  inputs_assignments = []
  for var in jaxpr.jaxpr.invars[len(params_flat) :]:
    inputs_assignments.append([])
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      root = graph.get_root((var, i))
      if root in data_axes and var.aval.shape[i] >= min_shard_size:  # pytype: disable=attribute-error
        name = (
            data_axis_name
            if len(data_axes) == 1
            else f'{data_axis_name}{data_axes.index(root)}'
        )
        inputs_assignments[-1].append(name)
      else:
        inputs_assignments[-1].append(None)
  inputs_assignments = inputs_treedef.unflatten(inputs_assignments)

  output_assignments = []
  for var in jaxpr.jaxpr.outvars:
    output_assignments.append([])
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      root = graph.get_root((var, i))
      if root in data_axes and var.aval.shape[i] >= min_shard_size:  # pytype: disable=attribute-error
        name = (
            data_axis_name
            if len(data_axes) == 1
            else f'{data_axis_name}{data_axes.index(root)}'
        )
        output_assignments[-1].append(name)
      else:
        output_assignments[-1].append(None)
  output_assignments = outputs_treedef.unflatten(output_assignments)

  def _convert_to_pspecs(pytree):
    return jax.tree.map(
        lambda p: P(*p),
        pytree,
        is_leaf=lambda p: isinstance(p, list),
    )

  params_assignments = _convert_to_pspecs(params_assignments)
  inputs_assignments = _convert_to_pspecs(inputs_assignments)
  output_assignments = _convert_to_pspecs(output_assignments)

  return (params_assignments, *inputs_assignments), output_assignments


def analyze_same_axes(fn, *inputs):
  """Analyze and anotate axes that should have the same sharding."""
  jaxpr, abs_ret = jax.make_jaxpr(fn, return_shape=True)(*inputs)
  graph = _parse_jaxpr(jaxpr)

  invars = []
  outvars = []
  assignments = {}
  for var in jaxpr.jaxpr.invars:
    invars.append([])
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      root = graph.get_root((var, i))
      if root not in assignments:
        assignments[root] = len(assignments)
      invars[-1].append(assignments[root])
  for var in jaxpr.jaxpr.outvars:
    outvars.append([])
    for i in range(var.aval.ndim):  # pytype: disable=attribute-error
      root = graph.get_root((var, i))
      if root not in assignments:
        assignments[root] = len(assignments)
      outvars[-1].append(assignments[root])

  return (
      jax.tree.flatten(inputs)[1].unflatten(invars),
      jax.tree.flatten(abs_ret)[1].unflatten(outvars),
  )


def _parse_jaxpr(jaxpr) -> data_structures.MergeableGraph:
  """Parses a jaxpr into an axis graph.

  The returned graph g will have
    1) (variable, axis_index) as nodes.
    2) Two nodes have to be assigned the same sharding if they have the same
       root, i.e., g.get_root(node1) == g.get_root(node2).
    3) Two nodes have to be assigned different shardings if there is an edge
       between their roots, i.e.,
         (g.get_root(node1), g.get_root(node2)) in g.get_edges().

  Args:
    jaxpr: a jaxpr.

  Returns:
    A MergeableGraph.
  """

  # Each node is a tuple (variable: jex.core.Var, axis: int). An edge between
  # two nodes means that the two axes cannot have the same sharding.
  graph = data_structures.MergeableGraph()

  def same_axis(node1, node2):
    var1, axis1 = node1
    var2, axis2 = node2
    assert var1.aval.shape[axis1] == var2.aval.shape[axis2]
    graph.merge_nodes(node1, node2)

  def parse_dot_general(eqn):
    assert len(eqn.outvars) == 1
    lhs, rhs = eqn.invars
    out = eqn.outvars[0]

    (lc, rc), (lb, rb) = eqn.params['dimension_numbers']

    # Handle contracting dimensions.
    for l, r in zip(lc, rc):
      same_axis((lhs, l), (rhs, r))

    # Handle batch dimensions.
    for l, r, o in zip(lb, rb, range(len(lb))):
      same_axis((lhs, l), (rhs, r))
      same_axis((lhs, l), (out, o))

    # Handle remaining dimensions.
    o = len(lb)
    for i in range(lhs.aval.ndim):
      if i not in lb and i not in lc:
        same_axis((lhs, i), (out, o))
        o += 1
    for j in range(rhs.aval.ndim):
      if j not in rb and j not in rc:
        same_axis((rhs, j), (out, o))
        o += 1

  def parse_squeeze(eqn):
    # For example, g:f32[4,16] = squeeze[dimensions=(0,)] f:f32[1,4,16]
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    out = eqn.outvars[0]
    j = 0
    for i in range(invar.aval.ndim):
      if i not in eqn.params['dimensions']:
        same_axis((invar, i), (out, j))
        j += 1

  def parse_broadcast_in_dim(eqn):
    # For example,
    # c:f32[1,4,4] = broadcast_in_dim[broadcast_dimensions=(1, 2)] a:f32[4,4]
    # The axes specified in broadcast_dimensions should share the same sharding
    # root as the corresponding input axes.
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    out = eqn.outvars[0]
    for i, j in zip(range(invar.aval.ndim), eqn.params['broadcast_dimensions']):
      # Skip if dimension is 1.
      if invar.aval.shape[i] == 1:
        continue
      same_axis((invar, i), (out, j))

  def parse_elementwise_with_broadcast(eqn):
    out = eqn.outvars[0]

    for invar in eqn.invars:
      if invar.aval.ndim == 0:
        continue
      assert invar.aval.ndim == out.aval.ndim, (eqn, eqn.invars)
      for i in range(invar.aval.ndim):
        if invar.aval.shape[i] == 1:
          continue
        assert invar.aval.shape[i] == out.aval.shape[i], eqn
        same_axis((invar, i), (out, i))

  def parse_reshape(eqn):
    # For example,
    # b:f32[16,4] = reshape[dimensions=None new_sizes=(16,4)] a:f32[4,4,4]
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    out = eqn.outvars[0]
    in_index = out_index = 0
    in_prod = out_prod = 1
    while in_index < invar.aval.ndim and out_index < out.aval.ndim:
      if out_prod < in_prod:
        out_prod *= out.aval.shape[out_index]
        out_index += 1
      elif in_prod < out_prod:
        in_prod *= invar.aval.shape[in_index]
        in_index += 1
      elif invar.aval.shape[in_index] < out.aval.shape[out_index]:
        in_prod *= invar.aval.shape[in_index]
        in_index += 1
      elif invar.aval.shape[in_index] > out.aval.shape[out_index]:
        out_prod *= out.aval.shape[out_index]
        out_index += 1
      else:
        # in_prod == out_prod
        # and invar.aval.shape[in_index] == out.aval.shape[out_index]
        if out.aval.shape[out_index] > 1:
          same_axis((invar, in_index), (out, out_index))
        in_index += 1
        out_index += 1

  def parse_sub_jaxpr(eqn):
    jaxpr = (
        eqn.params['jaxpr']  # pjit
        if 'jaxpr' in eqn.params
        else eqn.params['call_jaxpr']  # custom_jvp_call
    )
    subgraph = _parse_jaxpr(jaxpr)
    mapping = {}  # from subgraph nodes to graph nodes
    for a, b in zip(jaxpr.jaxpr.invars, eqn.invars):
      assert a.aval == b.aval
      for i in range(a.aval.ndim):
        a_root = subgraph.get_root((a, i))
        if a_root not in mapping:
          mapping[a_root] = (b, i)
        graph.merge_nodes(mapping[a_root], (b, i))
    for a, b in zip(jaxpr.jaxpr.outvars, eqn.outvars):
      assert a.aval == b.aval
      for i in range(a.aval.ndim):
        a_root = subgraph.get_root((a, i))
        if a_root not in mapping:
          mapping[a_root] = (b, i)
        graph.merge_nodes(mapping[a_root], (b, i))

    for x, y in subgraph.get_edges():
      if x not in mapping or y not in mapping:
        continue
      graph.add_edge(mapping[x], mapping[y])

  def parse_gather(eqn):
    # gather is very complex, we only support seen cases.
    dim_numbers = eqn.params['dimension_numbers']
    assert dim_numbers.collapsed_slice_dims == dim_numbers.start_index_map
    assert len(dim_numbers.collapsed_slice_dims) == 1
    slice_idx = dim_numbers.collapsed_slice_dims[0]

    operand, indices = eqn.invars
    out = eqn.outvars[0]
    assert out.aval.ndim == operand.aval.ndim + indices.aval.ndim - 2
    assert indices.aval.shape[-1] == 1
    for i in range(out.aval.ndim):
      if i < slice_idx:
        same_axis((operand, i), (out, i))
      elif i < slice_idx + indices.aval.ndim - 1:
        same_axis((indices, i - slice_idx), (out, i))
      else:
        same_axis((operand, i - indices.aval.ndim + 2), (out, i))

  def parse_transpose(eqn):
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    out = eqn.outvars[0]
    for i, j in zip(eqn.params['permutation'], range(out.aval.ndim)):
      same_axis((invar, i), (out, j))

  def parse_reduce(eqn):
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    out = eqn.outvars[0]
    axes = [int(a) for a in eqn.params['axes']]
    j = 0
    for i in range(invar.aval.ndim):
      if i not in axes:
        same_axis((invar, i), (out, j))
        j += 1

  def parse_slice(eqn):
    # For example:
    # b:f32[2,2] = slice[limit_indices=(3, 6) start_indices=(1, 4)] a:f32[5,7]
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    out = eqn.outvars[0]
    assert invar.aval.ndim == out.aval.ndim
    # Merge dimensions only if shapes match.
    for i in range(out.aval.ndim):
      if invar.aval.shape[i] == out.aval.shape[i]:
        same_axis((invar, i), (out, i))

  def parse_concatenate(eqn):
    # Merge corresponding axes except the concatenation dimension. For example:
    # a:f32[1,8,1,256] = concatenate[dimension=3]
    # b:f32[1,8,1,128] c:f32[1,8,1,128]
    out = eqn.outvars[0]
    concat_dim = eqn.params['dimension']
    for invar in eqn.invars:
      if invar.aval.ndim == 0 or invar.aval.ndim != out.aval.ndim:
        continue
      for i in range(out.aval.ndim):
        if i == concat_dim:
          continue
        if invar.aval.shape[i] == out.aval.shape[i]:
          same_axis((invar, i), (out, i))
        else:
          assert (
              invar.aval.shape[i] == 1
          ), f'Unexpected shape mismatch in concat: {eqn}'

  def parse_split(eqn):
    # Merge corresponding axes between input and outputs, skipping the split
    # axis. For example:
    # a:f32[1,8,1,128] b:f32[1,8,1,128] = split[axis=3 sizes=(128, 128)]
    # c:f32[1,8,1,256]
    assert len(eqn.invars) == 1
    invar = eqn.invars[0]
    split_axis = eqn.params['axis']
    for outvar in eqn.outvars:
      assert invar.aval.ndim == outvar.aval.ndim
      out_axis_idx = 0
      for in_axis_idx in range(invar.aval.ndim):
        if in_axis_idx == split_axis:
          continue
        # Check shapes match before merging
        assert (
            invar.aval.shape[in_axis_idx] == outvar.aval.shape[out_axis_idx]
        ), f'Shape mismatch in split: {eqn}'
        same_axis((invar, in_axis_idx), (outvar, out_axis_idx))
        out_axis_idx += 1

  # Generically add edge between all pairs of axes in the invars.
  for var in jaxpr.jaxpr.invars:
    for i, j in itertools.combinations(range(var.aval.ndim), 2):  # pytype: disable=attribute-error
      graph.add_edge((var, i), (var, j))

  # Generically add edge between all pairs of axes for each output variable
  # of each equation. This ensures that within a single output tensor,
  # different axes cannot share the same sharding.
  for eqn in jaxpr.eqns:
    for outvar in eqn.outvars:
      # Add edges between all pairs of axes for this specific output variable.
      if not hasattr(outvar.aval, 'ndim'):
        continue
      for i, j in itertools.combinations(range(outvar.aval.ndim), 2):
        graph.add_edge((outvar, i), (outvar, j))

    # Now, parse the specific primitive to add constraints between
    # input and output variables, or across variables.
    if eqn.primitive.name == 'dot_general':
      parse_dot_general(eqn)
    elif eqn.primitive.name == 'squeeze':
      parse_squeeze(eqn)
    elif eqn.primitive.name == 'broadcast_in_dim':
      parse_broadcast_in_dim(eqn)
    elif eqn.primitive.name == 'reshape':
      parse_reshape(eqn)
    elif eqn.primitive.name in ('jit', 'pjit', 'custom_jvp_call'):
      parse_sub_jaxpr(eqn)
    elif eqn.primitive.name == 'gather':
      parse_gather(eqn)
    elif eqn.primitive.name == 'transpose':
      parse_transpose(eqn)
    elif eqn.primitive.name in ('reduce_max', 'reduce_sum'):
      parse_reduce(eqn)
    elif eqn.primitive.name == 'slice':
      parse_slice(eqn)
    elif eqn.primitive.name == 'concatenate':
      parse_concatenate(eqn)
    elif eqn.primitive.name == 'split':
      parse_split(eqn)
    else:
      parse_elementwise_with_broadcast(eqn)

  return graph
