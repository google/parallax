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

"""Defines base sharding APIs for Parallax."""

import enum
import functools
from typing import Any, Callable
from flax import linen as nn
from flax import nnx
import jax
import jaxtyping
from parallax.sharding import auto_shard
from parallax.sharding import ddp
from parallax.sharding import fsdp

NamedSharding = jax.sharding.NamedSharding
PyTree = jaxtyping.PyTree
PartitionSpec = jax.sharding.PartitionSpec


class ShardingStrategy(enum.Enum):
  AUTO = 'auto'
  DDP = 'ddp'
  FSDP = 'fsdp'
  MANUAL = 'manual'


def create_sharded_model(
    model_or_fn: nn.Module | Callable[[], nnx.Module],
    sample_inputs: Any,
    strategy: ShardingStrategy = ShardingStrategy.AUTO,
    **kwargs: Any,
) -> nnx.Module | PyTree:
  """Creates a model instance with sharded parameters.

  If passed an `nn.Module`, returns a PyTree of model variables with sharding
  applied based on `strategy`.

  If passed a function that creates an `nnx.Module`, returns an NNX
  model with sharding applied to the params based on `strategy`. This needs to
  be a function (and not directly an `nnx.Module` so that the shardings can
  be applied *before* the model is instantiated, otherwise this will lead to
  OOMs for large models).

  Args:
    model_or_fn: A Flax Linen `nn.Module` or a function that creates an NNX
      model.
    sample_inputs: Example inputs to the model, used for tracing and shape
      inference.
    strategy: The sharding strategy to apply.
    **kwargs: Additional options for the chosen strategy. For 'auto', this can
      include `min_shard_size`, `data_axis_name`, and `model_axis_name`.

  Returns:
    - Sharded Flax `variables` if the input was a Linen Module.
    - A new, sharded NNX `Module` instance otherwise.
  """
  get_shardings_fn = _pick_shardings_fn(strategy)

  if isinstance(model_or_fn, nn.Module):
    # Support for Flax Linen models.
    model = model_or_fn
    variables = model.init(jax.random.PRNGKey(0), *sample_inputs)
    (params_shd, _), _ = get_shardings_fn(
        model.apply, variables, *sample_inputs, **kwargs
    )

    sharded_params = jax.lax.with_sharding_constraint(variables, params_shd)
    return sharded_params
  else:
    # Support for Flax NNX models.
    @nnx.jit
    def _create_sharded_model():
      model = model_or_fn()
      state = nnx.state(model)

      graphdef = nnx.graphdef(model)

      def fn(state, *inputs):
        return nnx.merge(graphdef, state)(*inputs)

      (params_shd, _), _ = get_shardings_fn(fn, state, *sample_inputs, **kwargs)
      sharded_state = jax.lax.with_sharding_constraint(state, params_shd)
      nnx.update(model, sharded_state)
      return model

    return _create_sharded_model()


def jit(
    func: Callable[..., Any] | None = None,
    strategy: ShardingStrategy = ShardingStrategy.AUTO,
    **kwargs,
):
  """JIT-compiles a function with a specified sharding strategy.

  Args:
    func: The function to be compiled.
    strategy: The sharding strategy to apply.
    **kwargs: Example keyword arguments for the function and options for the
      chosen strategy. For 'auto', this can include `min_shard_size`,
      `data_axis_name`, and `model_axis_name`.

  Returns:
    The JIT-compiled function with sharding applied.
  """
  if func is None:
    # @jit(strategy=...)
    return functools.partial(jit, strategy=strategy, **kwargs)

  # @jit
  return JitWrapped(func, strategy, kwargs)


class JitWrapped:
  """Wraps a function and sharding info."""

  def __init__(self, func, strategy, kwargs):
    # TODO(b/452969631): Enforce `params` as first arg to `func`.
    self.func = func
    self.strategy = strategy
    self.kwargs = kwargs
    functools.update_wrapper(self, func)

  def _get_jitted_fn(self, params, *args, **kwargs):
    """Constructs the final jit-wrapped function with strategy applied."""
    get_shardings_fn = _pick_shardings_fn(self.strategy)
    all_kwargs = {**self.kwargs, **kwargs}
    in_assignments, out_assignments = get_shardings_fn(
        self.func, params, *args, **all_kwargs
    )

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
      raise ValueError('A mesh context is required for sharding.')

    def _leaf_to_sharding(pspec_list):
      if isinstance(pspec_list, jax.sharding.Sharding):
        return pspec_list
      if pspec_list is None:
        return jax.sharding.PartitionSpec()
      else:
        return jax.sharding.PartitionSpec(*pspec_list)

    final_in_shardings = jax.tree.map(
        _leaf_to_sharding,
        in_assignments,
        is_leaf=lambda x: isinstance(x, list) or x is None,
    )
    final_out_shardings = jax.tree.map(
        _leaf_to_sharding,
        out_assignments,
        is_leaf=lambda x: isinstance(x, list) or x is None,
    )
    jitted_fn = jax.jit(
        self.func,
        in_shardings=final_in_shardings,
        out_shardings=final_out_shardings,
    )
    return jitted_fn, final_in_shardings

  def __call__(self, params, *args, **kwargs):
    jitted_fn, final_in_shardings = self._get_jitted_fn(params, *args, **kwargs)

    # Reshard inputs to match new shardings.
    args = jax.tree_util.tree_map(
        jax.device_put,
        (params,) + args,
        final_in_shardings,
    )

    return jitted_fn(*args, **kwargs)

  def lower(self, params, *args, **kwargs):
    jitted_fn, final_in_shardings = self._get_jitted_fn(params, *args, **kwargs)

    # Reshard inputs to match new shardings.
    args = jax.tree_util.tree_map(
        jax.device_put,
        (params,) + args,
        final_in_shardings,
    )
    return jitted_fn.lower(*args, **kwargs)


def _pick_shardings_fn(strategy: ShardingStrategy) -> Callable[..., Any]:
  if strategy == ShardingStrategy.AUTO:
    get_shardings_fn = auto_shard.get_shardings
  elif strategy == ShardingStrategy.DDP:
    get_shardings_fn = ddp.get_shardings
  elif strategy == ShardingStrategy.FSDP:
    get_shardings_fn = fsdp.get_shardings
  else:
    raise NotImplementedError(f"Strategy '{strategy}' is not yet implemented.")

  return get_shardings_fn
