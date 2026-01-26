"""Fiddle configs for Parallax examples."""

import dataclasses
import datetime
import os
from typing import Any

import fiddle as fdl
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import optax
import parallax
from parallax.examples import models
from tunix.models.gemma import model as gemma


@dataclasses.dataclass
class TrainConfig:
  model: nnx.Module
  optimizer: nnx.ModelAndOptimizer
  train_loader: grain.RandomAccessDataSource
  log_dir: str
  train_batch_size: int = 8
  train_total_steps: int = 1000


class TestingDataSource(grain.RandomAccessDataSource):
  """Repeats one batch for testing."""

  def __init__(self, inputs, labels):
    self._inputs = inputs
    self._labels = labels

  def __getitem__(self, idx):
    return self._inputs, self._labels

  def __len__(self):
    return len(self._inputs)


def default_config() -> fdl.Buildable:
  """Base config."""
  config = fdl.Config(TrainConfig)
  config.model = fdl.Config(
      models.SimpleMLP,
      din=10,
      dmid=10,
      dout=2,
      rngs=nnx.Rngs(0),
  )
  config.optimizer = fdl.Config(
      nnx.ModelAndOptimizer,
      model=config.model,
      tx=fdl.Config(optax.adam, learning_rate=0.0005, b1=0.9),
  )
  config.log_dir = os.path.join(
      '/tmp/tensorboard',
      'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'),
  )
  config.train_loader = fdl.Config(
      TestingDataSource,
      inputs=(jnp.ones((2, 10)),),
      labels=jnp.ones((2,), dtype=jnp.int32),
  )

  return config


def simple_mlp_auto_sharded() -> fdl.Buildable:
  """SimpleMLP using auto-sharding."""
  config = default_config()

  def create_model_fn():
    return models.SimpleMLP(
        din=10,
        dmid=10,
        dout=2,
        rngs=nnx.Rngs(0),
    )

  config.model = fdl.Config(
      parallax.create_sharded_model,
      model_or_fn=create_model_fn,
      sample_inputs=(jnp.ones((8, 10), dtype=jnp.int32),),
      strategy=parallax.ShardingStrategy.AUTO,
      data_axis_name='fsdp',
      model_axis_name='tp',
  )
  config.optimizer = fdl.Config(
      nnx.ModelAndOptimizer,
      model=config.model,
      tx=fdl.Config(optax.adam, learning_rate=0.0005, b1=0.9),
  )
  return config


def mini_gpt() -> fdl.Buildable:
  """Config for a mini GPT model."""
  config = default_config()
  config.model = fdl.Config(
      models.MiniGPT,
      vocab_size=256,
      maxlen=8,
      rngs=nnx.Rngs(0),
  )
  config.optimizer = fdl.Config(
      nnx.ModelAndOptimizer,
      model=config.model,
      tx=fdl.Config(optax.adam, learning_rate=0.0005, b1=0.9),
  )
  config.train_loader = fdl.Config(
      TestingDataSource,
      inputs=(jnp.ones((1, 8), dtype=jnp.int32),),
      labels=jnp.ones((1, 8), dtype=jnp.int32),
  )
  return config


def _make_gemma_inputs(
    batch_size: int,
    cache_size: int,
    sequence_length: int,
) -> tuple[tuple[Any, ...], jax.Array]:
  """Create inputs and labels for a Gemma model."""
  attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
  last_tokens = jnp.tile(jnp.arange(sequence_length), (batch_size, 1))
  positions = jnp.tile(jnp.arange(sequence_length), (batch_size, 1))
  # TODO(jeffcarp): Add cache.
  cache = None
  inputs = (last_tokens, positions, cache, attention_mask)
  labels = jnp.ones((batch_size, sequence_length), dtype=jnp.int32)
  return inputs, labels


def _make_gemma(
    gemma_config: gemma.ModelConfig,
    batch_size: int = 8,
    cache_size: int = 8,
    sequence_length: int = 8,
) -> fdl.Buildable:
  """Creates a Fiddle config for a Gemma model.

  Args:
    gemma_config: The configuration for the Gemma model.
    batch_size: The batch size for the training data.
    cache_size: The size of the cache for the model.
    sequence_length: The length of the input sequences.

  Returns:
    A Fiddle config for the Gemma model.
  """
  config = default_config()
  config.model = fdl.Config(
      gemma.Gemma,
      config=gemma_config,
      rngs=nnx.Rngs(params=42),
  )
  config.optimizer = fdl.Config(
      nnx.ModelAndOptimizer,
      model=config.model,
      tx=fdl.Config(optax.adam, learning_rate=0.0005, b1=0.9),
  )
  inputs, labels = _make_gemma_inputs(
      batch_size=batch_size,
      cache_size=cache_size,
      sequence_length=sequence_length,
  )
  config.train_loader = fdl.Config(
      TestingDataSource,
      inputs=inputs,
      labels=labels,
  )
  return config


def gemma_tiny() -> fdl.Buildable:
  """Config for a tiny Gemma model."""
  config = _make_gemma(
      dataclasses.replace(
          gemma.ModelConfig.gemma_2b(),
          num_layers=2,
          num_embed=256,
          embed_dim=256,
          num_heads=2,
      ),
  )
  return config


def gemma_2b() -> fdl.Buildable:
  """Config for a 2B Gemma model."""
  config = _make_gemma(gemma.ModelConfig.gemma_2b())
  return config


def gemma_2b_parallax_auto_sharded() -> fdl.Buildable:
  """Gemma 2 2B using Parallax auto-sharding."""
  config = gemma_2b()

  def create_model_fn():
    return gemma.Gemma(
        config=gemma.ModelConfig.gemma_2b(),
        rngs=nnx.Rngs(params=42),
    )

  inputs, _ = _make_gemma_inputs(
      batch_size=8,
      cache_size=8,
      sequence_length=8,
  )
  config.model = fdl.Config(
      parallax.create_sharded_model,
      model_or_fn=create_model_fn,
      inputs=inputs,
      data_axis_name='fsdp',
      model_axis_name='tp',
  )
  config.optimizer.model = config.model
  return config


def _make_xla_auto_sharded_gemma(
    model: nnx.Module,
    mesh,
    train_loader,
) -> nnx.Module:
  """Creates a Gemma model with XLA auto-sharding."""
  jax.config.update('jax_use_shardy_partitioner', False)

  def auto_shard(f, *args):
    args = jax.tree.map(
        lambda x: jax._src.core.ShapedArray(x.shape, x.dtype),  # pylint: disable=protected-access
        args,
    )
    lowered = (
        jax.jit(
            f,
            in_shardings=jax.experimental.pjit.AUTO(mesh),
            out_shardings=jax.experimental.pjit.AUTO(mesh),
        )
        .lower(*args)
        .compile()
    )
    return jax.tree.map(
        lambda x: x.spec, (*lowered.input_shardings, lowered.output_shardings)
    )

  model_inputs, _ = next(iter(train_loader))
  graphdef, state = nnx.split(model)
  state = nnx.to_pure_dict(state)

  def f(state, model_inputs):
    model = nnx.merge(graphdef, state)
    return model(*model_inputs)

  fn_shardings, state_shardings, in_shardings = auto_shard(
      f,
      state,
      model_inputs,
  )
  f_state_shardings, f_in_shardings = fn_shardings
  del state_shardings, in_shardings, f_in_shardings

  def _apply_sharding(parameter_value, sharding_spec_list):
    if not isinstance(parameter_value, jax.Array) or sharding_spec_list is None:
      return parameter_value

    partition_spec = jax.sharding.PartitionSpec(*sharding_spec_list)
    named_sharding = jax.sharding.NamedSharding(mesh, partition_spec)
    sharded_parameter = jax.device_put(parameter_value, named_sharding)
    return sharded_parameter

  sharded_state = jax.tree_util.tree_map(
      _apply_sharding,
      state,
      f_state_shardings,
  )
  nnx.update(model, sharded_state)
  return model


def gemma_2b_xla_auto_sharding() -> fdl.Buildable:
  config = gemma_2b()

  config.model = fdl.Config(
      _make_xla_auto_sharded_gemma,
      model=config.model,
      mesh=parallax.auto_mesh(),
      train_loader=config.train_loader,
  )
  config.optimizer.model = config.model
  return config
