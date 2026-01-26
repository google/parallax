"""NNX implementation of GPT-OSS."""

import math
from flax import nnx
import jax
import jax.numpy as jnp
from parallax.examples.gpt_oss import config as config_lib
from parallax.examples.gpt_oss import moe_layer


def _generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    cfg: config_lib.Config,
) -> tuple[jax.Array, jax.Array]:
  """Yarn RoPE implementation."""
  base, factor = cfg.rope_theta, cfg.rope_factor
  original_max_pos = cfg.rope_original_max_position_embeddings
  low = (
      features * math.log(original_max_pos / (cfg.rope_beta_fast * 2 * math.pi))
  ) / (2 * math.log(base))
  high = (
      features * math.log(original_max_pos / (cfg.rope_beta_slow * 2 * math.pi))
  ) / (2 * math.log(base))
  low, high = max(low, 0), min(high, features - 1)

  timescale = base ** (jnp.arange(0, features, 2, dtype=jnp.float32) / features)
  rot_freq_extra, rot_freq_inter = 1.0 / timescale, 1.0 / (factor * timescale)

  high = high if low != high else (high + 0.001)
  interp_factor = 1 - jnp.clip(
      (jnp.arange(features // 2, dtype=jnp.float32) - low) / (high - low),
      min=0,
      max=1,
  )

  rotational_frequency = (
      rot_freq_inter * (1 - interp_factor) + rot_freq_extra * interp_factor
  )
  sinusoid_inp = jnp.einsum(
      "BT,k->BTk",
      positions,
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )

  m_scale = 1.0
  attention_scaling = (
      1.0 if factor <= 1 else (0.1 * m_scale * math.log(factor) + 1.0)
  )
  return (
      jnp.sin(sinusoid_inp)[:, :, : features // 2] * attention_scaling,
      jnp.cos(sinusoid_inp)[:, :, : features // 2] * attention_scaling,
  )


def _apply_rotary_embedding(
    x: jax.Array, sin: jax.Array, cos: jax.Array
) -> jax.Array:
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  sin, cos = sin[:, None, :, :], cos[:, None, :, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class GroupedQueryAttention(nnx.Module):
  """A grouped query attention layer."""

  def __init__(self, config: config_lib.Config):
    super().__init__()
    self.q_proj = nnx.Linear(
        in_features=config.embed,
        out_features=config.q_heads * config.head_dim,
        rngs=nnx.Rngs(0),
    )
    self.k_proj = nnx.Linear(
        in_features=config.embed,
        out_features=config.kv_heads * config.head_dim,
        rngs=nnx.Rngs(0),
    )
    self.v_proj = nnx.Linear(
        in_features=config.embed,
        out_features=config.kv_heads * config.head_dim,
        rngs=nnx.Rngs(0),
    )
    self.o_proj = nnx.Linear(
        in_features=config.q_heads * config.head_dim,
        out_features=config.embed,
        rngs=nnx.Rngs(0),
    )
    self.sinks = nnx.Param(
        jax.random.normal(nnx.Rngs(0).params(), (config.q_heads,))
    )
    self.config = config

  def __call__(self, x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    batch_size, seq_len, _ = x.shape
    q = self.q_proj(x).reshape(
        batch_size, seq_len, self.config.q_heads, self.config.head_dim
    )
    k = self.k_proj(x).reshape(
        batch_size, seq_len, self.config.kv_heads, self.config.head_dim
    )
    v = self.v_proj(x).reshape(
        batch_size, seq_len, self.config.kv_heads, self.config.head_dim
    )

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    q = _apply_rotary_embedding(q, sin, cos)
    k = _apply_rotary_embedding(k, sin, cos)

    if self.config.q_heads != self.config.kv_heads:
      k = jnp.repeat(k, self.config.q_heads // self.config.kv_heads, axis=1)
      v = jnp.repeat(v, self.config.q_heads // self.config.kv_heads, axis=1)

    attention_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(
        self.config.head_dim
    )

    sinks = self.sinks.value.reshape(1, -1, 1, 1)
    attention_scores = attention_scores + sinks

    attention_scores = nnx.softmax(attention_scores, axis=-1)
    attention_output = jnp.matmul(attention_scores, v)
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, -1
    )
    return self.o_proj(attention_output)


class Block(nnx.Module):
  """A Transformer block."""

  def __init__(self, config: config_lib.Config):
    super().__init__()
    self.attention = GroupedQueryAttention(config)
    self.moe = moe_layer.MoELayer(config)
    self.ln1 = nnx.RMSNorm(num_features=config.embed, rngs=nnx.Rngs(0))
    self.ln2 = nnx.RMSNorm(num_features=config.embed, rngs=nnx.Rngs(0))

  def __call__(self, x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    x = x + self.attention(self.ln1(x), sin, cos)
    x = x + self.moe(self.ln2(x))
    return x


class GptOss(nnx.Module):
  """A GPT-OSS model for demo purposes."""

  def __init__(self, config: config_lib.Config):
    super().__init__()
    self.config = config
    self.embedding = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.embed,
        rngs=nnx.Rngs(0),
    )
    self.layers = nnx.List([Block(config) for _ in range(config.num_layers)])
    self.ln_f = nnx.RMSNorm(num_features=config.embed, rngs=nnx.Rngs(0))
    self.decoder = nnx.Linear(
        in_features=config.embed,
        out_features=config.vocab_size,
        rngs=nnx.Rngs(0),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    _, seq_len = x.shape
    x = self.embedding(x)

    positions = jnp.arange(seq_len, dtype=jnp.float32).reshape(1, -1)
    sin, cos = _generate_pos_embeddings(
        positions, self.config.head_dim, self.config
    )

    for layer in self.layers:
      x = layer(x, sin, cos)
    x = self.ln_f(x)
    x = self.decoder(x)
    return x
