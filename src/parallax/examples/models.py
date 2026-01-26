"""Example models."""

from flax import linen as nn
from flax import nnx
import jax.numpy as jnp


def SimpleMLP(
    din: int,
    dmid: int,
    dout: int,
    *,
    rngs: nnx.Rngs | None = None,
) -> nnx.Module:
  rngs = rngs or nnx.Rngs(42)
  return nnx.Sequential(
      nnx.Linear(
          in_features=din,
          out_features=dmid,
          kernel_init=nnx.initializers.xavier_uniform(),
          bias_init=nnx.initializers.zeros_init(),
          rngs=rngs,
      ),
      nnx.Linear(
          in_features=dmid,
          out_features=dout,
          kernel_init=nnx.initializers.xavier_uniform(),
          bias_init=nnx.initializers.zeros_init(),
          rngs=rngs,
      ),
  )


class SimpleMLPLinen(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=self.features, name='dense_1')(x)
    x = nn.relu(x)
    x = nn.Dense(features=x.shape[-1], name='dense_2')(x)
    return x


class TransformerBlock(nnx.Module):
  """Defines a block for a transformer model."""

  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      ff_dim: int,
      *,
      rngs: nnx.Rngs,
      rate: float = 0.1,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.mha = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_dim,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.dropout1 = nnx.Dropout(rate=rate)
    self.layer_norm1 = nnx.LayerNorm(
        epsilon=1e-6,
        num_features=embed_dim,
        scale_init=nnx.initializers.ones_init(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.linear1 = nnx.Linear(
        in_features=embed_dim,
        out_features=ff_dim,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.linear2 = nnx.Linear(
        in_features=ff_dim,
        out_features=embed_dim,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.dropout2 = nnx.Dropout(rate=rate)
    self.layer_norm2 = nnx.LayerNorm(
        epsilon=1e-6,
        num_features=embed_dim,
        scale_init=nnx.initializers.ones_init(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
        param_dtype=param_dtype,
    )

  def __call__(self, inputs, training: bool = False):
    input_shape = inputs.shape
    _, seq_len, _ = input_shape

    # Create causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))

    # Apply MultiHeadAttention with causal mask
    attention_output = self.mha(inputs_q=inputs, mask=mask, decode=False)
    attention_output = self.dropout1(
        attention_output, deterministic=not training
    )
    out1 = self.layer_norm1(inputs + attention_output)

    # Feed-forward network
    ffn_output = self.linear1(out1)
    ffn_output = nnx.relu(ffn_output)
    ffn_output = self.linear2(ffn_output)
    ffn_output = self.dropout2(ffn_output, deterministic=not training)

    return self.layer_norm2(out1 + ffn_output)


class TokenAndPositionEmbedding(nnx.Module):
  """Defines an embedding layer for a transformer model."""

  def __init__(
      self,
      maxlen: int,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.token_emb = nnx.Embed(
        num_embeddings=vocab_size,
        features=embed_dim,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.pos_emb = nnx.Embed(
        num_embeddings=maxlen,
        features=embed_dim,
        rngs=rngs,
        param_dtype=param_dtype,
    )

  def __call__(self, x):
    positions = jnp.arange(0, x.shape[1])[None, :]
    position_embedding = self.pos_emb(positions)
    token_embedding = self.token_emb(x)
    return token_embedding + position_embedding


def MiniGPT(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
    embed_dim: int = 256,
    num_heads: int = 8,
    feed_forward_dim: int = 256,
    num_transformer_blocks: int = 8,
    param_dtype: jnp.dtype = jnp.float32,
) -> nnx.Module:
  """Generates a Mini GPT model as a NNX Sequential module."""
  layers = [
      TokenAndPositionEmbedding(
          maxlen,
          vocab_size,
          embed_dim,
          rngs=rngs,
          param_dtype=param_dtype,
      ),
  ]
  for _ in range(num_transformer_blocks):
    layers.append(
        TransformerBlock(
            embed_dim,
            num_heads,
            feed_forward_dim,
            rngs=rngs,
            param_dtype=param_dtype,
        )
    )
  layers.append(
      nnx.Linear(
          in_features=embed_dim,
          out_features=vocab_size,
          kernel_init=nnx.initializers.xavier_uniform(),
          bias_init=nnx.initializers.zeros_init(),
          rngs=rngs,
          param_dtype=param_dtype,
      )
  )
  return nnx.Sequential(*layers)


def MiniGPT8M(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=256,
      num_heads=32,
      feed_forward_dim=256,
      num_transformer_blocks=8,
  )


def MiniGPT600M(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=1280,
      num_heads=20,
      feed_forward_dim=1280 * 4,
      num_transformer_blocks=24,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT1B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=1536,
      num_heads=32,
      feed_forward_dim=1536 * 4,
      num_transformer_blocks=32,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT5B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=4096,
      num_heads=32,
      feed_forward_dim=4096 * 4,
      num_transformer_blocks=25,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT7B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=4096,
      num_heads=32,
      feed_forward_dim=4096 * 4,
      num_transformer_blocks=32,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT10B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=5120,
      num_heads=40,
      feed_forward_dim=20480,
      num_transformer_blocks=32,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT27B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=6656,
      num_heads=52,
      feed_forward_dim=26624,
      num_transformer_blocks=40,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT30B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=7424,
      num_heads=58,
      feed_forward_dim=29696,
      num_transformer_blocks=60,
      param_dtype=jnp.bfloat16,
  )


def MiniGPT70B(
    vocab_size: int,
    rngs: nnx.Rngs,
    maxlen: int = 256,
) -> nnx.Module:
  return MiniGPT(
      vocab_size=vocab_size,
      rngs=rngs,
      maxlen=maxlen,
      embed_dim=8192,
      num_heads=64,
      feed_forward_dim=28672,
      num_transformer_blocks=80,
      param_dtype=jnp.bfloat16,
  )
