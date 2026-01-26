"""Transformer encoder and decoder model."""

from flax import nnx
import jax.numpy as jnp


class TransformerEncoderBlock(nnx.Module):
  """Transformer encoder block consisting of multiple sub-layers.

  Attributes:
      embed_dim: The dimension of the embedding.
      feedforward_dim: The dimension of the feedforward layer.
      num_heads: The number of attention heads.
      dropout_rate: The dropout rate.
      attn_dim: The dimension of the attention layer.
      rngs: The random number generators.
  """

  def __init__(
      self,
      embed_dim: int,
      feedforward_dim: int,
      num_heads: int,
      dropout_rate: float,
      attn_dim: int,
      rngs: nnx.Rngs,
  ):
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.feedforward_dim = feedforward_dim
    self.dropout_rate = dropout_rate
    self.attn_dim = attn_dim

    # Multi-head attention layer
    self.attention = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_dim,
        qkv_features=attn_dim,
        decode=False,
        rngs=rngs,
    )

    # MLP layer (two linear layers)
    self.feedforward = nnx.Sequential(
        nnx.Linear(
            embed_dim,
            feedforward_dim,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        ),
        nnx.relu,
        nnx.Linear(
            feedforward_dim,
            embed_dim,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        ),
    )

    # Layer Normalization
    self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

    # Dropout layers for regularization
    self.dropout_1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    self.dropout_2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

  def __call__(self, inputs, mask=None, deterministic: bool = False):
    # Attention mechanism
    attention_output = self.attention(
        inputs_q=inputs,
        inputs_k=inputs,
        inputs_v=inputs,
        mask=mask,
        decode=False,
    )

    # per nnx.Dropout - disable (deterministic=True) for eval,
    # keep (False) for training
    # Add & Norm after attention
    proj_input = self.layernorm_1(
        inputs + self.dropout_1(attention_output, deterministic=deterministic)
    )

    # Feedforward (MLP) layer
    proj_output = self.feedforward(proj_input)

    # Add & Norm after feedforward
    return self.layernorm_2(
        proj_input + self.dropout_2(proj_output, deterministic=deterministic)
    )


class TransformerEncoder(nnx.Module):
  """Transformer encoder consisting of multiple encoder blocks.

  Attributes:
      embed_dim: The dimension of the embedding.
      feedforward_dim: The dimension of the feedforward layer.
      num_heads: The number of attention heads.
      num_layers: The number of encoder layers.
      dropout_rate: The dropout rate.
      attn_dim: The dimension of the attention layer.
      rngs: The random number generators.
  """

  def __init__(
      self,
      embed_dim: int,
      feedforward_dim: int,
      num_heads: int,
      num_layers: int,
      dropout_rate: float,
      attn_dim: int,
      rngs: nnx.Rngs,
  ):
    self.embed_dim = embed_dim
    self.feedforward_dim = feedforward_dim
    self.num_heads = num_heads
    self.num_layers = num_layers  # Number of layers to stack
    self.attn_dim = attn_dim
    self.dropout_rate = dropout_rate

    # Create a list of encoder layers
    self.layers = [
        TransformerEncoderBlock(
            embed_dim, feedforward_dim, num_heads, dropout_rate, attn_dim, rngs
        )
        for _ in range(num_layers)
    ]

  def __call__(self, inputs, mask=None, deterministic: bool = False):
    # Pass the inputs through all layers sequentially
    x = inputs
    for layer in self.layers:
      x = layer(x, mask, deterministic=deterministic)
    return x


class PositionalEmbedding(nnx.Module):
  """Adds positional embeddings to input token embeddings.

  Attributes:
      sequence_length: The length of the input sequence.
      vocab_size: The size of the vocabulary.
      embed_dim: The dimension of the embeddings.
      rngs: The random number generators.
  """

  def __init__(
      self,
      sequence_length: int,
      vocab_size: int,
      embed_dim: int,
      rngs: nnx.Rngs,
  ):
    self.token_embeddings = nnx.Embed(
        num_embeddings=vocab_size, features=embed_dim, rngs=rngs
    )
    self.position_embeddings = nnx.Embed(
        num_embeddings=sequence_length, features=embed_dim, rngs=rngs
    )
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim

  def __call__(self, inputs):
    length = inputs.shape[1]
    positions = jnp.arange(0, length)[None, :]
    embedded_tokens = self.token_embeddings(inputs)
    embedded_positions = self.position_embeddings(positions)
    return embedded_tokens + embedded_positions

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    else:
      return jnp.not_equal(inputs, 0)


class TransformerDecoderBlock(nnx.Module):
  """Transformer decoder block consisting of multiple sub-layers.

  Attributes:
      embed_dim: The dimension of the embedding.
      feedforward_dim: The dimension of the feedforward layer.
      num_heads: The number of attention heads.
      dropout_rate: The dropout rate.
      attn_dim: The dimension of the attention layer.
      rngs: The random number generators.
  """

  def __init__(
      self,
      embed_dim: int,
      feedforward_dim: int,
      num_heads: int,
      dropout_rate: float,
      attn_dim: int,
      rngs: nnx.Rngs,
  ):
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.feedforward_dim = feedforward_dim
    self.dropout_rate = dropout_rate
    self.attn_dim = attn_dim

    # Self-attention layer
    self.attention_1 = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_dim,
        decode=False,
        qkv_features=attn_dim,
        rngs=rngs,
    )

    # Cross-attention layer (attending to encoder outputs)
    self.attention_2 = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_dim,
        decode=False,
        qkv_features=attn_dim,
        rngs=rngs,
    )

    # MLP layer (feedforward)
    self.feedforward = nnx.Sequential(
        nnx.Linear(
            embed_dim,
            feedforward_dim,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        ),
        nnx.relu,
        nnx.Linear(
            feedforward_dim,
            embed_dim,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        ),
    )

    # Layer Normalization
    self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.layernorm_3 = nnx.LayerNorm(embed_dim, rngs=rngs)

    # Dropout layers for regularization
    self.dropout_1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    self.dropout_2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    self.dropout_3 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

  def __call__(
      self,
      inputs,
      encoder_outputs,
      self_attention_mask=None,
      cross_attention_mask=None,
      deterministic: bool = False,
  ):

    # First attention (self-attention)
    attention_output_1 = self.attention_1(
        inputs_q=inputs,
        inputs_v=inputs,
        inputs_k=inputs,
        mask=self_attention_mask,
    )
    out_1 = self.layernorm_1(
        inputs + self.dropout_1(attention_output_1, deterministic=deterministic)
    )

    # Second attention (cross-attention with encoder outputs)
    attention_output_2 = self.attention_2(
        inputs_q=out_1,
        inputs_v=encoder_outputs,
        inputs_k=encoder_outputs,
        mask=cross_attention_mask,
    )
    out_2 = self.layernorm_2(
        out_1 + self.dropout_2(attention_output_2, deterministic=deterministic)
    )

    # Feedforward projection
    proj_output = self.feedforward(out_2)
    return self.layernorm_3(
        out_2 + self.dropout_3(proj_output, deterministic=deterministic)
    )


class TransformerDecoder(nnx.Module):
  """Transformer decoder consisting of multiple decoder blocks.

  Attributes:
      embed_dim: The dimension of the embedding.
      feedforward_dim: The dimension of the feedforward layer.
      num_heads: The number of attention heads.
      num_layers: The number of decoder layers.
      dropout_rate: The dropout rate.
      attn_dim: The dimension of the attention layer.
      rngs: The random number generators.
  """

  def __init__(
      self,
      embed_dim: int,
      feedforward_dim: int,
      num_heads: int,
      num_layers: int,
      dropout_rate: float,
      attn_dim: int,
      rngs: nnx.Rngs,
  ):
    self.embed_dim = embed_dim
    self.feedforward_dim = feedforward_dim
    self.num_heads = num_heads
    self.num_layers = num_layers  # Number of layers in the decoder
    self.dropout_rate = dropout_rate
    self.attn_dim = attn_dim

    # Create a list of decoder layers
    self.layers = [
        TransformerDecoderBlock(
            embed_dim, feedforward_dim, num_heads, dropout_rate, attn_dim, rngs
        )
        for _ in range(num_layers)
    ]

  def __call__(
      self,
      inputs,
      encoder_outputs,
      self_attention_mask=None,
      cross_attention_mask=None,
      deterministic: bool = False,
  ):
    # Pass the inputs through all decoder layers sequentially
    x = inputs
    for layer in self.layers:
      x = layer(
          x,
          encoder_outputs,
          self_attention_mask,
          cross_attention_mask,
          deterministic=deterministic,
      )
    return x


class TransformerModel(nnx.Module):
  """Complete Transformer model with encoder and decoder.

  Attributes:
      num_encoder: The number of encoder layers.
      num_decoder: The number of decoder layers.
      enc_seq_len: The length of the encoder input sequence.
      dec_seq_len: The length of the decoder input sequence.
      vocab_size: The size of the vocabulary.
      embed_dim: The dimension of the embedding.
      feedforward_dim: The dimension of the feedforward layer.
      num_heads: The number of attention heads.
      dropout_rate: The dropout rate.
      attn_dim: The dimension of the attention layer.
      rngs: The random number generators.
  """

  def __init__(
      self,
      num_encoder: int,
      num_decoder: int,
      enc_seq_len: int,
      dec_seq_len: int,
      vocab_size: int,
      embed_dim: int,
      feedforward_dim: int,
      num_heads: int,
      dropout_rate: float,
      attn_dim: int,
      rngs: nnx.Rngs,
  ):
    self.pad_token = 0
    self.enc_seq_len = enc_seq_len
    self.dec_seq_len = dec_seq_len
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.feedforward_dim = feedforward_dim
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.attn_dim = attn_dim

    if num_encoder > 0:
      self.encoder = TransformerEncoder(
          embed_dim=embed_dim,
          feedforward_dim=feedforward_dim,
          num_heads=num_heads,
          num_layers=num_encoder,
          dropout_rate=dropout_rate,
          attn_dim=attn_dim,
          rngs=rngs,
      )

      self.encoder_positional_embedding = PositionalEmbedding(
          enc_seq_len, vocab_size, embed_dim, rngs=rngs
      )
    else:
      self.encoder = None
      self.encoder_positional_embedding = None

    self.decoder_positional_embedding = PositionalEmbedding(
        dec_seq_len, vocab_size, embed_dim, rngs=rngs
    )

    self.decoder = TransformerDecoder(
        embed_dim=embed_dim,
        feedforward_dim=feedforward_dim,
        num_heads=num_heads,
        num_layers=num_decoder,
        dropout_rate=dropout_rate,
        attn_dim=attn_dim,
        rngs=rngs,
    )

    # Final linear layer to generate logits
    self.dense = nnx.Linear(
        embed_dim,
        vocab_size,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
    )

  def __call__(
      self,
      encoder_inputs: jnp.ndarray,
      decoder_inputs: jnp.ndarray,
      deterministic: bool = False,
  ):
    if self.encoder is not None:
      # Apply positional embedding to encoder inputs
      x = self.encoder_positional_embedding(encoder_inputs)

      encoder_mask = nnx.make_attention_mask(
          encoder_inputs != self.pad_token,
          encoder_inputs != self.pad_token,
          dtype=bool,
      )
      encoder_outputs = self.encoder(
          x, mask=encoder_mask, deterministic=deterministic
      )

      # decoder attends over encoder_output, but mask is computed from
      # encoder_inputs (the paddings in the inputs)
      encoder_decoder_mask = nnx.make_attention_mask(
          decoder_inputs != self.pad_token,  # Queries
          encoder_inputs != self.pad_token,  # Keys,
          dtype=bool,
      )
    else:
      encoder_outputs = None
      encoder_decoder_mask = None

    # Apply positional embedding to decoder inputs
    y = self.decoder_positional_embedding(decoder_inputs)

    decoder_padding_mask = nnx.make_attention_mask(
        decoder_inputs != self.pad_token,
        decoder_inputs != self.pad_token,
        dtype=bool,
    )
    decoder_causal_mask = nnx.make_causal_mask(decoder_inputs)
    decoder_mask = nnx.combine_masks(decoder_padding_mask, decoder_causal_mask)

    # Pass the decoder input, encoder outputs, and the combined mask to
    # the decoder
    decoder_outputs = self.decoder(
        y,
        encoder_outputs,
        decoder_mask,
        encoder_decoder_mask,
        deterministic=deterministic,
    )

    # Final linear transformation to generate logits
    logits = self.dense(decoder_outputs)

    return logits
