"""Configuration for GPT-OSS model."""

import dataclasses


@dataclasses.dataclass
class Config:
  """Configuration for GptOss model."""

  embed: int
  q_heads: int
  kv_heads: int
  num_layers: int
  head_dim: int
  vocab_size: int
  max_seq_len: int
  # MoE config
  moe_num_experts: int
  moe_experts_per_tok: int
  # RoPE config
  max_position_embeddings: int = 131072
  rope_theta: float = 500000.0
  rope_factor: float = 32.0
  rope_original_max_position_embeddings: int = 4096
  rope_beta_slow: float = 1.0
  rope_beta_fast: float = 32.0
