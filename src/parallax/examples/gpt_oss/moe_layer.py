"""MoE layer implementation."""

from flax import nnx
import jax
from jax import numpy as jnp
from parallax.examples.gpt_oss import config as config_lib


class MoELayer(nnx.Module):
  """A truly sparse Mixture of Experts layer with offloading capability."""

  def __init__(self, config: config_lib.Config):
    super().__init__()
    self.config = config
    self.router = nnx.Linear(
        in_features=config.embed,
        out_features=config.moe_num_experts,
        rngs=nnx.Rngs(0),
    )
    self.we_gate = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (config.moe_num_experts, config.embed, config.embed),
        )
    )
    self.we_up = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (config.moe_num_experts, config.embed, config.embed),
        )
    )
    self.we_down = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (config.moe_num_experts, config.embed, config.embed),
        )
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    num_tokens, _ = x_flat.shape

    router_logits = self.router(x_flat)
    topk_weights, topk_indices = jax.lax.top_k(
        router_logits, k=self.config.moe_experts_per_tok
    )
    topk_weights = nnx.softmax(topk_weights, axis=-1)

    # Flatten topk_indices and create a corresponding token_idx array.
    token_idx = jnp.arange(num_tokens)
    token_idx_mesh = jnp.repeat(token_idx, self.config.moe_experts_per_tok)
    expert_idx_mesh = topk_indices.flatten()

    # Sort by expert index to group tokens.
    sort_permutation = jnp.argsort(expert_idx_mesh)
    sorted_expert_idx = expert_idx_mesh[sort_permutation]
    sorted_token_idx = token_idx_mesh[sort_permutation]

    # Group tokens for each expert.
    group_sizes = jnp.bincount(
        sorted_expert_idx, length=self.config.moe_num_experts
    )
    sorted_tokens = x_flat[sorted_token_idx]

    # Apply gate and up projection.
    # TODO(b/453671475): Set `ragged_dot_tiling` in xla_metadata here for better
    # performance.
    ff_gate = jax.lax.ragged_dot(sorted_tokens, self.we_gate.value, group_sizes)
    ff_up = jax.lax.ragged_dot(sorted_tokens, self.we_up.value, group_sizes)

    # Apply activation.
    activated_ff_gate = nnx.silu(ff_gate) * ff_up

    # Apply down projection.
    ff_out = jax.lax.ragged_dot(
        activated_ff_gate, self.we_down.value, group_sizes
    )

    # Un-sort to get back to original token order.
    output_flat = jnp.zeros_like(x_flat)
    output_flat = output_flat.at[sorted_token_idx].add(ff_out)

    # Weight the outputs.
    output_flat *= jnp.repeat(
        topk_weights.flatten(), self.config.moe_experts_per_tok
    )[..., None]

    return output_flat.reshape(x_shape)

  def offload_state(
      self, x: jax.Array, s_dev: jax.sharding.Sharding
  ) -> nnx.State:
    """Offloads only the router and active expert parameters."""
    router_logits = self.router(x.reshape(-1, x.shape[-1]))
    topk_indices = jax.lax.top_k(
        router_logits, k=self.config.moe_experts_per_tok
    )[1]
    expert_indices = jnp.unique(
        topk_indices, size=self.config.moe_experts_per_tok
    )

    state = nnx.state(self)
    dev_state = {}

    # Offload router state.
    dev_state['router'] = jax.tree.map(
        lambda l: jax.device_put(l, s_dev), state['router']
    )

    # Offload only activated experts.
    we_gate = state['we_gate']
    we_up = state['we_up']
    we_down = state['we_down']
    we_gate.value = we_gate.value.at[expert_indices].set(
        jax.device_put(we_gate.value[expert_indices], s_dev)
    )
    we_up.value = we_up.value.at[expert_indices].set(
        jax.device_put(we_up.value[expert_indices], s_dev)
    )
    we_down.value = we_down.value.at[expert_indices].set(
        jax.device_put(we_down.value[expert_indices], s_dev)
    )
    dev_state['we_gate'] = we_gate
    dev_state['we_up'] = we_up
    dev_state['we_down'] = we_down

    return dev_state
