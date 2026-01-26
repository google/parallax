"""Parallax library."""

from flax import nnx
import jax
from parallax import offload
from parallax.sharding import base
from parallax.sharding import utils


# Sharding APIs.
jit = base.jit
auto_mesh = utils.auto_mesh
create_sharded_model = base.create_sharded_model
ShardingStrategy = base.ShardingStrategy

# Model offloading utilities.
create_offloaded_model = offload.create_offloaded_model
offload_method = offload.offload_method
offload_model = offload.offload_model

offload_train_step = offload.offload_train_step
offload_forward = offload.offload_forward
offload_backward = offload.offload_backward
remat_model = offload.remat_model
