"""Manual sharding of Linen models using pjit."""

from flax import traverse_util
import jax
from jax import sharding
from jax.experimental import pjit
from parallax.sharding import utils

# TODO(b/452969631): Integrate into main sharding APIs.


def _flatten_tree(tree, prefix=()):
  """Flattens a nested dictionary into a flat dictionary with tuple keys.

  Args:
    tree: A nested dictionary.
    prefix: A tuple representing the current path in the nested dictionary.

  Returns:
    A flat dictionary with tuple keys.
  """
  flat = {}
  for k, v in tree.items():
    path = prefix + (k,)
    if isinstance(v, dict):
      flat.update(_flatten_tree(v, path))
    else:
      flat[path] = v
  return flat


def shard_linen_model(model, params, sharding_config):
  """Shards a Linen model according to a sharding configuration.

  Args:
    model: The Linen model.
    params: The model parameters.
    sharding_config: A dictionary containing the sharding configuration.

  Returns:
    A tuple containing:
      - The pjit-ed apply function.
      - The sharded model parameters.
      - The mesh.
  """

  mesh = utils.auto_mesh()

  flat_params = traverse_util.flatten_dict(params)
  flat_pspecs = _flatten_tree(sharding_config["parameters"])

  sharded_flat_params = {}
  with mesh:
    for k, v in flat_params.items():
      if k in flat_pspecs:
        pspec = flat_pspecs[k]
        sharded_flat_params[k] = jax.device_put(
            v, sharding.NamedSharding(mesh, pspec)
        )
        print(f"Sharded param {k} with {pspec}")
      else:
        print(f"[WARN] No sharding spec for param {k}, keeping unsharded")
        sharded_flat_params[k] = v
  sharded_params = traverse_util.unflatten_dict(sharded_flat_params)

  def apply_fn(params, x):
    y = model.apply(params, x)
    return y

  pjit_runner = pjit.pjit(
      apply_fn,
      in_shardings=(sharding_config["parameters"], sharding_config["in"]),
      out_shardings=(sharding_config["out"]),
  )

  return pjit_runner, sharded_params, mesh
