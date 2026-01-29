# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supports manual sharding of intermediates, allowing precise control over their distribution across devices."""

import functools

import jax
from jax.experimental import pjit
import numpy as np

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding

# TODO(jeffcarp): Integrate into main sharding APIs.


def apply_sharding_constraints(f, sharding_config):

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    def tagged(name, val):
      if name in sharding_config:
        val = jax.lax.with_sharding_constraint(val, sharding_config[name])
      return val

    return f(*args, **kwargs, tag=tagged)

  return wrapped


def shard_model(model_fn, params, sharding_config):
  """Shards a model across multiple devices using pjit.

  Args:
    model_fn: The model function to shard. It should accept input 'x' and
      parameters 'params'. It should return the output and a dictionary of
      intermediate values.
    params: A dictionary of model parameters.
    sharding_config: A dictionary specifying the sharding configuration for
      inputs, outputs, parameters, and intermediates, as well as the mesh axes.

  Returns:
    A tuple containing:
      - The output of the model.
      - A dictionary of intermediate values.
      - The device mesh used for sharding.
  """
  devices = np.array(jax.devices()[:4]).reshape(2, 2)
  mesh = Mesh(devices, sharding_config["mesh_axes"])

  pjit_model = pjit.pjit(
      model_fn,
      in_shardings=(sharding_config["in"], sharding_config["parameters"]),
      out_shardings=(
          sharding_config["out"],
          sharding_config["intermediates"],
      ),
  )

  with mesh:
    params = {
        k: jax.device_put(
            v, NamedSharding(mesh, sharding_config["parameters"][k])
        )
        for k, v in params.items()
    }
    # out, intermediates = pjit_model(x, params)

  return pjit_model, params, mesh
