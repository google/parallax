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

"""Utility functions for sharding."""

from typing import Sequence

import jax


def auto_mesh(
    devices: Sequence[jax.Device] | None = None,
):
  """Creates a JAX mesh with a simple heuristic based on device count."""
  # TODO(jeffcarp): Make this smarter based on model and inputs.
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  if num_devices == 1:
    return jax.make_mesh((1,), ('model',), devices=devices)
  elif num_devices <= 4 or num_devices % 2 != 0:
    # Prefer 1D mesh for small counts or any odd counts.
    return jax.make_mesh((num_devices,), ('model',), devices=devices)
  else:
    return jax.make_mesh(
        (num_devices // 2, 2),
        ('data', 'model'),
        devices=devices,
    )
