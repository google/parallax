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

"""Tests for DDP sharding."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from parallax.examples import models
from parallax.sharding import ddp

NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


class DdpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = jax.make_mesh(
        (4, 2),
        ('data', 'model'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('data', 'model')),
    )
    self.enter_context(jax.set_mesh(self.mesh))

  def test_get_shardings(self):
    model = models.SimpleMLP(16, 64, 16, rngs=nnx.Rngs(0))
    inputs = (jnp.ones((4, 16)),)
    graphdef, state = nnx.split(model)

    def fn(state, *inputs):
      return nnx.merge(graphdef, state)(*inputs)

    (params_assignments, inputs_assignments), output_assignments = (
        ddp.get_shardings(fn, state, *inputs)
    )

    expected_params = {
        'layers': {
            0: {'bias': P(None), 'kernel': P(None, None)},
            1: {'bias': P(None), 'kernel': P(None, None)},
        }
    }

    self.assertEqual(nnx.to_pure_dict(params_assignments), expected_params)
    self.assertEqual(inputs_assignments, P('data', None))
    self.assertEqual(output_assignments, P('data', None))


if __name__ == '__main__':
  # Fake 8 CPUs for testing.
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
  absltest.main()
