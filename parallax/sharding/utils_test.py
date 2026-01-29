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

"""Tests for OffloadModel funtionality."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from parallax.sharding import utils


class ShardingUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single_device',
          device_count=1,
          expected_mesh_shape={'model': 1},
      ),
      dict(
          testcase_name='two_devices',
          device_count=2,
          expected_mesh_shape={'model': 2},
      ),
      dict(
          testcase_name='four_devices',
          device_count=4,
          expected_mesh_shape={'model': 4},
      ),
      dict(
          testcase_name='eight_devices',
          device_count=8,
          expected_mesh_shape={'data': 4, 'model': 2},
      ),
  )
  def test_auto_mesh(self, device_count, expected_mesh_shape):
    devices = jax.devices()[:device_count]
    mesh = utils.auto_mesh(devices)
    self.assertDictEqual(mesh.shape, expected_mesh_shape)


if __name__ == '__main__':
  # Fake 8 CPU devices; tests use a subset of these.
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
  absltest.main()
