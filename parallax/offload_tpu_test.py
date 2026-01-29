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

"""Tests for OffloadModel functionality on TPU."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from parallax import offload
from parallax.examples.gpt_oss import config as gpt_config_lib
from parallax.examples.gpt_oss import moe_layer


class OffloadModelTpuTest(parameterized.TestCase):

  def test_offload_model(self):
    """Ensures numerics between offloaded and non-offloaded models match."""
    make_model = lambda: nnx.Sequential(nnx.Linear(2, 2, rngs=nnx.Rngs(1)))
    mesh = jax.make_mesh(
        (1,), ('x',), axis_types=(jax.sharding.AxisType.Auto,) * len(('x',))
    )
    with mesh:
      graphdef, state = offload.create_offloaded_model(make_model)
      self.assertIsInstance(graphdef, nnx.GraphDef)
      self.assertIsInstance(state, nnx.State)
      self.assertEqual(
          state['layers'][0]['kernel'].sharding.memory_kind, 'pinned_host'
      )

    s_host = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x'), memory_kind='pinned_host'
    )
    s_dev = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))

    @functools.partial(
        jax.jit,
        in_shardings=(None, s_host, s_dev),
        out_shardings=s_host,
    )
    def run_offloaded(graphdef, state, inputs):
      model = nnx.merge(graphdef, state)
      offload.offload_model(model, s_dev)
      return model(inputs)

    # Create inputs on device
    inputs = jax.device_put(jnp.array([[0.0, 1.0], [1.0, 0.0]]), s_dev)
    outputs = run_offloaded(graphdef, state, inputs)

    self.assertEqual(outputs.sharding.memory_kind, 'pinned_host')

    reference_outputs = make_model()(inputs)
    np.testing.assert_allclose(reference_outputs, outputs, atol=1e-2)

    compiled_forward = run_offloaded.lower(graphdef, state, inputs).compile()
    _, state_sharding, input_sharding = compiled_forward.input_shardings[0]
    layer_sharding = state_sharding['layers'][0]
    self.assertEqual(layer_sharding['kernel'].memory_kind, 'pinned_host')
    self.assertEqual(layer_sharding['bias'].memory_kind, 'pinned_host')
    self.assertEqual(input_sharding.memory_kind, 'device')

  def test_offload_model_with_moe(self):
    """Ensures a model with MoE layers can be offloaded."""

    config = gpt_config_lib.Config(
        embed=2,
        q_heads=2,
        kv_heads=2,
        num_layers=2,
        head_dim=2,
        vocab_size=2,
        max_seq_len=2,
        moe_num_experts=2,
        moe_experts_per_tok=1,
    )
    make_model = lambda: moe_layer.MoELayer(config)

    with jax.make_mesh(
        (1,), ('x',), axis_types=(jax.sharding.AxisType.Auto,) * len(('x',))
    ):
      graphdef, state = offload.create_offloaded_model(make_model)
      self.assertIsInstance(graphdef, nnx.GraphDef)
      self.assertIsInstance(state, nnx.State)
      self.assertEqual(
          state['router']['kernel'].sharding.memory_kind, 'pinned_host'
      )

  def test_moe_selective_offload(self):
    """Tests that MoELayerPallas selectively offloads expert parameters."""
    config = gpt_config_lib.Config(
        embed=2,
        q_heads=2,
        kv_heads=2,
        num_layers=2,
        head_dim=2,
        vocab_size=2,
        max_seq_len=2,
        moe_num_experts=2,
        moe_experts_per_tok=1,
    )
    model = moe_layer.MoELayer(config)

    # Set router to always select expert 0.
    model.router.kernel.value = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    model.router.bias.value = jnp.array([10.0, 0.0])

    # Set weights of expert 1 to NaN.
    model.we_gate.value = model.we_gate.value.at[1].set(jnp.nan)
    model.we_up.value = model.we_up.value.at[1].set(jnp.nan)
    model.we_down.value = model.we_down.value.at[1].set(jnp.nan)

    mesh = jax.make_mesh(
        (1,), ('x',), axis_types=(jax.sharding.AxisType.Auto,) * len(('x',))
    )
    s_dev = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    inputs = jnp.ones((2, 2, 2))

    @jax.jit
    def run_offloaded(model, inputs):
      offload.offload_method(model, '__call__', s_dev=s_dev)
      return model(inputs)

    output = run_offloaded(model, inputs)

    # If expert 1's weights were loaded, the output would be NaN.
    self.assertFalse(np.isnan(output).any())


if __name__ == '__main__':
  absltest.main()
