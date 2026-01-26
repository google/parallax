# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from flax import nnx
import jax
import jax.numpy as jnp
from parallax.examples import models
from parallax.sharding import auto_shard

from absl.testing import absltest
from absl.testing import parameterized

NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


class AutoShardTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = jax.make_mesh(
        (4, 2),
        ('data', 'model'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('data', 'model')),
    )
    self.enter_context(jax.set_mesh(self.mesh))

  @parameterized.named_parameters(
      dict(
          testcase_name='dot_general',
          fn=lambda x, y: x @ y,
          inputs_fn=lambda: (jnp.ones((2, 3)), jnp.ones((3, 4))),
          expected_assignments=(([0, 1], [1, 2]), [0, 2]),
      ),
      dict(
          testcase_name='reshape',
          fn=lambda x: x.reshape((2, 2, 4, 2, 2)),
          inputs_fn=lambda: (jnp.ones((1, 2, 16, 2)),),
          expected_assignments=(([0, 1, 2, 3],), [1, 4, 5, 6, 3]),
      ),
      dict(
          testcase_name='add',
          fn=lambda x, y: x + y,
          inputs_fn=lambda: (jnp.ones((4, 4)), jnp.ones((4, 4))),
          expected_assignments=(([0, 1], [0, 1]), [0, 1]),
      ),
      dict(
          testcase_name='take',
          fn=lambda x, indices: jnp.take(x, indices, axis=0),
          inputs_fn=lambda: (
              jnp.ones((999, 256)),
              jnp.ones((3, 5), dtype=jnp.int32),
          ),
          expected_assignments=(([0, 1], [2, 3]), [2, 3, 1]),
      ),
      dict(
          testcase_name='relu',
          fn=nnx.relu,
          inputs_fn=lambda: (jnp.ones((999, 256)),),
          expected_assignments=(([0, 1],), [0, 1]),
      ),
      dict(
          testcase_name='slice',
          fn=lambda x: x[1:3, 4:6],
          inputs_fn=lambda: (jnp.ones((5, 7)),),
          expected_assignments=(([0, 1],), [2, 3]),
      ),
      dict(
          testcase_name='slice_preserve_sharding',
          fn=lambda x: x[1:3, :],
          inputs_fn=lambda: (jnp.ones((5, 7)),),
          expected_assignments=(([0, 1],), [2, 1]),
      ),
      dict(
          testcase_name='concatenate',
          fn=lambda x, y: jnp.concatenate([x, y], axis=1),
          inputs_fn=lambda: (jnp.ones((4, 4)), jnp.ones((4, 4))),
          expected_assignments=(([0, 1], [0, 2]), [0, 3]),
      ),
      dict(
          testcase_name='split',
          fn=lambda x: jnp.split(x, 2, axis=1),
          inputs_fn=lambda: (jnp.ones((4, 4)),),
          expected_assignments=(([0, 1],), [[0, 2], [0, 3]]),
      ),
      dict(
          testcase_name='broadcast_in_dim',
          fn=lambda x: jnp.broadcast_to(x, (2, 4, 4)),
          inputs_fn=lambda: (jnp.ones((4, 4)),),
          expected_assignments=(([0, 1],), [2, 0, 1]),
      ),
      dict(
          testcase_name='broadcast_in_dim_gemma',
          fn=lambda x: jnp.broadcast_to(x, (1, 8, 4, 8)),
          inputs_fn=lambda: (jnp.ones((1, 1, 1, 8)),),
          expected_assignments=(([0, 1, 2, 3],), [4, 5, 6, 3]),
      ),
  )
  def test_analyze_same_axes(self, fn, inputs_fn, expected_assignments):
    inputs = inputs_fn()
    assignments = auto_shard.analyze_same_axes(fn, *inputs)
    self.assertEqual(assignments, expected_assignments)

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_mlp',
          model_fn=lambda: models.SimpleMLP(16, 64, 16, rngs=nnx.Rngs(0)),
          inputs_fn=lambda: (jnp.ones((4, 16)),),
          min_shard_size=0,
          expected_assignments=(
              {
                  'layers': {
                      0: {'bias': P('model'), 'kernel': P(None, 'model')},
                      1: {'bias': P(None), 'kernel': P('model', None)},
                  },
              },
              [P('data', None)],
              P('data', None),
          ),
      ),
      dict(
          testcase_name='mini_gpt',
          model_fn=lambda: models.MiniGPT(
              vocab_size=999,
              maxlen=128,
              feed_forward_dim=1024,
              rngs=nnx.Rngs(0),
              num_transformer_blocks=2,
          ),
          inputs_fn=lambda: (jnp.ones((3, 5), dtype=jnp.int32),),
          min_shard_size=0,
          expected_assignments=(
              {
                  'layers': {
                      0: {
                          'pos_emb': {'embedding': P(None, 'model')},
                          'token_emb': {'embedding': P(None, 'model')},
                      },
                      1: {
                          'layer_norm1': {
                              'bias': P('model'),
                              'scale': P('model'),
                          },
                          'layer_norm2': {
                              'bias': P('model'),
                              'scale': P('model'),
                          },
                          'linear1': {
                              'bias': P(None),
                              'kernel': P('model', None),
                          },
                          'linear2': {
                              'bias': P('model'),
                              'kernel': P(None, 'model'),
                          },
                          'mha': {
                              'key': {
                                  'bias': P(None, None),
                                  'kernel': P('model', None, None),
                              },
                              'out': {
                                  'bias': P('model'),
                                  'kernel': P(None, None, 'model'),
                              },
                              'query': {
                                  'bias': P(None, None),
                                  'kernel': P('model', None, None),
                              },
                              'value': {
                                  'bias': P(None, None),
                                  'kernel': P('model', None, None),
                              },
                          },
                      },
                      2: {
                          'layer_norm1': {
                              'bias': P('model'),
                              'scale': P('model'),
                          },
                          'layer_norm2': {
                              'bias': P('model'),
                              'scale': P('model'),
                          },
                          'linear1': {
                              'bias': P(None),
                              'kernel': P('model', None),
                          },
                          'linear2': {
                              'bias': P('model'),
                              'kernel': P(None, 'model'),
                          },
                          'mha': {
                              'key': {
                                  'bias': P(None, None),
                                  'kernel': P('model', None, None),
                              },
                              'out': {
                                  'bias': P('model'),
                                  'kernel': P(None, None, 'model'),
                              },
                              'query': {
                                  'bias': P(None, None),
                                  'kernel': P('model', None, None),
                              },
                              'value': {
                                  'bias': P(None, None),
                                  'kernel': P('model', None, None),
                              },
                          },
                      },
                      3: {
                          'bias': P(None),
                          'kernel': P('model', None),
                      },
                  }
              },
              [P('data0', 'data1')],
              P('data0', 'data1', None),
          ),
      ),
      dict(
          testcase_name='min_shard_size',
          model_fn=lambda: models.SimpleMLP(
              16,
              8,
              1024,
              rngs=nnx.Rngs(0),
          ),
          inputs_fn=lambda: (jnp.ones((256, 16), dtype=jnp.int32),),
          min_shard_size=128,
          expected_assignments=(
              {
                  'layers': {
                      0: {'bias': P(None), 'kernel': P(None, None)},
                      1: {'bias': P(None), 'kernel': P(None, None)},
                  },
              },
              [P('data', None)],
              P('data', None),
          ),
      ),
  )
  def test_auto_shard_on_model(
      self,
      model_fn,
      inputs_fn,
      expected_assignments,
      min_shard_size,
  ):
    model = model_fn()
    inputs = inputs_fn()
    graphdef, state = nnx.split(model)

    def fn(state, *inputs):
      model = nnx.merge(graphdef, state)
      return model(*inputs)

    (model_shd, *in_shd), out_shd = auto_shard.get_shardings(
        fn,
        state,
        *inputs,
        min_shard_size=min_shard_size,
    )
    model_shd = nnx.to_pure_dict(model_shd)
    self.assertEqual(model_shd, expected_assignments[0])
    self.assertEqual(in_shd, expected_assignments[1])
    self.assertEqual(out_shd, expected_assignments[2])


if __name__ == '__main__':
  # Fake 8 CPUs for testing.
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
  absltest.main()
