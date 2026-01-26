"""Tests for Parallax example training."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from flax import nnx
import jax
from parallax.examples import configs
from parallax.examples import trainer

jax.config.update('jax_threefry_partitionable', False)


class TrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = jax.make_mesh(
        (4, 2),
        ('fsdp', 'tp'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('fsdp', 'tp')),
    )
    self.enter_context(jax.set_mesh(self.mesh))

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_mlp',
          config_fn=configs.default_config,
      ),
      dict(
          testcase_name='simple_mlp_auto_sharded',
          config_fn=configs.simple_mlp_auto_sharded,
      ),
      dict(
          testcase_name='mini_gpt',
          config_fn=configs.mini_gpt,
      ),
      dict(
          testcase_name='gemma_tiny',
          config_fn=configs.gemma_tiny,
      ),
  )
  def test_train(self, config_fn):
    configurable = config_fn()
    configurable.train_total_steps = 5
    config = fdl.build(configurable)

    var_before = jax.tree_util.tree_leaves(nnx.state(config.model))[0]

    trainer.train(config)

    # Assert training updated weights.
    var_after = jax.tree_util.tree_leaves(nnx.state(config.model))[0]
    self.assertNotEqual(var_before.flatten()[0], var_after.flatten()[0])


if __name__ == '__main__':
  # Fake 8 CPUs for testing.
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
  absltest.main()
