from absl.testing import absltest
import jax.numpy as jnp
from parallax.examples.gpt_oss import config as config_lib
from parallax.examples.gpt_oss import moe_layer as moe_layer_lib


class MoELayerTest(absltest.TestCase):

  def test_moe_layer_forward_pass(self):
    config = config_lib.Config(
        embed=256,
        q_heads=4,
        kv_heads=4,
        num_layers=2,
        head_dim=64,
        vocab_size=1024,
        max_seq_len=1024,
        moe_num_experts=2,
        moe_experts_per_tok=1,
    )
    moe_layer = moe_layer_lib.MoELayer(config)
    dummy_input = jnp.ones((1, 10, config.embed))
    output = moe_layer(dummy_input)
    self.assertEqual(output.shape, (1, 10, config.embed))


if __name__ == '__main__':
  absltest.main()
