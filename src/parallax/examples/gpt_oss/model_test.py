from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from parallax.examples.gpt_oss import config as config_lib
from parallax.examples.gpt_oss import model as model_lib


class GptOssTest(parameterized.TestCase):

  def test_model_forward_pass(self):
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
    model = model_lib.GptOss(config)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    output = model(dummy_input)
    self.assertEqual(output.shape, (1, 10, config.vocab_size))


if __name__ == '__main__':
  absltest.main()
