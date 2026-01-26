"""Entry point for Parallax training examples."""

from absl import app
import fiddle as fdl
from fiddle import absl_flags as fdl_flags
from parallax.examples import configs
from parallax.examples import trainer


_CONFIG_FLAG = fdl_flags.DEFINE_fiddle_config(
    "config",
    help_string="The name of the Fiddle config",
)


def main(_):
  config = fdl.build(_CONFIG_FLAG.value or configs.default_config())
  trainer.train(config)


if __name__ == "__main__":
  app.run(main)
