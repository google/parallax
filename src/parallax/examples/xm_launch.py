r"""XManager script for launching Parallax experiments.

gxm third_party/py/parallax/examples/xm_launch.py \
    --xm_resource_alloc=cml/cml-shared-ml-user
"""

from absl import app
from xmanager import xm
from xmanager import xm_abc


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with xm_abc.create_experiment(experiment_title='Parallax') as experiment:
    [executable] = experiment.package([
        xm.bazel_binary(
            label='//third_party/py/parallax/examples:main',
            executor_spec=xm_abc.Borg.Spec(),
        ),
    ])

    experiment.add(
        xm.Job(
            executable,
            args={},
            executor=xm_abc.Borg(
                xm.JobRequirements(dragonfish='4x4'),
            ),
        ),
    )


if __name__ == '__main__':
  app.run(main)
