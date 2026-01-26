"""Parallax example trainer."""

import time
from flax import nnx
import jax
import optax
from parallax.examples import configs
from parallax.examples import utils
import tqdm


def compute_losses_and_logits(
    model: nnx.Module,
    inputs: tuple[jax.Array, ...],
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes the loss and logits for a given model and inputs."""
  outputs = model(*inputs)
  if isinstance(outputs, tuple):
    # TODO(jeffcarp): Handle Gemma KV cache.
    logits, _ = outputs
  else:
    logits = outputs
  loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits,
      labels=labels,
  ).mean()
  return loss, outputs


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    inputs: tuple[jax.Array, ...],
    labels: jax.Array,
):
  """Performs a single training step."""
  grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
  (loss, _), grads = grad_fn(model, inputs, labels)
  optimizer.update(grads)
  return loss


def _dump_hlo(
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    train_loader,
    filepath: str = '/tmp/train_hlo.txt',
):
  """Dumps the HLO of the train_step function to a file."""
  first_inputs, first_labels = next(iter(train_loader))
  model.train()
  traced = train_step.trace(model, optimizer, first_inputs, first_labels)
  lowered = traced.lower()

  compiled = lowered.compile()
  print('Cost analysis:', compiled.cost_analysis())

  with open(filepath, 'w') as f:
    f.write(lowered.as_text())
  print(f'HLO for train_step written to {filepath}')


def train(config: configs.TrainConfig):
  """Trains a model based on a config."""
  model = config.model
  start_time = time.time()
  _dump_hlo(model, config.optimizer, config.train_loader)

  progress_bar = tqdm.tqdm(
      enumerate(config.train_loader),
      total=config.train_total_steps,
  )
  progress_bar.set_postfix(
      {
          'loss': 'n/a',
          'steps/sec': 'n/a',
      }
      | utils.formatted_memory_stats()
  )
  for step, batch in progress_bar:
    inputs, labels = batch
    model.train()
    loss = train_step(model, config.optimizer, inputs, labels)
    jax.block_until_ready(loss)
    end_time = time.time()
    steps_per_second = 1 / (end_time - start_time)
    start_time = end_time
    progress_bar.set_postfix(
        {
            'loss': '{:.3f}'.format(loss.item()),
            'steps/sec': '{:.2f}'.format(steps_per_second),
        }
        | utils.formatted_memory_stats()
    )

    if step >= config.train_total_steps:
      break
