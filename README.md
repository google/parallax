# Parallax

**Parallax** is a library for automatically scaling JAX models. It simplifies
the process of training large models by offering automatic parallelism
strategies and memory optimizations, allowing you to focus on your model
architecture rather than sharding configurations.

**Parallax helps you:**

*   **Automatically shard** your JAX models and functions without manually
    defining `PartitionSpec`s.
*   **Apply advanced parallelism** strategies like Fully Sharded Data Parallel
    (FSDP) and Distributed Data Parallel (DDP) with ease.
*   **Optimize memory usage** through model offloading (keeping weights in CPU
    RAM) and rematerialization.

With Parallax, you can scale off-the-shelf JAX models to run on larger hardware
configurations or fit larger models on existing hardware without extensive code
modifications.

> This is not an officially supported Google product. This project is not
> eligible for the
> [Google Open Source Software Vulnerability Rewards Program](
  https://bughunters.google.com/open-source-security).

## Installation

You can install Parallax using pip:

```bash
pip install google-parallax
```

## Usage

Parallax integrates seamlessly with [Flax
NNX](https://flax.readthedocs.io/en/latest/index.html) models. Here is a simple
example of how to use auto-sharding:

```python
import parallax
from flax import nnx
import jax
import jax.numpy as jnp

model  = parallax.create_sharded_model(
    model_or_fn=lambda: Model(...),
    sample_inputs=(jnp.ones((1, 32)),),
    strategy=parallax.ShardingStrategy.AUTO,
    data_axis_name='fsdp',
    model_axis_name='tp',
)
```

## Features

*   **AutoSharding**: Automatically discover optimal sharding strategies.
*   **FSDP & DDP**: Ready-to-use implementations of common parallel training
    strategies.
*   **Model Offloading**: Stream model weights from CPU to device memory to
    train larger models.
*   **Rematerialization**: Automatic activation recomputation to save memory.

## Contributing

We welcome contributions! Please check
[`docs/contributing.md`](docs/contributing.md) for details on how to submit
pull requests and report bugs.

## Support

If you encounter any issues, please report them on our [GitHub
Issues](https://github.com/google/parallax/issues) page.
