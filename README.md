# Parallax

[![PyPI](https://img.shields.io/pypi/v/google-parallax)](https://pypi.org/project/google-parallax/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/parallax/blob/main/docs/getting_started.ipynb)

Parallax is a library for easily scaling off-the-shelf JAX models. It includes
optimizations in two main categories:

*   **Parallelism**:
    *   **AutoSharding**: Automatically determine `PartitionSpecs` for JAX
        Models and Functions.
    *   **Sharding Recipes**: Apply Distributed Data Parallel (DDP) or Fully
        Sharded Data Parallel (FSDP) to JAX models.
*   **Memory optimizations**:
    *   **Model Offloading**: Keep model weights in CPU RAM and stream to
        TPU/GPU HBM as needed.
    *   **Rematerialization**: Intentionally recompute activations to save HBM
        space.

The ethos of Parallax is **humans shouldn't be writing their own
[`Mesh`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh) or
[`PartitionSpec`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec)**
&mdash; computers are much better at discovering optimal sharding strategies, so
let them do the hard work.

> This is not an officially supported Google product. This project is not
> eligible for the
> [Google Open Source Software Vulnerability Rewards Program](
  https://bughunters.google.com/open-source-security).

## Why Parallax?

With plain JAX, you write sharding rules for every parameter by hand. Parallax
does that work for you and lets you take over where you need to:

| Task                       | Manual JAX                                        | Parallax                                              |
| -------------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| Choose parameter shardings | Write `PartitionSpec` rules by hand               | `strategy=parallax.ShardingStrategy.AUTO`             |
| Apply DDP or FSDP          | Write sharding rules for each model               | `strategy=parallax.ShardingStrategy.DDP` (or `.FSDP`) |
| Rematerialize activations  | Wrap layers with [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) | `parallax.remat_model(model)` (sequential models) |
| Offload weights            | Manage [memory kinds](https://docs.jax.dev/en/latest/201/memory-spaces.html#building-blocks) and transfers | `parallax.offload_model(model, ...)` |

Parallax is most useful for large models, or when you want to try different
parallelism strategies without rewriting your model.

## Install Parallax

Install the latest release of Parallax from
[PyPI](https://pypi.org/project/google-parallax/):

```sh
pip install google-parallax
```

## Get started

To shard your first model with Parallax, see
[Getting Started with Parallax](docs/getting_started.ipynb). The guide also
shows you how to run on multi-host clusters.

## Support

For questions, bug reports, or feature requests, please open an issue on
[GitHub Issues](https://github.com/google/parallax/issues).

## Contribute

To contribute to Parallax, see the [contributing guide](docs/contributing.md).

## Citing Parallax

To cite Parallax please use this citation:

```bibtex
@software{parallax2026,
  title = {{Parallax}},
  author = {Carpenter, Jeff and Yan, Stella and Liu, Dangyi and others},
  year = {2026},
  url = {https://github.com/google/parallax},
}
```
