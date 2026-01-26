"""Helper methods for Parallax demos."""

import time

import jax
import jax.numpy as jnp


def print_hbm():
  """Displays working HBM for each device."""
  for i, device in enumerate(jax.devices()):
    memory_stats = device.memory_stats()
    percent_used = (
        memory_stats['bytes_in_use'] / memory_stats['bytes_limit'] * 100
    )
    print(
        f'Device{i}:'
        f' {memory_stats["bytes_in_use"] / 1e9:.1f}GB/{memory_stats["bytes_limit"] / 1e9:.1f}GB'
        f' ({percent_used:.01f}%) | ',
        end='',
    )
  print()


def clear_hbm(log: bool = False):
  """Clears JAX allocated HBM. Useful for testing in Colab."""
  live_arrays_before = len(jax.live_arrays())
  for x in jax.live_arrays():
    x.delete()
  live_arrays_after = len(jax.live_arrays())
  if log:
    print(
        f'Reduced live arrays from {live_arrays_before} to {live_arrays_after}'
    )


def formatted_memory_stats() -> dict[str, str]:
  """Returns a dictionary of formatted memory stats for each device."""
  stats = {}
  total_bytes_in_use = 0
  total_bytes_limit = 0
  for device in jax.devices():
    key = f'{device.id}'
    memory_stats = device.memory_stats()
    if not memory_stats:
      continue
    bytes_in_use = memory_stats['bytes_in_use']
    bytes_limit = memory_stats['bytes_limit']
    total_bytes_in_use += bytes_in_use
    total_bytes_limit += bytes_limit
    percent_used = bytes_in_use / bytes_limit * 100
    value = '%.1f%% (%.1f/%.1fGB)' % (
        percent_used,
        bytes_in_use / 1e9,
        bytes_limit / 1e9,
    )
    stats[key] = value

  if total_bytes_limit > 0:
    stats['total HBM'] = '%.1f%% (%.1f/%.1fGB)' % (
        total_bytes_in_use / total_bytes_limit * 100,
        total_bytes_in_use / 1e9,
        total_bytes_limit / 1e9,
    )

  return stats


def make_gpt_inputs(batch_size: int, max_len=1024):
  return (
      jnp.array([[42] * max_len] * batch_size),
      jnp.ones((batch_size, max_len), dtype=jnp.int32),
  )


class Timer:
  """Simple timer for performance benchmarking."""

  def __init__(self, timings_list: list[float] | None = None):
    self.timings_list = [] if timings_list is None else timings_list
    self.start_time = None

  def __enter__(self):
    self.start_time = time.perf_counter()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    end_time = time.perf_counter()
    duration_ms = (end_time - self.start_time) * 1000
    if self.timings_list is not None:
      self.timings_list.append(duration_ms)
