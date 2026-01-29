# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for OffloadModel functionality."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
from parallax import offload
from parallax.examples import models


class OffloadModelTest(parameterized.TestCase):

  def test_offload_train_step(self):
    """Ensures numerics match between chunked and un-chunked workflows."""
    reference_model = models.SimpleMLP(din=2, dmid=10, dout=2)
    optimizer = nnx.Optimizer(reference_model, optax.adam(1e-3), wrt=nnx.Param)
    inputs = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    labels = jnp.array([1, 1])

    @nnx.jit
    def loss_fn(model, inputs):
      logits = model(inputs)
      loss = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits,
          labels=labels,
      ).mean()
      return loss, logits

    (expected_loss, expected_outputs), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(reference_model, inputs)
    optimizer.update(reference_model, grads)

    actual_model = models.SimpleMLP(din=2, dmid=10, dout=2)
    outputs, loss = offload.offload_train_step(
        model=actual_model,
        loss_fn=optax.softmax_cross_entropy_with_integer_labels,
        optimizer=optimizer,
        labels=labels,
        inputs=inputs,
    )
    np.testing.assert_array_equal(outputs, expected_outputs)
    np.testing.assert_array_equal(expected_loss, loss)
    # Assert all model weights match after gradient update.
    np.testing.assert_array_equal(
        nnx.state(reference_model),
        nnx.state(actual_model),
    )

  def test_remat_model(self):
    """Ensures numerics match between rematted and non-rematted model."""
    inputs = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    labels = jnp.array([1, 1])

    @nnx.jit
    def loss_fn(model, inputs):
      logits = model(inputs)
      loss = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits,
          labels=labels,
      ).mean()
      return loss, logits

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    reference_model = models.SimpleMLP(din=2, dmid=10, dout=2)
    reference_optimizer = nnx.Optimizer(
        reference_model, optax.adam(1e-3), wrt=nnx.Param
    )
    (exp_loss, exp_outputs), exp_grads = grad_fn(reference_model, inputs)
    reference_optimizer.update(reference_model, exp_grads)

    actual_model = offload.remat_model(
        models.SimpleMLP(din=2, dmid=10, dout=2),
    )
    actual_optimizer = nnx.Optimizer(
        actual_model, optax.adam(1e-3), wrt=nnx.Param
    )
    (act_loss, act_outputs), act_grads = grad_fn(actual_model, inputs)
    actual_optimizer.update(actual_model, act_grads)

    np.testing.assert_array_equal(act_outputs, exp_outputs)
    np.testing.assert_array_equal(act_loss, exp_loss)
    # Assert all model weights match after gradient update.
    np.testing.assert_allclose(
        actual_model.layers[0].kernel.value,
        reference_model.layers[0].kernel.value,
        atol=1e-3,
        rtol=1e-1,
    )


if __name__ == '__main__':
  absltest.main()
