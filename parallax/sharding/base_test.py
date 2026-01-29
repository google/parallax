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

"""Tests for base sharding APIs."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from parallax.examples import models
from parallax.sharding import base

NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec
PartitionSpec = jax.sharding.PartitionSpec


class BaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = jax.make_mesh(
        (4, 2),
        ('data', 'model'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('data', 'model')),
    )
    self.enter_context(jax.set_mesh(self.mesh))

  def test_jit_decorator(self):
    model = models.SimpleMLPLinen(features=64)
    x = jnp.ones((4, 16))
    params = model.init(jax.random.PRNGKey(42), x)
    reference_output = model.apply(params, x)

    @base.jit
    def forward(params, inputs):
      return model.apply(params, inputs)

    output = forward(params, x)
    np.testing.assert_array_almost_equal(output, reference_output, decimal=6)

    # Verify compiled shardings.
    compiled = forward.lower(params, x).compile()  # type: ignore
    (param_shardings, x_shardings), _ = compiled.input_shardings
    expected_param_shardings = {
        'params': {
            'dense_1': {'bias': P('model'), 'kernel': P(None, 'model')},
            'dense_2': {'bias': P(None), 'kernel': P('model', None)},
        },
    }
    expected_param_shardings = jax.tree.map(
        lambda p: NamedSharding(self.mesh, spec=p),
        expected_param_shardings,
        is_leaf=lambda p: isinstance(p, PartitionSpec),
    )
    expected_input_shardings = NamedSharding(self.mesh, P('data', None))
    self.assertDictEqual(param_shardings, expected_param_shardings)
    self.assertEqual(x_shardings, expected_input_shardings)

  def test_jit_decorator_with_args(self):
    model = models.SimpleMLPLinen(features=64)
    x = jnp.ones((4, 16))
    params = model.init(jax.random.PRNGKey(42), x)
    reference_output = model.apply(params, x)

    @base.jit(strategy=base.ShardingStrategy.DDP)
    def forward(params, inputs):
      return model.apply(params, inputs)

    output = forward(params, x)
    np.testing.assert_array_almost_equal(output, reference_output, decimal=6)

    # Verify compiled shardings.
    compiled = forward.lower(params, x).compile()  # type: ignore
    (param_shardings, x_shardings), _ = compiled.input_shardings
    expected_param_shardings = {
        'params': {
            'dense_1': {'bias': P(None), 'kernel': P(None, None)},
            'dense_2': {'bias': P(None), 'kernel': P(None, None)},
        },
    }
    expected_param_shardings = jax.tree.map(
        lambda p: NamedSharding(self.mesh, spec=p),
        expected_param_shardings,
        is_leaf=lambda p: isinstance(p, jax.sharding.PartitionSpec),
    )
    expected_input_shardings = NamedSharding(self.mesh, P('data', None))
    self.assertDictEqual(param_shardings, expected_param_shardings)
    self.assertEqual(x_shardings, expected_input_shardings)

  def test_jit_function_call(self):
    model = models.SimpleMLPLinen(features=64)
    x = jnp.ones((4, 16))
    params = model.init(jax.random.PRNGKey(42), x)
    reference_output = model.apply(params, x)

    def forward(params, inputs):
      return model.apply(params, inputs)

    sharded_forward = base.jit(forward, strategy=base.ShardingStrategy.DDP)
    output = sharded_forward(params, x)
    np.testing.assert_array_almost_equal(output, reference_output, decimal=6)

    # Verify compiled shardings.
    compiled = sharded_forward.lower(params, x).compile()  # type: ignore
    (param_shardings, x_shardings), _ = compiled.input_shardings
    expected_param_shardings = {
        'params': {
            'dense_1': {'bias': P(None), 'kernel': P(None, None)},
            'dense_2': {'bias': P(None), 'kernel': P(None, None)},
        },
    }
    expected_param_shardings = jax.tree.map(
        lambda p: NamedSharding(self.mesh, spec=p),
        expected_param_shardings,
        is_leaf=lambda p: isinstance(p, jax.sharding.PartitionSpec),
    )
    expected_input_shardings = NamedSharding(self.mesh, P('data', None))
    self.assertDictEqual(param_shardings, expected_param_shardings)
    self.assertEqual(x_shardings, expected_input_shardings)

  @parameterized.named_parameters(
      dict(
          testcase_name='auto',
          strategy=base.ShardingStrategy.AUTO,
          expected_shardings={
              'params': {
                  'dense_1': {'bias': P('model'), 'kernel': P(None, 'model')},
                  'dense_2': {'bias': P(None), 'kernel': P('model', None)},
              }
          },
      ),
      dict(
          testcase_name='ddp',
          strategy=base.ShardingStrategy.DDP,
          expected_shardings={
              'params': {
                  'dense_1': {'bias': P(None), 'kernel': P(None, None)},
                  'dense_2': {'bias': P(None), 'kernel': P(None, None)},
              }
          },
      ),
      dict(
          testcase_name='fsdp',
          strategy=base.ShardingStrategy.FSDP,
          expected_shardings={
              'params': {
                  'dense_1': {'bias': P('model'), 'kernel': P(None, 'model')},
                  'dense_2': {'bias': P('model'), 'kernel': P(None, 'model')},
              }
          },
      ),
  )
  def test_create_sharded_model_linen(self, strategy, expected_shardings):
    sharded_vars = base.create_sharded_model(
        models.SimpleMLPLinen(features=64),
        sample_inputs=(jnp.ones((4, 16)),),
        strategy=strategy,
    )
    actual_shardings = jax.tree.map(lambda v: v.sharding.spec, sharded_vars)
    self.assertDictEqual(actual_shardings, expected_shardings)

  @parameterized.named_parameters(
      dict(
          testcase_name='auto',
          strategy=base.ShardingStrategy.AUTO,
          expected_shardings={
              'layers': {
                  0: {'bias': P('model'), 'kernel': P(None, 'model')},
                  1: {'bias': P(), 'kernel': P('model')},
              }
          },
      ),
      dict(
          testcase_name='ddp',
          strategy=base.ShardingStrategy.DDP,
          expected_shardings={
              'layers': {
                  0: {'bias': P(), 'kernel': P()},
                  1: {'bias': P(), 'kernel': P()},
              }
          },
      ),
      dict(
          testcase_name='fsdp',
          strategy=base.ShardingStrategy.FSDP,
          expected_shardings={
              'layers': {
                  0: {'bias': P('model'), 'kernel': P(None, 'model')},
                  1: {'bias': P('model'), 'kernel': P(None, 'model')},
              }
          },
      ),
  )
  def test_create_sharded_model_nnx(self, strategy, expected_shardings):
    sharded_model = base.create_sharded_model(
        lambda: models.SimpleMLP(16, 64, 16, rngs=nnx.Rngs(0)),
        sample_inputs=(jnp.ones((4, 16)),),
        strategy=strategy,
    )
    sharded_state = nnx.state(sharded_model)
    actual_shardings = nnx.to_pure_dict(
        jax.tree.map(lambda v: v.sharding.spec, sharded_state)
    )
    self.assertDictEqual(actual_shardings, expected_shardings)

  def test_sharded_nnx_optimizer_training_loop(self):
    """Ensures create_sharded_model works within an NNX training loop."""
    dummy_inputs, dummy_labels = jnp.ones((4, 16)), jnp.ones((4, 16))
    sharded_model = base.create_sharded_model(
        lambda: models.SimpleMLP(16, 64, 16, rngs=nnx.Rngs(0)),
        sample_inputs=(jnp.ones((4, 16)),),
        strategy=base.ShardingStrategy.AUTO,
    )
    optimizer = nnx.Optimizer(sharded_model, optax.adam(1e-3), wrt=nnx.Param)
    graphdef, model_state = nnx.split(sharded_model)

    def _state_to_pspecs(module):
      return nnx.to_pure_dict(
          jax.tree.map(lambda v: v.sharding.spec, nnx.state(module))
      )

    # Verify both model and optimizer are created with the correct shardings.
    model_shardings = _state_to_pspecs(sharded_model)
    opt_shardings = _state_to_pspecs(optimizer)
    expected_opt_shardings = {
        'opt_state': {
            0: {
                'count': PartitionSpec(),
                'mu': model_shardings,
                'nu': model_shardings,
            }
        },
        'step': PartitionSpec(),
    }
    self.assertDictEqual(opt_shardings, expected_opt_shardings)

    # Verify model and optimizer shardings are integrated into train step.
    def loss_fn(model, inputs, labels):
      logits = model(inputs)
      return jnp.mean((logits - labels) ** 2)

    @nnx.jit
    def train_step(state, optimizer, inputs, labels):
      model = nnx.merge(graphdef, state)
      grad_fn = nnx.value_and_grad(loss_fn)
      loss, grads = grad_fn(model, inputs, labels)
      optimizer.update(model, grads)
      return loss

    train_step(model_state, optimizer, dummy_inputs, dummy_labels)

    compiled = train_step.lower(
        model_state,
        optimizer,
        dummy_inputs,
        dummy_labels,
    ).compile()

    def _unwrap_nnx_node(pytree):
      pytree = jax.tree.map(
          lambda x: x.states[0] if isinstance(x, nnx.NodeStates) else x,
          pytree,
          is_leaf=lambda x: isinstance(x, nnx.NodeStates),
      )
      # Unwrap lists - might be able to be deduped
      pytree = jax.tree.map(
          lambda x: x[0], pytree, is_leaf=lambda x: isinstance(x, list)
      )
      pytree = jax.tree.map(
          lambda x: x.spec,
          pytree,
          is_leaf=lambda x: isinstance(x, jax.NamedSharding),
      )
      return nnx.to_pure_dict(pytree)

    # Model and optimizer sharding are correctly picked up in jitted inputs.
    self.assertDictEqual(
        _unwrap_nnx_node(compiled.input_shardings[0][0]),
        model_shardings,
    )

    unwrapped_pspecs = compiled.input_shardings[0][1].states[0]
    unwrapped_pspecs = jax.tree.map(
        lambda ns: ns.spec,
        unwrapped_pspecs,
        is_leaf=lambda ns: isinstance(ns, NamedSharding),
    )
    # Model shardings are reflected in compiled optimizer shardings.
    self.assertListEqual(
        unwrapped_pspecs,
        [
            PartitionSpec(),
            PartitionSpec('model'),
            PartitionSpec(None, 'model'),
            PartitionSpec(),
            PartitionSpec('model'),
            PartitionSpec('model'),
            PartitionSpec(None, 'model'),
            PartitionSpec(),
            PartitionSpec('model'),
            PartitionSpec(),
        ],
    )

    # Inputs and labels are not sharded.
    self.assertEqual(compiled.input_shardings[0][2].spec, PartitionSpec())
    self.assertEqual(compiled.input_shardings[0][3].spec, PartitionSpec())

  def test_create_sharded_model_axis_names(self):
    mesh = jax.make_mesh(
        (4, 2),
        ('apples', 'oranges'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('apples', 'oranges')),
    )
    with jax.set_mesh(mesh):
      sharded_model = base.create_sharded_model(
          lambda: models.SimpleMLP(16, 64, 16, rngs=nnx.Rngs(0)),
          sample_inputs=(jnp.ones((4, 16)),),
          strategy=base.ShardingStrategy.AUTO,
          data_axis_name='apples',
          model_axis_name='oranges',
      )
    sharded_state = nnx.state(sharded_model)
    actual_shardings = nnx.to_pure_dict(
        jax.tree.map(lambda v: v.sharding.spec, sharded_state)
    )
    expected_shardings = {
        'layers': {
            0: {'bias': P('oranges'), 'kernel': P(None, 'oranges')},
            1: {'bias': P(), 'kernel': P('oranges')},
        }
    }
    self.assertDictEqual(actual_shardings, expected_shardings)


if __name__ == '__main__':
  # Fake 8 CPUs for testing.
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
  absltest.main()
