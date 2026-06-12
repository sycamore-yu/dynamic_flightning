import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

from flightning.envs import HoveringStateEnv
from flightning.envs.wrappers import NormalizeActionWrapper
from flightning.modules.mlp import SHACActor, SHACCritic
from flightning.algos.dva import (
    train as train_dva,
    DVAConfig,
    DVAObservation,
    _dva_actor_loss_terms,
)


def test_dva_state_only_smoke():
    """3.1 Add a state-only D.VA smoke test that runs a tiny rollout and asserts finite metrics."""
    env = HoveringStateEnv()
    env = NormalizeActionWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_net = SHACActor([obs_dim, 32, action_dim], initial_scale=0.1)
    critic_net = SHACCritic([obs_dim, 32, 1], initial_scale=0.1)

    key = jax.random.PRNGKey(0)
    key_a, key_c, key_t = jax.random.split(key, 3)

    actor_params = actor_net.initialize(key_a)
    critic_params = critic_net.initialize(key_c)

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_params,
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_params,
        tx=optax.adam(1e-3),
    )

    config = DVAConfig(
        logging=False,
        critic_iterations=2,
        num_batches=2,
        critic_method="td-lambda",
    )

    result = train_dva(
        env,
        actor_state,
        critic_state,
        num_epochs=2,
        num_steps_per_epoch=5,
        num_envs=4,
        key=key_t,
        config=config,
    )

    assert "runner_state" in result
    assert "metrics" in result
    assert jnp.isfinite(result["metrics"]["actor_loss"][-1])
    assert jnp.isfinite(result["metrics"]["value_loss"][-1])


def test_dva_adapter():
    """3.2 Add an adapter test that verifies custom actor/critic observation shapes and JAX transformation compatibility."""
    env = HoveringStateEnv()
    env = NormalizeActionWrapper(env)

    obs_dim = env.observation_space.shape[0]  # 23 dimensions

    # Custom adapter: split observation
    def custom_adapter(obs: jax.Array) -> DVAObservation:
        actor_obs = obs[..., :10]
        critic_obs = obs[..., 10:]
        return DVAObservation(actor_obs=actor_obs, critic_obs=critic_obs)

    # Verify adapter output under JAX JIT and VMAP
    dummy_obs = jnp.zeros((obs_dim,))
    assert dummy_obs.shape == (23,)
    adapted = custom_adapter(dummy_obs)
    assert adapted.actor_obs.shape == (10,)
    assert adapted.critic_obs.shape == (13,)

    jit_adapter = jax.jit(custom_adapter)
    jit_adapted = jit_adapter(dummy_obs)
    assert jit_adapted.actor_obs.shape == (10,)
    assert jit_adapted.critic_obs.shape == (13,)

    vmap_adapter = jax.vmap(custom_adapter)
    batch_obs = jnp.zeros((4, obs_dim))
    vmap_adapted = vmap_adapter(batch_obs)
    assert vmap_adapted.actor_obs.shape == (4, 10)
    assert vmap_adapted.critic_obs.shape == (4, 13)

    # Smoke run with the custom adapter on HoveringStateEnv
    actor_net = SHACActor([10, 32, env.action_space.shape[0]], initial_scale=0.1)
    critic_net = SHACCritic([13, 32, 1], initial_scale=0.1)

    key = jax.random.PRNGKey(0)
    key_a, key_c, key_t = jax.random.split(key, 3)

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.initialize(key_a),
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.initialize(key_c),
        tx=optax.adam(1e-3),
    )

    config = DVAConfig(
        logging=False,
        critic_iterations=1,
        num_batches=1,
    )

    result = train_dva(
        env,
        actor_state,
        critic_state,
        observation_adapter=custom_adapter,
        num_epochs=1,
        num_steps_per_epoch=2,
        num_envs=2,
        key=key_t,
        config=config,
    )
    assert "runner_state" in result
    assert "metrics" in result
    assert jnp.isfinite(result["metrics"]["actor_loss"][-1])


def test_dva_gradient_semantics():
    """3.3 Add a gradient semantics test showing actor observation gradients are stopped
    while actor-parameter gradients through action-dependent reward remain finite and nonzero.
    """
    w = jnp.array(1.5)
    obs = jnp.array(2.0)

    def loss_dva(w_val, obs_val):
        obs_stop = jax.lax.stop_gradient(obs_val)
        action = w_val * obs_stop
        next_obs = obs_val + action
        reward = -(next_obs ** 2 + action ** 2)
        return -reward

    # Compute gradients w.r.t w (policy parameters) and obs (observation)
    grad_w, grad_obs = jax.grad(loss_dva, argnums=(0, 1))(w, obs)

    # 1. Parameter gradient w.r.t w should be finite and nonzero.
    assert jnp.isfinite(grad_w)
    assert grad_w != 0.0

    # 2. Observation gradient w.r.t obs through the actor input should be stopped,
    # but the gradient through the dynamics (next_obs = obs + action) is still present.
    assert jnp.allclose(grad_obs, 10.0)

    # Compare with BPTT version (no stop_gradient)
    def loss_bptt(w_val, obs_val):
        action = w_val * obs_val
        next_obs = obs_val + action
        reward = -(next_obs ** 2 + action ** 2)
        return -reward
    _, grad_obs_bptt = jax.grad(loss_bptt, argnums=(0, 1))(w, obs)
    assert jnp.allclose(grad_obs_bptt, 34.0)
    assert not jnp.allclose(grad_obs, grad_obs_bptt)


def test_dva_critic_update_smoke():
    """3.4 Add a critic update smoke test that verifies actor loss, critic loss, target values,
    and gradient norms are finite.
    """
    env = HoveringStateEnv()
    env = NormalizeActionWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_net = SHACActor([obs_dim, 16, action_dim], initial_scale=0.1)
    critic_net = SHACCritic([obs_dim, 16, 1], initial_scale=0.1)

    key = jax.random.PRNGKey(123)
    key_a, key_c, key_t = jax.random.split(key, 3)

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.initialize(key_a),
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.initialize(key_c),
        tx=optax.adam(1e-3),
    )

    # Test "one-step" critic target method
    config_one_step = DVAConfig(
        logging=False,
        critic_iterations=2,
        num_batches=2,
        critic_method="one-step",
        max_grad_norm=0.5,
    )

    result = train_dva(
        env,
        actor_state,
        critic_state,
        num_epochs=1,
        num_steps_per_epoch=4,
        num_envs=4,
        key=key_t,
        config=config_one_step,
    )

    assert jnp.isfinite(result["metrics"]["actor_loss"][-1])
    assert jnp.isfinite(result["metrics"]["value_loss"][-1])


def test_dynamic_avoidance_dva_finite_metrics():
    """Verify that training on DynamicAvoidanceEnv with dynamic_avoidance_dva_adapter
    and stop_lidar_grad=True produces finite actor and critic metrics.
    """
    from flightning.envs.dynamic_avoidance_env import (
        DynamicAvoidanceEnv,
        dynamic_avoidance_dva_adapter,
    )
    from flightning.modules.cnn_lidar_policy import CNNLidarActor

    # We use stop_lidar_grad=True to prevent unstable/NaN visual gradients
    env = DynamicAvoidanceEnv(stop_lidar_grad=True)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_model = CNNLidarActor(feature_list=[442, 32, action_dim])
    critic_model = SHACCritic(feature_list=[127, 32, 1])

    key = jax.random.PRNGKey(42)
    key_init, key_train = jax.random.split(key, 2)
    key_actor, key_critic = jax.random.split(key_init)

    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.initialize(key_actor),
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.initialize(key_critic),
        tx=optax.adam(1e-3),
    )

    config = DVAConfig(
        logging=False,
        critic_iterations=1,
        num_batches=1,
        critic_method="td-lambda",
        max_grad_norm=1.0,
    )

    result = train_dva(
        env=env,
        actor_state=actor_state,
        critic_state=critic_state,
        observation_adapter=dynamic_avoidance_dva_adapter,
        num_epochs=1,
        num_steps_per_epoch=2,
        num_envs=2,
        key=key_train,
        config=config,
    )

    assert "runner_state" in result
    assert "metrics" in result
    assert jnp.isfinite(result["metrics"]["actor_loss"][-1])
    assert jnp.isfinite(result["metrics"]["value_loss"][-1])


def test_dynamic_avoidance_dva_requires_privileged_adapter():
    from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
    from flightning.modules.cnn_lidar_policy import CNNLidarActor

    env = DynamicAvoidanceEnv(stop_lidar_grad=True)
    action_dim = env.action_space.shape[0]

    actor_model = CNNLidarActor(feature_list=[442, 16, action_dim])
    critic_model = SHACCritic(feature_list=[226, 16, 1])

    key = jax.random.PRNGKey(7)
    key_actor, key_critic, key_train = jax.random.split(key, 3)
    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.initialize(key_actor),
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.initialize(key_critic),
        tx=optax.adam(1e-3),
    )

    with pytest.raises(ValueError, match="privileged critic observation adapter"):
        train_dva(
            env=env,
            actor_state=actor_state,
            critic_state=critic_state,
            num_epochs=1,
            num_steps_per_epoch=1,
            num_envs=1,
            key=key_train,
            config=DVAConfig(logging=False, critic_iterations=1, num_batches=1),
        )


def test_dva_bootstrap_semantics():
    """Verify D.VA's done/truncated bootstrap contract:
    - Early termination (terminated=1) zeros out bootstrap value.
    - Time-limit truncation (truncated=1, terminated=0) preserves bootstrap value.
    """
    gamma = 0.99

    # Case 1: Early termination at step 0 (terminated=1, done=1)
    rewards = jnp.array([[-1.0]])
    dones = jnp.array([[1.0]])
    terminated = jnp.array([[1.0]])
    next_values = jnp.array([[10.0]])  # V(s_1) = 10.0
    loss_terms = _dva_actor_loss_terms(rewards, dones, terminated, next_values, gamma)

    # Loss term should be -(acc_tp1 + bootstrap) = -(-1.0 + 0.0) = 1.0
    assert jnp.allclose(loss_terms[0], 1.0)

    # Case 2: Time-limit truncation at step 0 (terminated=0, done=1, truncated=1)
    terminated_trunc = jnp.array([[0.0]])
    loss_terms_trunc = _dva_actor_loss_terms(
        rewards,
        dones,
        terminated_trunc,
        next_values,
        gamma,
    )

    # Loss term should be -(acc_tp1 + bootstrap) = -(-1.0 + gamma * 1.0 * 10.0) = -(8.9) = -8.9
    expected = -(-1.0 + gamma * 10.0)
    assert jnp.allclose(loss_terms_trunc[0], expected)
