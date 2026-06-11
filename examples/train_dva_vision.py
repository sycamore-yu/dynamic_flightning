"""Headless feature-style D.VA smoke script using a non-default observation adapter.

This exercises adapter-shaped actor and critic observations. It is not a
complete visual sensor training run.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flightning.envs import HoveringStateEnv
from flightning.envs.wrappers import NormalizeActionWrapper
from flightning.modules.mlp import SHACActor, SHACCritic
from flightning.algos.dva import train as train_dva, DVAConfig, DVAObservation


def main():
    print("=== 1. Initialize Hovering State Environment ===")
    env = HoveringStateEnv()
    env = NormalizeActionWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Original Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {action_dim}")

    print("\n=== 2. Define Custom Observation Adapter ===")
    # Split the flat state observation to exercise D.VA's decoupled adapter path.
    actor_obs_dim = obs_dim // 2
    critic_obs_dim = obs_dim - actor_obs_dim

    print(f"Adapted Actor Observation dimension: {actor_obs_dim}")
    print(f"Adapted Critic Observation dimension: {critic_obs_dim}")

    def custom_observation_adapter(obs: jax.Array) -> DVAObservation:
        actor_obs = obs[..., :actor_obs_dim]
        critic_obs = obs[..., actor_obs_dim:]
        return DVAObservation(actor_obs=actor_obs, critic_obs=critic_obs)

    print("\n=== 3. Set Up Actor & Critic Networks ===")
    actor_model = SHACActor([actor_obs_dim, 64, 64, action_dim], initial_scale=0.1)
    critic_model = SHACCritic([critic_obs_dim, 64, 64, 1], initial_scale=0.1)

    key = jax.random.PRNGKey(42)
    key_init, key_train = jax.random.split(key, 2)
    key_actor, key_critic = jax.random.split(key_init)

    actor_params = actor_model.initialize(key_actor)
    critic_params = critic_model.initialize(key_critic)

    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=1e-3)
    )
    critic_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=1e-3)
    )

    print("\n=== 4. Train Policy using D.VA with Custom Adapter ===")
    num_epochs = 5
    num_steps_per_epoch = 50
    num_envs = 8

    print(f"Training for {num_epochs} epochs with {num_envs} vectorized environments...")
    config = DVAConfig(
        logging=True,
        logging_freq=1,
        critic_iterations=4,
        num_batches=2,
        critic_method="td-lambda",
        max_grad_norm=1.0,
    )

    result = train_dva(
        env=env,
        actor_state=actor_state,
        critic_state=critic_state,
        observation_adapter=custom_observation_adapter,
        num_epochs=num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        num_envs=num_envs,
        key=key_train,
        config=config
    )

    final_actor_loss = result["metrics"]["actor_loss"][-1]
    final_value_loss = result["metrics"]["value_loss"][-1]
    print(f"Training Complete! Final D.VA actor loss: {final_actor_loss:.4f}, value loss: {final_value_loss:.4f}")


if __name__ == "__main__":
    main()
