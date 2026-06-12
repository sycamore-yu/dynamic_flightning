"""Headless D.VA validation script for dynamic avoidance training.

This script runs in the flightning conda environment and trains a dynamic obstacle
avoidance policy using the generic D.VA algorithm with a privileged critic schema.
"""

import os

# Run on GPU if available
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flightning.envs.dynamic_avoidance_env import (
    DynamicAvoidanceConfig,
    DynamicAvoidanceEnv,
    dynamic_avoidance_dva_adapter,
)
from flightning.modules.cnn_lidar_policy import CNNLidarActor
from flightning.modules.mlp import SHACCritic
from flightning.algos.dva import train as train_dva, DVAConfig


def main():
    print("=== 1. Initialize Dynamic Avoidance Environment ===")
    env_config = DynamicAvoidanceConfig(
        stop_lidar_grad=True,
        clearance_weight=0.5,
        motion_risk_weight=0.05,
        barrier_temperature=0.75,
    )
    env = DynamicAvoidanceEnv(config=env_config)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Actor Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {action_dim}")

    print("\n=== 2. Set Up Actor & Critic Networks ===")
    # Actor network: CNNLidarActor fusing 216 LiDAR dims + 10 state dims
    feature_list_actor = [442, 64, 64, action_dim]
    actor_model = CNNLidarActor(feature_list=feature_list_actor)

    # Critic network: SHACCritic processing 127-dimensional privileged state
    feature_list_critic = [127, 64, 64, 1]
    critic_model = SHACCritic(feature_list=feature_list_critic)

    key = jax.random.PRNGKey(42)
    key_init, key_train = jax.random.split(key, 2)
    key_actor, key_critic = jax.random.split(key_init)

    actor_params = actor_model.initialize(key_actor)
    critic_params = critic_model.initialize(key_critic)

    # 3. Explicitly declare actor/critic optimizer policy with fixed learning rates
    actor_lr = 3e-5
    critic_lr = 1e-3
    print(f"Explicit Optimizer Policy: Adam optimizer for both actor (lr={actor_lr}) and critic (lr={critic_lr})")

    actor_tx = optax.adam(learning_rate=actor_lr)
    critic_tx = optax.adam(learning_rate=critic_lr)

    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=actor_tx
    )
    critic_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=critic_tx
    )

    print("\n=== 4. Train Policy using D.VA with Privileged Critic Schema ===")
    num_epochs = 80
    num_steps_per_epoch = 20
    num_envs = 8

    print(f"Training for {num_epochs} epochs with {num_envs} vectorized environments...")
    config = DVAConfig(
        logging=True,
        logging_freq=10,
        critic_iterations=2,
        num_batches=2,
        critic_method="td-lambda",
        gamma=0.9,
        lam=0.9,
        max_grad_norm=0.5,
    )

    result = train_dva(
        env=env,
        actor_state=actor_state,
        critic_state=critic_state,
        observation_adapter=dynamic_avoidance_dva_adapter,
        num_epochs=num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        num_envs=num_envs,
        key=key_train,
        config=config
    )

    final_actor_loss = result["metrics"]["actor_loss"][-1]
    final_value_loss = result["metrics"]["value_loss"][-1]
    actor_losses = jnp.asarray(result["metrics"]["actor_loss"])
    value_losses = jnp.asarray(result["metrics"]["value_loss"])
    actor_plateau_ref = jnp.mean(actor_losses[40:60])
    actor_tail = jnp.mean(actor_losses[-20:])
    value_tail = jnp.mean(value_losses[-20:])
    print(f"Training Complete! Final D.VA actor loss: {final_actor_loss:.4f}, value loss: {final_value_loss:.4f}")
    print(
        "Convergence check: "
        f"actor plateau ref={actor_plateau_ref:.4f}, actor tail={actor_tail:.4f}, "
        f"value initial={value_losses[0]:.4f}, value tail={value_tail:.4f}"
    )

    # Check finite metrics
    assert jnp.isfinite(final_actor_loss), "Actor loss is not finite!"
    assert jnp.isfinite(final_value_loss), "Value loss is not finite!"
    assert value_tail < 0.25 * value_losses[0], "Critic value loss did not converge enough for validation."
    assert jnp.abs(actor_tail - actor_plateau_ref) < 0.25 * jnp.maximum(jnp.abs(actor_plateau_ref), 1.0), (
        "Actor loss did not reach a stable plateau for validation."
    )
    print("Verification passed: D.VA actor and critic metrics are finite.")

    print("\n=== 5. Run Evaluation and Export Rerun .rrd file ===")
    from flightning.visualization.rerun_dynamic_avoidance import RerunVizAdapter, HAS_RERUN
    
    rrd_path = "examples/outputs/dynamic_avoidance_dva_rerun.rrd"
    print(f"Initializing RerunVizAdapter to export to {rrd_path}...")
    viz = RerunVizAdapter(save_path=rrd_path)

    trained_state = result["runner_state"].actor_state

    key_eval = jax.random.PRNGKey(123)
    state, obs = env.reset(key_eval)

    total_steps = 150
    total_reward = 0.0
    success_radius = 3.0
    reached_target = False
    min_target_dist = float(jnp.linalg.norm(state.target_pos - state.quadrotor_state.p))

    for step in range(total_steps):
        scan = obs[:216].reshape(1, 36, 6)
        viz.log_state(state, scan, step_idx=step)

        target_dist = jnp.linalg.norm(state.target_pos - state.quadrotor_state.p)
        min_target_dist = min(min_target_dist, float(target_dist))
        if target_dist <= success_radius:
            reached_target = True
            print(f"Target reached at step {step}. Distance to target: {float(target_dist):.3f} m")
            break

        key_eval, key_step = jax.random.split(key_eval)
        action = trained_state.apply_fn(trained_state.params, obs)

        # Step environment
        transition = env._step(state, action, key_step)
        state = transition.state
        obs = transition.obs
        total_reward += float(transition.reward)

        if transition.terminated or transition.truncated:
            print(f"Episode ended at step {step}. Reason: {'Collision/Out of Bounds' if transition.terminated else 'Timeout'}")
            break

    print(
        "D.VA policy evaluation: "
        f"reached_target={reached_target}, min_target_dist={min_target_dist:.3f} m, "
        f"total_reward={total_reward:.4f}"
    )
    if HAS_RERUN:
        print(f"RRD file successfully exported to {os.path.abspath(rrd_path)}.")
    else:
        print("RRD file export was skipped because rerun-sdk is not installed.")


if __name__ == "__main__":
    main()
