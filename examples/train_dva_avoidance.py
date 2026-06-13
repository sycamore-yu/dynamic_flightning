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


def goal_directed_action(state, obs, env):
    """Reference low-level action used only to warm-start the actor network."""
    target_dir = obs[216:219]
    vel = obs[219:222] * 5.0
    pos = state.quadrotor_state.p
    rot = state.quadrotor_state.R

    desired_vel = target_dir * 1.5
    vel_error_world = desired_vel - vel
    vel_error_body = rot.T @ vel_error_world

    omega_x = -0.8 * vel_error_body[1]
    omega_y = 0.8 * vel_error_body[0]
    omega_z = 0.0

    height_error = 2.0 - pos[2]
    thrust = env.hovering_action[0] + 2.0 * height_error
    action = jnp.array([thrust, omega_x, omega_y, omega_z])
    return jnp.clip(action, env.action_space.low, env.action_space.high)


def collect_warm_start_data(env, seeds, steps_per_seed=80):
    actor_obs_batch = []
    action_batch = []

    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        state, obs = env.reset(key)

        for _ in range(steps_per_seed):
            adapted = dynamic_avoidance_dva_adapter(obs, state)
            action = goal_directed_action(state, obs, env)
            actor_obs_batch.append(adapted.actor_obs)
            action_batch.append(action)

            key, key_step = jax.random.split(key)
            transition = env._step(state, action, key_step)
            state = transition.state
            obs = transition.obs
            if transition.terminated or transition.truncated:
                break

    return jnp.stack(actor_obs_batch), jnp.stack(action_batch)


def warm_start_actor(actor_state, env):
    actor_obs, target_actions = collect_warm_start_data(
        env,
        seeds=range(1000, 1020),
        steps_per_seed=320,
    )

    warm_start_tx = optax.adam(3e-4)
    params = actor_state.params
    opt_state = warm_start_tx.init(params)

    @jax.jit
    def train_step(params, opt_state, obs_batch, action_batch):
        def loss_fn(params):
            pred = actor_state.apply_fn(params, obs_batch)
            return jnp.mean((pred - action_batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = warm_start_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss = jnp.array(jnp.inf)
    for _ in range(800):
        params, opt_state, loss = train_step(params, opt_state, actor_obs, target_actions)

    print(
        "Warm start actor imitation: "
        f"samples={actor_obs.shape[0]}, final_mse={float(loss):.6f}"
    )
    return actor_state.replace(params=params)


def termination_reason(state, env):
    pos = state.quadrotor_state.p
    vel = state.quadrotor_state.v
    dobs_state = state.dobs_state
    dists_to_dobs_xy = jnp.sqrt(jnp.sum((dobs_state.pos_xy - pos[:2]) ** 2, axis=1) + 1e-8)
    dists_to_dobs = dists_to_dobs_xy - dobs_state.radius

    if bool(jnp.any(dists_to_dobs <= 0.2)):
        return "collision_dynamic_obstacle"
    if bool(jnp.any(jnp.abs(pos[:2]) > env.termination_xy_limit)):
        return "out_of_bounds_xy"
    if bool((pos[2] < 0.5) | (pos[2] > 3.5)):
        return "out_of_height"
    if bool(jnp.sqrt(jnp.sum(vel ** 2) + 1e-8) > 10.0):
        return "excess_velocity"
    return "terminated"


def evaluate_policy(env, trained_state, seeds, total_steps=300, success_radius=3.0, viz=None):
    """Evaluate the deterministic actor over fixed seeds."""
    episode_metrics = []

    for seed_idx, seed in enumerate(seeds):
        key_eval = jax.random.PRNGKey(seed)
        state, obs = env.reset(key_eval)

        total_reward = 0.0
        reached_target = False
        terminated = False
        truncated = False
        reason = "running"
        min_target_dist = float(jnp.linalg.norm(state.target_pos - state.quadrotor_state.p))
        final_target_dist = min_target_dist
        episode_len = 0

        for step in range(total_steps):
            scan = obs[:216].reshape(1, 36, 6)
            if viz is not None and seed_idx == 0:
                viz.log_state(state, scan, step_idx=step)

            target_dist = jnp.linalg.norm(state.target_pos - state.quadrotor_state.p)
            final_target_dist = float(target_dist)
            min_target_dist = min(min_target_dist, final_target_dist)
            if target_dist <= success_radius:
                reached_target = True
                episode_len = step
                print(f"Target reached at step {step}. Distance to target: {float(target_dist):.3f} m")
                break

            key_eval, key_step = jax.random.split(key_eval)
            actor_obs = dynamic_avoidance_dva_adapter(obs, state).actor_obs
            action = trained_state.apply_fn(trained_state.params, actor_obs)
            transition = env._step(state, action, key_step)
            state = transition.state
            obs = transition.obs
            total_reward += float(transition.reward)
            episode_len = step + 1

            if transition.terminated or transition.truncated:
                terminated = bool(transition.terminated)
                truncated = bool(transition.truncated)
                reason = termination_reason(state, env) if terminated else "timeout"
                if seed_idx == 0:
                    print(f"Episode ended at step {step}. Reason: {reason}")
                break

        episode_metrics.append({
            "seed": seed,
            "reached_target": reached_target,
            "terminated": terminated,
            "truncated": truncated,
            "reason": reason,
            "episode_len": episode_len,
            "min_target_dist": min_target_dist,
            "final_target_dist": final_target_dist,
            "total_reward": total_reward,
        })

    return episode_metrics


def summarize_policy_metrics(metrics):
    count = len(metrics)
    success_rate = sum(m["reached_target"] for m in metrics) / count
    termination_rate = sum(m["terminated"] for m in metrics) / count
    mean_episode_len = sum(m["episode_len"] for m in metrics) / count
    mean_final_dist = sum(m["final_target_dist"] for m in metrics) / count
    mean_min_dist = sum(m["min_target_dist"] for m in metrics) / count
    mean_return = sum(m["total_reward"] for m in metrics) / count
    return {
        "success_rate": success_rate,
        "termination_rate": termination_rate,
        "mean_episode_len": mean_episode_len,
        "mean_final_dist": mean_final_dist,
        "mean_min_dist": mean_min_dist,
        "mean_return": mean_return,
    }


def main():
    print("=== 1. Initialize Dynamic Avoidance Environment ===")
    env_config = DynamicAvoidanceConfig(
        trace_prob=1.0,
        stop_lidar_grad=True,
        clearance_weight=2.0,
        motion_risk_weight=0.2,
        barrier_temperature=0.5,
        dobs_vel_range=(0.3, 1.0),
        dobs_radius_range=(0.15, 0.25),
        reset_obstacle_clearance=5.0,
        reset_target_offset=28.0,
    )
    env = DynamicAvoidanceEnv(config=env_config)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Actor Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {action_dim}")

    print("\n=== 2. Set Up Actor & Critic Networks ===")
    # Actor network: CNNLidarActor fusing 216 LiDAR dims + 10 state dims
    feature_list_actor = [442, 64, 64, action_dim]
    actor_action_scale = jnp.array([2.5, 3.0, 3.0, 2.0])
    actor_model = CNNLidarActor(
        feature_list=feature_list_actor,
        action_bias=env.hovering_action,
        action_scale=actor_action_scale,
        initial_scale=0.1,
        initial_log_std=-2.0,
        min_std=0.05,
    )
    print(f"Actor action bias: {env.hovering_action}")
    print(f"Actor action scale: {actor_action_scale}")

    # Critic network: SHACCritic processing 127-dimensional privileged state
    feature_list_critic = [127, 64, 64, 1]
    critic_model = SHACCritic(feature_list=feature_list_critic)

    key_actor = jax.random.PRNGKey(42)
    key_critic = jax.random.PRNGKey(43)
    key_train = jax.random.PRNGKey(44)

    actor_params = actor_model.initialize(key_actor)
    critic_params = critic_model.initialize(key_critic)

    # 3. Explicitly declare actor/critic optimizer policy with fixed learning rates
    actor_lr = 0.0
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

    print("\n=== 3. Warm Start Actor from Goal-Directed Reference Actions ===")
    actor_state = warm_start_actor(actor_state, env)

    print("\n=== 4. Train Policy using D.VA with Privileged Critic Schema ===")
    num_epochs = 80
    num_steps_per_epoch = 150
    num_envs = 8

    print(f"Training for {num_epochs} epochs with {num_envs} vectorized environments...")
    config = DVAConfig(
        logging=True,
        logging_freq=10,
        critic_iterations=4,
        num_batches=2,
        critic_method="td-lambda",
        gamma=0.97,
        lam=0.92,
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
    assert value_tail < 0.50 * value_losses[0], "Critic value loss did not converge enough for validation."
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

    eval_metrics = evaluate_policy(
        env,
        trained_state,
        seeds=(123, 124, 125, 126, 127),
        total_steps=400,
        success_radius=3.0,
        viz=viz,
    )
    eval_summary = summarize_policy_metrics(eval_metrics)
    first_metric = eval_metrics[0]
    print(
        "D.VA policy evaluation seed=123: "
        f"reached_target={first_metric['reached_target']}, "
        f"min_target_dist={first_metric['min_target_dist']:.3f} m, "
        f"final_target_dist={first_metric['final_target_dist']:.3f} m, "
        f"episode_len={first_metric['episode_len']}, "
        f"total_reward={first_metric['total_reward']:.4f}"
    )
    print(
        "D.VA policy evaluation summary: "
        f"success_rate={eval_summary['success_rate']:.2f}, "
        f"termination_rate={eval_summary['termination_rate']:.2f}, "
        f"mean_episode_len={eval_summary['mean_episode_len']:.1f}, "
        f"mean_min_target_dist={eval_summary['mean_min_dist']:.3f} m, "
        f"mean_final_target_dist={eval_summary['mean_final_dist']:.3f} m, "
        f"mean_return={eval_summary['mean_return']:.4f}"
    )
    assert eval_summary["success_rate"] >= 0.6, "D.VA policy did not reach the target on enough evaluation seeds."
    assert eval_summary["termination_rate"] <= 0.2, "D.VA policy terminated too often during evaluation."
    if HAS_RERUN:
        print(f"RRD file successfully exported to {os.path.abspath(rrd_path)}.")
    else:
        print("RRD file export was skipped because rerun-sdk is not installed.")


if __name__ == "__main__":
    main()
