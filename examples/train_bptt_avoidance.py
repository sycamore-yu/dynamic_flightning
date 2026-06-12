"""Example script demonstrating training a dynamic obstacle avoidance policy with BPTT in Flightning.

This script demonstrates:
1. Initializing the DynamicAvoidanceEnv environment.
2. Setting up the CNNLidarActor network.
3. Training using Backpropagation Through Time (BPTT).
4. Running a single evaluation rollout and exporting it to a Rerun `.rrd` file for offline viewing.

Design Documentation:
- `analytic_lidar_grad`: The default gradient route (stop_lidar_grad=False) where JAX backpropagates gradients of
  the LiDAR scan directly through MjLidarJax raytracing and sensor geometry back to the drone and obstacle states.
- `stop_lidar_grad`: A stability/ablation baseline (stop_lidar_grad=True) which stops gradients at the raytracer input.
- D.VA (Differentiable Visual Avoidance) algorithm: Out of scope for this change; belongs to the `add-dva-algorithm` change.
"""

import os
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
from flightning.modules.cnn_lidar_policy import CNNLidarActor
from flightning.algos.bptt import train as train_bptt, Config as BPTTConfig
from flightning.visualization.rerun_dynamic_avoidance import RerunVizAdapter, HAS_RERUN

def main():
    print("=== 1. Initialize Dynamic Avoidance Environment ===")
    # By default, stop_lidar_grad=False, which uses the default analytic_lidar_grad route.
    env = DynamicAvoidanceEnv(stop_lidar_grad=False)
    
    # 226 dimensions: lidar_flat(216) + target_dir(3) + velocity(3) + last_action(4)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {action_dim}")

    print("\n=== 2. Set Up CNN Lidar Actor Network ===")
    # CNN outputs 432 features. Fusion layer shape is 432 + 10 = 442.
    feature_list = [442, 64, 64, action_dim]
    actor_model = CNNLidarActor(feature_list=feature_list)

    key = jax.random.PRNGKey(42)
    key_init, key_train, key_eval = jax.random.split(key, 3)

    actor_params = actor_model.initialize(key_init)
    
    tx = optax.adam(learning_rate=1e-3)
    train_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=tx
    )

    print("\n=== 3. Train Policy using BPTT ===")
    # Run a short training loop for demonstration purposes (5 epochs, 50 steps, 8 envs)
    num_epochs = 5
    num_steps_per_epoch = 50
    num_envs = 8
    
    print(f"Training for {num_epochs} epochs with {num_envs} vectorized environments...")
    config = BPTTConfig(logging=True, logging_freq=1)
    
    result = train_bptt(
        env=env,
        train_state=train_state,
        num_epochs=num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        num_envs=num_envs,
        key=key_train,
        config=config
    )
    
    trained_state = result["runner_state"].train_state
    final_loss = result["metrics"][-1]
    print(f"Training Complete! Final BPTT loss: {final_loss:.4f}")

    print("\n=== 4. Run Evaluation and Export Rerun .rrd file ===")
    rrd_path = "dynamic_avoidance_rollout.rrd"
    print(f"Initializing RerunVizAdapter to export to {rrd_path}...")
    viz = RerunVizAdapter(save_path=rrd_path)

    # Reset a single environment
    state, obs = env.reset(key_eval)
    
    # Simple loop to run evaluation rollout
    total_steps = 150
    total_reward = 0.0
    
    for step in range(total_steps):
        # Predict deterministic action
        action = trained_state.apply_fn(trained_state.params, obs)
        
        # scan is decoded from the first 216 dimensions of obs
        scan = obs[:216].reshape(1, 36, 6)
        viz.log_state(state, scan, step_idx=step)
        
        # Step environment
        transition = env._step(state, action, key_eval)
        state = transition.state
        obs = transition.obs
        total_reward += float(transition.reward)
        
        if transition.terminated or transition.truncated:
            print(f"Episode ended at step {step}. Reason: {'Collision/Out of Bounds' if transition.terminated else 'Timeout'}")
            break
            
    print(f"Evaluation Complete! Total reward: {total_reward:.4f}")
    if HAS_RERUN:
        print(f"RRD file successfully exported to {os.path.abspath(rrd_path)}.")
    else:
        print("RRD file export was skipped because rerun-sdk is not installed.")

if __name__ == "__main__":
    main()
