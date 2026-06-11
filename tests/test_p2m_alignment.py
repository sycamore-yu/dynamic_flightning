import jax
import jax.numpy as jnp
from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
from flightning.modules.dynamic_obstacle_field import DynamicObstacleField, DynamicObstacleFieldState

def test_p2m_obstacle_alignment_and_trace_prob():
    # Verify trace prob edge cases specifically
    # Obstacle is at (19.0, 0.0) moving at [2.0, 0.0]
    state = DynamicObstacleFieldState(
        pos_xy=jnp.array([[19.0, 0.0]]),
        vel_xy=jnp.array([[2.0, 0.0]]),
        radius=jnp.array([0.3]),
        hit=jnp.array([0.0])
    )
    drone_pos = jnp.array([0.0, 9.0, 1.0])  # valid drone in center
    key = jax.random.PRNGKey(0)

    # 1. trace_prob = 1.0 => random >= 1.0 is False => reflect => vel should be [-2.0, 0.0]
    next_state_reflect = DynamicObstacleField.update(state, drone_pos, key, dt=0.02, trace_prob=1.0)
    assert jnp.allclose(next_state_reflect.vel_xy[0], jnp.array([-2.0, 0.0]))

    # 2. trace_prob = 0.0 => random >= 0.0 is True => trace => velocity directed to drone
    next_state_trace = DynamicObstacleField.update(state, drone_pos, key, dt=0.02, trace_prob=0.0)
    # Direction is [-19.0, 9.0], so velocity x should be negative and y should be positive
    assert next_state_trace.vel_xy[0, 0] < 0.0
    assert next_state_trace.vel_xy[0, 1] > 0.0

def test_fixed_geometry_lidar_scan():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    # Set a fixed drone pos and orientation
    quad_state = state.quadrotor_state.replace(
        p=jnp.array([0.0, 0.0, 2.0]),
        R=jnp.eye(3)
    )
    
    # 40 obstacles at [0,0]
    dobs_state = state.dobs_state.replace(
        pos_xy=jnp.zeros((40, 2))
    )
    
    fixed_state = state.replace(
        quadrotor_state=quad_state,
        dobs_state=dobs_state
    )

    # Compute scan
    scan = env.lidar_sensor.get_scan(
        fixed_state.quadrotor_state.p,
        fixed_state.quadrotor_state.R,
        fixed_state.dobs_state.pos_xy,
        stop_lidar_grad=False
    )

    assert scan.shape == (1, 36, 6)
    # Check that scan values are within [0, 10.0]
    assert jnp.all(scan >= 0.0)
    assert jnp.all(scan <= 10.0)

def test_p2m_reward_computation():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)
    state1, _ = env.reset(key)
    transition = env._step(state1, env.hovering_action, key)
    state2 = transition.state
    scan = env.lidar_sensor.get_scan(state2.quadrotor_state.p, state2.quadrotor_state.R, state2.dobs_state.pos_xy)

    rewards_dict = env.compute_p2m_reward(state1, state2, scan)

    expected_keys = [
        "reward_velocity",
        "reward_acceleration",
        "reward_jerk",
        "reward_height",
        "reward_goal",
        "reward_safety",
        "reward_dobs",
        "reward_total"
    ]
    for k in expected_keys:
        assert k in rewards_dict
        assert jnp.isfinite(rewards_dict[k])
