import jax
import jax.numpy as jnp
from flightning.modules.dynamic_obstacle_field import DynamicObstacleField, DynamicObstacleFieldState

def test_deterministic_prng():
    key = jax.random.PRNGKey(42)
    state1 = DynamicObstacleField.reset(key)
    state2 = DynamicObstacleField.reset(key)

    assert jnp.allclose(state1.pos_xy, state2.pos_xy)
    assert jnp.allclose(state1.vel_xy, state2.vel_xy)
    assert jnp.allclose(state1.radius, state2.radius)

    drone_pos = jnp.array([0.0, 0.0, 1.0])
    key_step = jax.random.PRNGKey(123)

    next_state1 = DynamicObstacleField.update(state1, drone_pos, key_step, dt=0.02, trace_prob=0.3)
    next_state2 = DynamicObstacleField.update(state2, drone_pos, key_step, dt=0.02, trace_prob=0.3)

    assert jnp.allclose(next_state1.pos_xy, next_state2.pos_xy)
    assert jnp.allclose(next_state1.vel_xy, next_state2.vel_xy)

def test_cylinder_pose():
    pos_xy = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    pos, rot = DynamicObstacleField.get_cylinder_poses(pos_xy, dobs_height=4.0)

    expected_pos = jnp.array([[1.0, 2.0, 2.0], [3.0, 4.0, 2.0]])
    assert jnp.allclose(pos, expected_pos)
    assert rot.shape == (2, 3, 3)
    assert jnp.allclose(rot[0], jnp.eye(3))

def test_boundary_reflection():
    state = DynamicObstacleFieldState(
        pos_xy=jnp.array([[19.0, 0.0]]),
        vel_xy=jnp.array([[2.0, 0.0]]),
        radius=jnp.array([0.3]),
        hit=jnp.array([0.0])
    )
    drone_pos = jnp.array([20.0, 20.0, 1.0])  # Invalid drone (outside center)
    key = jax.random.PRNGKey(0)

    next_state = DynamicObstacleField.update(state, drone_pos, key, dt=0.02, trace_prob=0.5)

    # Reflects since no valid drone is present
    assert jnp.allclose(next_state.vel_xy[0], jnp.array([-2.0, 0.0]))
    assert jnp.allclose(next_state.pos_xy[0], jnp.array([19.0 + (-2.0 * 0.02), 0.0]))

def test_trace_prob_semantics():
    state = DynamicObstacleFieldState(
        pos_xy=jnp.array([[19.0, 0.0]]),
        vel_xy=jnp.array([[2.0, 0.0]]),
        radius=jnp.array([0.3]),
        hit=jnp.array([0.0])
    )
    drone_pos = jnp.array([19.0, 10.0, 1.0])  # Valid drone pos (still in center since x in [-10, 10]... wait, x=19 is NOT in [-10, 10]!)
    # Let's put drone at x=5, y=5. It is valid!
    # Let's put obstacle at (19.0, 5.0) moving with [2.0, 0.0].
    # Direction to drone at (5.0, 5.0) is [-14.0, 0.0], so tracing velocity is [-2.0, 0.0].
    # Wait, to make trace different from reflect, let's put drone at (19.0, 9.0) (which is valid since 9.0 in [-10, 10] and we put it at x=0.0, y=9.0).
    # Obstacle at (19.0, 0.0). Drone at (0.0, 9.0).
    # Direction is [-19.0, 9.0]. Normalizing: norm is sqrt(361 + 81) = 21.02.
    # Trace velocity will have positive y velocity! Reflect velocity will have 0 y velocity.
    drone_pos = jnp.array([0.0, 9.0, 1.0])  # x=0, y=9 (valid drone in center)
    key = jax.random.PRNGKey(0)

    # If trace_prob = 1.0, random >= 1.0 is False, so it ALWAYS reflects: vel becomes [-2.0, 0.0]
    next_state_reflect = DynamicObstacleField.update(state, drone_pos, key, dt=0.02, trace_prob=1.0)
    assert jnp.allclose(next_state_reflect.vel_xy[0], jnp.array([-2.0, 0.0]))

    # If trace_prob = 0.0, random >= 0.0 is True, so it ALWAYS traces: y-velocity becomes positive
    next_state_trace = DynamicObstacleField.update(state, drone_pos, key, dt=0.02, trace_prob=0.0)
    assert next_state_trace.vel_xy[0, 1] > 0.0
