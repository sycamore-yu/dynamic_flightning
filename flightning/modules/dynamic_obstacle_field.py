import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import chex
from flightning.utils.pytrees import CustomPyTree

@jdc.pytree_dataclass
class DynamicObstacleFieldState(CustomPyTree):
    pos_xy: jax.Array  # (N, 2)
    vel_xy: jax.Array  # (N, 2)
    radius: jax.Array  # (N,)
    hit: jax.Array     # (N,)

class DynamicObstacleField:
    """Manages P2M-compatible dynamic obstacles."""

    @staticmethod
    def reset(
        key: chex.PRNGKey,
        num_obstacles: int = 40,
        pos_x_range: tuple[float, float] = (-18.0, 18.0),
        pos_y_range: tuple[float, float] = (-18.0, 18.0),
        vel_range: tuple[float, float] = (1.0, 5.0),
        rad_range: tuple[float, float] = (0.25, 0.45)
    ) -> DynamicObstacleFieldState:
        """Initialize dynamic obstacle states."""
        key_pos_x, key_pos_y, key_vel_norm, key_vel_angle, key_rad = jax.random.split(key, 5)

        pos_x = jax.random.uniform(key_pos_x, shape=(num_obstacles,), minval=pos_x_range[0], maxval=pos_x_range[1])
        pos_y = jax.random.uniform(key_pos_y, shape=(num_obstacles,), minval=pos_y_range[0], maxval=pos_y_range[1])
        pos_xy = jnp.stack([pos_x, pos_y], axis=-1)

        vel_norm = jax.random.uniform(key_vel_norm, shape=(num_obstacles,), minval=vel_range[0], maxval=vel_range[1])
        vel_angle = jax.random.uniform(key_vel_angle, shape=(num_obstacles,), minval=0.0, maxval=2.0 * jnp.pi)
        vel_x = vel_norm * jnp.cos(vel_angle)
        vel_y = vel_norm * jnp.sin(vel_angle)
        vel_xy = jnp.stack([vel_x, vel_y], axis=-1)

        radius = jax.random.uniform(key_rad, shape=(num_obstacles,), minval=rad_range[0], maxval=rad_range[1])
        hit = jnp.zeros((num_obstacles,))

        return DynamicObstacleFieldState(
            pos_xy=pos_xy,
            vel_xy=vel_xy,
            radius=radius,
            hit=hit
        )

    @staticmethod
    def update(
        state: DynamicObstacleFieldState,
        drone_pos: jax.Array,
        key: chex.PRNGKey,
        dt: float,
        trace_prob: float,
        pos_x_range: tuple[float, float] = (-18.0, 18.0),
        pos_y_range: tuple[float, float] = (-18.0, 18.0)
    ) -> DynamicObstacleFieldState:
        """Update obstacle velocities and positions."""
        pos_x = state.pos_xy[:, 0]
        pos_y = state.pos_xy[:, 1]

        # 1. Boundary check
        touch_x_lower = pos_x < pos_x_range[0]
        touch_x_upper = pos_x > pos_x_range[1]
        touch_y_lower = pos_y < pos_y_range[0]
        touch_y_upper = pos_y > pos_y_range[1]

        touch_bound = touch_x_lower | touch_x_upper | touch_y_lower | touch_y_upper

        # 2. Check if drone is in center
        drone_pos_xy = drone_pos[:2]
        drone_in_center = (drone_pos_xy[0] > -10.0) & (drone_pos_xy[0] < 10.0) & \
                          (drone_pos_xy[1] > -10.0) & (drone_pos_xy[1] < 10.0)

        # 3. Tracing direction & velocity
        direction_to_drone = drone_pos_xy[None, :] - state.pos_xy
        direction_norm = jnp.linalg.norm(direction_to_drone, axis=-1, keepdims=True)
        direction_unit = direction_to_drone / (direction_norm + 1e-8)

        vel_norm = jnp.linalg.norm(state.vel_xy, axis=-1, keepdims=True)
        trace_vel = direction_unit * vel_norm

        # 4. PRNG for trace choices
        key_prob, _ = jax.random.split(key)
        random_probs = jax.random.uniform(key_prob, shape=(state.pos_xy.shape[0],))
        is_trace = random_probs >= trace_prob

        # Trace toward drone if chosen and drone is in center, otherwise reflect original velocity
        new_vel_candidate = jnp.where(
            (is_trace[:, None]) & drone_in_center,
            trace_vel,
            -state.vel_xy
        )

        # 5. Overrule candidate if it moves further out of bounds (reflection constraint)
        vel_x = new_vel_candidate[:, 0]
        vel_y = new_vel_candidate[:, 1]
        reflect_x = (touch_x_lower & (vel_x <= 0.0)) | (touch_x_upper & (vel_x >= 0.0))
        reflect_y = (touch_y_lower & (vel_y <= 0.0)) | (touch_y_upper & (vel_y >= 0.0))
        needs_reflect = reflect_x | reflect_y

        final_new_vel = jnp.where(needs_reflect[:, None], -state.vel_xy, new_vel_candidate)

        # Apply new velocity only to touching obstacles
        vel_xy = jnp.where(touch_bound[:, None], final_new_vel, state.vel_xy)

        # 6. Step position
        pos_xy = state.pos_xy + vel_xy * dt

        return state.replace(pos_xy=pos_xy, vel_xy=vel_xy)

    @staticmethod
    def get_cylinder_poses(pos_xy: jax.Array, dobs_height: float) -> tuple[jax.Array, jax.Array]:
        """Convert 2D positions to 3D cylinder positions and rotation matrices."""
        num_obstacles = pos_xy.shape[0]
        z = jnp.full((num_obstacles, 1), dobs_height / 2.0)
        geom_xpos = jnp.concatenate([pos_xy, z], axis=-1)
        geom_xmat = jnp.tile(jnp.eye(3)[None, :, :], (num_obstacles, 1, 1))
        return geom_xpos, geom_xmat

