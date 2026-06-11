import jax
import jax.numpy as jnp
from flightning.utils import spaces

class ObservationBuilder:
    """Observation Builder for the dynamic avoidance environment.

    Layout:
        lidar_flat(216) + target_dir(3) + velocity(3) + last_action(4) = 226 dimensions.
    """

    @staticmethod
    def get_observation(
        lidar_scan: jax.Array,  # shape (1, 36, 6) or (36, 6)
        drone_pos: jax.Array,   # shape (3,)
        target_pos: jax.Array,  # shape (3,)
        drone_vel: jax.Array,   # shape (3,)
        last_action: jax.Array, # shape (4,)
        vel_max: float = 5.0
    ) -> jax.Array:
        # 1. Flatten LiDAR scan
        lidar_flat = lidar_scan.flatten()  # shape (216,)

        # 2. Compute target direction (unit vector)
        rpos = target_pos - drone_pos
        dist = jnp.sqrt(jnp.sum(rpos ** 2) + 1e-8)
        target_dir = rpos / dist

        # 3. Normalize velocity by vel_max
        velocity_norm = drone_vel / vel_max

        # 4. Concatenate all features
        obs = jnp.concatenate([
            lidar_flat,
            target_dir,
            velocity_norm,
            last_action
        ])
        return obs

    @staticmethod
    def get_observation_shape() -> tuple[int, ...]:
        return (216 + 3 + 3 + 4,)

    @staticmethod
    def get_observation_space(action_space: spaces.Box, cutoff_dist: float = 10.0) -> spaces.Box:
        # lidar: [0, cutoff_dist]
        # target_dir: [-1.0, 1.0]
        # velocity_norm: [-10.0, 10.0] (roughly, or we can use larger bounds like [-np.inf, np.inf] or [-2.0, 2.0])
        # last_action: action_space bounds
        low = jnp.concatenate([
            jnp.zeros(216),
            -jnp.ones(3),
            -jnp.ones(3) * 5.0,  # normalized velocity lower bound (corresponding to -25 m/s)
            action_space.low
        ])
        high = jnp.concatenate([
            jnp.ones(216) * cutoff_dist,
            jnp.ones(3),
            jnp.ones(3) * 5.0,   # normalized velocity upper bound (corresponding to 25 m/s)
            action_space.high
        ])
        return spaces.Box(low=low, high=high, shape=(226,))
