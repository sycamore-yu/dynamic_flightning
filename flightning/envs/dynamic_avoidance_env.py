from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Optional, Union, Any, Dict, TYPE_CHECKING
import chex
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from flightning.objects import Quadrotor, QuadrotorState
from flightning.modules.dynamic_obstacle_field import DynamicObstacleField, DynamicObstacleFieldState
from flightning.modules.observation_builder import ObservationBuilder
from flightning.utils import math as math_utils
from flightning.utils import spaces
from flightning.utils.pytrees import stack_pytrees
import flightning.envs.env_base as env_base
from flightning.envs.env_base import EnvTransition

if TYPE_CHECKING:
    from flightning.sensors.mujoco_lidar_sensor import MujocoLidarSensor


@dataclass(frozen=True)
class DynamicAvoidanceConfig:
    max_steps_in_episode: int = 500
    dt: float = 0.02
    delay: float = 0.02
    drone_path: Optional[str] = None
    trace_prob: float = 0.3
    stop_lidar_grad: bool = False
    cutoff_dist: float = 10.0
    dobs_height: float = 4.0
    arena_half_extent: float = 20.0
    termination_margin: float = 2.0
    reset_margin: float = 2.0
    reset_inner_extent: float = 20.0
    reset_target_offset: float = 12.0

    @property
    def termination_xy_limit(self) -> float:
        return self.arena_half_extent - self.termination_margin

    @property
    def reset_path_extent(self) -> float:
        return 2.0 * (self.termination_xy_limit - self.reset_margin)

    @property
    def num_last_actions(self) -> int:
        return int(np.ceil(self.delay / self.dt)) + 1


@dataclass(frozen=True)
class DynamicAvoidanceStatic:
    config: DynamicAvoidanceConfig
    quadrotor: Quadrotor = field(compare=False, hash=False)
    lidar_sensor: "MujocoLidarSensor" = field(compare=False, hash=False)


@lru_cache(maxsize=None)
def _get_dynamic_avoidance_static(config: DynamicAvoidanceConfig) -> DynamicAvoidanceStatic:
    from flightning.sensors.mujoco_lidar_sensor import MujocoLidarSensor

    if config.drone_path is not None:
        quadrotor = Quadrotor.from_yaml(config.drone_path)
    else:
        quadrotor = Quadrotor.default_quadrotor()

    lidar_sensor = MujocoLidarSensor(
        scan_mode="p2m_oversample",
        cutoff_dist=config.cutoff_dist,
        dobs_height=config.dobs_height,
    )
    return DynamicAvoidanceStatic(config=config, quadrotor=quadrotor, lidar_sensor=lidar_sensor)

@jdc.pytree_dataclass
class DynamicAvoidanceEnvState(env_base.EnvState):
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    target_pos: jax.Array  # shape (3,)
    start_pos: jax.Array   # shape (3,)
    dobs_state: DynamicObstacleFieldState
    last_actions: jax.Array  # shape (num_last_actions, 4)


@partial(jax.jit, static_argnames=("static",))
def _reset_jit(
    key: chex.PRNGKey, static: DynamicAvoidanceStatic
) -> tuple[DynamicAvoidanceEnvState, jax.Array]:
    cfg = static.config
    quadrotor = static.quadrotor
    key_pos, key_yaw, key_dobs, key_dr = jax.random.split(key, 4)

    # Sample start and target positions using the 4 P2M sectors, scaled to
    # the first-version Flightning arena so reset states start inside bounds.
    out_max = cfg.reset_path_extent
    in_max = cfg.reset_inner_extent
    offset = cfg.reset_target_offset
    fly_height = 2.0

    val = jax.random.uniform(key_pos, minval=-in_max/2.0, maxval=in_max/2.0)
    sector = jax.random.randint(key_pos, shape=(), minval=0, maxval=4)

    start_pos = jax.lax.switch(
        sector,
        [
            lambda v: jnp.array([v, -out_max/2.0, fly_height]),
            lambda v: jnp.array([out_max/2.0, v, fly_height]),
            lambda v: jnp.array([-out_max/2.0, v, fly_height]),
            lambda v: jnp.array([v, out_max/2.0, fly_height]),
        ],
        val,
    )

    target_pos = jax.lax.switch(
        sector,
        [
            lambda v: jnp.array([-v, out_max/2.0 - offset, fly_height]),
            lambda v: jnp.array([-out_max/2.0 + offset, -v, fly_height]),
            lambda v: jnp.array([out_max/2.0 - offset, -v, fly_height]),
            lambda v: jnp.array([-v, -out_max/2.0 + offset, fly_height]),
        ],
        val,
    )

    dir_vector = target_pos - start_pos
    yaw = jnp.arctan2(dir_vector[1], dir_vector[0])
    yaw_noise = 0.1 * jax.random.normal(key_yaw)
    yaw_angle = yaw + yaw_noise

    cos_y = jnp.cos(yaw_angle)
    sin_y = jnp.sin(yaw_angle)
    R = jnp.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ])

    quadrotor_state = quadrotor.create_state(
        p=start_pos,
        R=R,
        v=jnp.zeros(3),
        omega=jnp.zeros(3),
        dr_key=key_dr,
    )

    dobs_state = DynamicObstacleField.reset(
        key_dobs,
        pos_x_range=(-cfg.termination_xy_limit, cfg.termination_xy_limit),
        pos_y_range=(-cfg.termination_xy_limit, cfg.termination_xy_limit),
    )

    thrust_hover = 9.81 * quadrotor._mass
    hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
    last_actions = jnp.tile(hovering_action, (cfg.num_last_actions, 1))

    new_state = DynamicAvoidanceEnvState(
        time=0.0,
        step_idx=0,
        quadrotor_state=quadrotor_state,
        target_pos=target_pos,
        start_pos=start_pos,
        dobs_state=dobs_state,
        last_actions=last_actions,
    )

    scan = static.lidar_sensor.get_scan(
        quadrotor_state.p,
        quadrotor_state.R,
        dobs_state.pos_xy,
        stop_lidar_grad=cfg.stop_lidar_grad,
    )
    obs = ObservationBuilder.get_observation(
        lidar_scan=scan,
        drone_pos=quadrotor_state.p,
        target_pos=target_pos,
        drone_vel=quadrotor_state.v,
        last_action=last_actions[-1],
    )

    return new_state, obs


@partial(jax.jit, static_argnames=("static",))
def _step_jit(
    state: DynamicAvoidanceEnvState,
    action: jax.Array,
    key: chex.PRNGKey,
    static: DynamicAvoidanceStatic,
) -> EnvTransition:
    cfg = static.config
    quadrotor = static.quadrotor
    key_dobs, _key_step = jax.random.split(key)

    action_low = jnp.concatenate(
        [jnp.array([quadrotor._thrust_min * 4.0]), quadrotor._omega_max * -1.0]
    )
    action_high = jnp.concatenate(
        [jnp.array([quadrotor._thrust_max * 4.0]), quadrotor._omega_max]
    )
    action = jnp.clip(action, action_low, action_high)

    last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
    last_actions = last_actions.at[-1].set(action)

    dt_1 = cfg.delay % cfg.dt
    action_1 = last_actions[0]
    f_1, omega_1 = action_1[0], action_1[1:]
    quadrotor_state = quadrotor.step(
        state.quadrotor_state, f_1, omega_1, dt_1
    )

    if cfg.delay > 0:
        dt_2 = cfg.dt - dt_1
        action_2 = last_actions[1]
        f_2, omega_2 = action_2[0], action_2[1:]
        quadrotor_state = quadrotor.step(
            quadrotor_state, f_2, omega_2, dt_2
        )

    dobs_state = DynamicObstacleField.update(
        state.dobs_state,
        quadrotor_state.p,
        key_dobs,
        dt=cfg.dt,
        trace_prob=cfg.trace_prob,
        pos_x_range=(-cfg.termination_xy_limit, cfg.termination_xy_limit),
        pos_y_range=(-cfg.termination_xy_limit, cfg.termination_xy_limit),
    )

    next_state = state.replace(
        time=state.time + cfg.dt,
        step_idx=state.step_idx + 1,
        quadrotor_state=quadrotor_state,
        dobs_state=dobs_state,
        last_actions=last_actions,
    )

    scan = static.lidar_sensor.get_scan(
        quadrotor_state.p,
        quadrotor_state.R,
        dobs_state.pos_xy,
        stop_lidar_grad=cfg.stop_lidar_grad,
    )
    obs = ObservationBuilder.get_observation(
        lidar_scan=scan,
        drone_pos=quadrotor_state.p,
        target_pos=state.target_pos,
        drone_vel=quadrotor_state.v,
        last_action=last_actions[-1],
    )

    reward = _get_reward_jit(state, next_state, scan, static)

    dists_to_dobs_xy = jnp.sqrt(jnp.sum((dobs_state.pos_xy - quadrotor_state.p[:2]) ** 2, axis=1) + 1e-8)
    dists_to_dobs = dists_to_dobs_xy - dobs_state.radius

    collision_dobs = jnp.any(dists_to_dobs <= 0.2)
    out_of_bounds = jnp.any(jnp.abs(quadrotor_state.p[:2]) > cfg.termination_xy_limit)
    out_of_height = (quadrotor_state.p[2] < 0.5) | (quadrotor_state.p[2] > 3.5)
    vel_mag = jnp.sqrt(jnp.sum(quadrotor_state.v ** 2) + 1e-8)
    excess_vel = vel_mag > 10.0
    nan_state = (
        jnp.any(jnp.isnan(quadrotor_state.p)) |
        jnp.any(jnp.isnan(quadrotor_state.v)) |
        jnp.any(jnp.isnan(quadrotor_state.R))
    )

    terminated = collision_dobs | out_of_bounds | out_of_height | excess_vel | nan_state
    truncated = next_state.step_idx >= cfg.max_steps_in_episode

    return EnvTransition(
        next_state, obs, reward, terminated, truncated, dict()
    )


def _get_reward_jit(
    last_state: DynamicAvoidanceEnvState,
    next_state: DynamicAvoidanceEnvState,
    scan: jax.Array,
    static: DynamicAvoidanceStatic,
) -> jax.Array:
    cfg = static.config
    quadrotor = static.quadrotor
    pos = next_state.quadrotor_state.p
    prev_pos = last_state.quadrotor_state.p
    vel = next_state.quadrotor_state.v
    target = next_state.target_pos
    last_action = next_state.last_actions[-1]
    prev_action = last_state.last_actions[-1]

    dist_to_goal = jnp.sqrt(jnp.sum((target - pos) ** 2) + 1e-8)
    prev_dist_to_goal = jnp.sqrt(jnp.sum((target - prev_pos) ** 2) + 1e-8)
    r_goal_progress = (prev_dist_to_goal - dist_to_goal) * 10.0
    r_goal_dist = -0.5 * dist_to_goal

    vel_mag = jnp.sqrt(jnp.sum(vel ** 2) + 1e-8)
    r_speed = -1.0 * jax.nn.relu(vel_mag - 5.0) ** 2

    r_height = -2.0 * (jax.nn.relu(0.5 - pos[2]) ** 2 + jax.nn.relu(pos[2] - 3.5) ** 2)

    thrust_hover = 9.81 * quadrotor._mass
    r_action_mag = -0.01 * (last_action[0] - thrust_hover) ** 2 - 0.01 * jnp.sum(last_action[1:] ** 2)
    r_action_smooth = -0.01 * jnp.sum((last_action - prev_action) ** 2)

    max_scan_val = jnp.max(scan)
    min_dist_to_obs = cfg.cutoff_dist - max_scan_val
    r_clearance = -5.0 * jax.nn.relu(1.5 - min_dist_to_obs) ** 2

    dobs_state = next_state.dobs_state
    dists_to_dobs_xy = jnp.sqrt(jnp.sum((dobs_state.pos_xy - pos[:2]) ** 2, axis=1) + 1e-8)
    dists_to_dobs = dists_to_dobs_xy - dobs_state.radius
    r_dobs_risk = -5.0 * jnp.sum(jax.nn.relu(1.5 - dists_to_dobs) ** 2)

    return r_goal_progress + r_goal_dist + r_speed + r_height + r_action_mag + r_action_smooth + r_clearance + r_dobs_risk


class DynamicAvoidanceEnv(env_base.Env[DynamicAvoidanceEnvState]):
    """Environment for quadrotor dynamic obstacle avoidance using JAX differentiable simulation."""

    def __init__(
        self,
        *,
        config: Optional[DynamicAvoidanceConfig] = None,
        max_steps_in_episode: int = 500,
        dt: float = 0.02,
        delay: float = 0.02,
        drone_path: Optional[str] = None,
        trace_prob: float = 0.3,
        stop_lidar_grad: bool = False,
        cutoff_dist: float = 10.0,
        dobs_height: float = 4.0,
        arena_half_extent: float = 20.0,
        termination_margin: float = 2.0,
        reset_margin: float = 2.0,
        reset_inner_extent: float = 20.0,
        reset_target_offset: float = 12.0,
    ):
        if config is None:
            config = DynamicAvoidanceConfig(
                max_steps_in_episode=max_steps_in_episode,
                dt=dt,
                delay=delay,
                drone_path=drone_path,
                trace_prob=trace_prob,
                stop_lidar_grad=stop_lidar_grad,
                cutoff_dist=cutoff_dist,
                dobs_height=dobs_height,
                arena_half_extent=arena_half_extent,
                termination_margin=termination_margin,
                reset_margin=reset_margin,
                reset_inner_extent=reset_inner_extent,
                reset_target_offset=reset_target_offset,
            )

        self.config = config
        self._static = _get_dynamic_avoidance_static(config)
        self.max_steps_in_episode = config.max_steps_in_episode
        self.dt = np.array(config.dt)
        self.delay = np.array(config.delay)
        self.trace_prob = config.trace_prob
        self.stop_lidar_grad = config.stop_lidar_grad
        self.cutoff_dist = config.cutoff_dist
        self.dobs_height = config.dobs_height
        self.arena_half_extent = config.arena_half_extent
        self.termination_xy_limit = config.termination_xy_limit
        self.reset_path_extent = config.reset_path_extent
        self.reset_inner_extent = config.reset_inner_extent
        self.reset_target_offset = config.reset_target_offset

        self.quadrotor = self._static.quadrotor

        self.omega_min = self.quadrotor._omega_max * -1.0
        self.omega_max = self.quadrotor._omega_max
        self.thrust_min = self.quadrotor._thrust_min
        self.thrust_max = self.quadrotor._thrust_max

        self.num_last_actions = config.num_last_actions
        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        self.lidar_sensor = self._static.lidar_sensor

    def reset(
        self, key: chex.PRNGKey, state: Optional[DynamicAvoidanceEnvState] = None
    ) -> tuple[DynamicAvoidanceEnvState, jax.Array]:
        del state
        return _reset_jit(key, self._static)

    def _step(
        self, state: DynamicAvoidanceEnvState, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:
        return _step_jit(state, action, key, self._static)

    def _get_reward(
        self, last_state: DynamicAvoidanceEnvState, next_state: DynamicAvoidanceEnvState, scan: jax.Array
    ) -> jax.Array:
        # Fetch current and previous state details
        pos = next_state.quadrotor_state.p
        prev_pos = last_state.quadrotor_state.p
        vel = next_state.quadrotor_state.v
        target = next_state.target_pos
        last_action = next_state.last_actions[-1]
        prev_action = last_state.last_actions[-1]

        # 1. Goal Progress
        dist_to_goal = jnp.sqrt(jnp.sum((target - pos) ** 2) + 1e-8)
        prev_dist_to_goal = jnp.sqrt(jnp.sum((target - prev_pos) ** 2) + 1e-8)
        r_goal_progress = (prev_dist_to_goal - dist_to_goal) * 10.0
        r_goal_dist = -0.5 * dist_to_goal

        # 2. Speed Band
        vel_mag = jnp.sqrt(jnp.sum(vel ** 2) + 1e-8)
        r_speed = -1.0 * jax.nn.relu(vel_mag - 5.0) ** 2

        # 3. Height Band
        r_height = -2.0 * (jax.nn.relu(0.5 - pos[2]) ** 2 + jax.nn.relu(pos[2] - 3.5) ** 2)

        # 4. Action Magnitude (deviation from hover thrust and angular rate size)
        thrust_hover = 9.81 * self.quadrotor._mass
        r_action_mag = -0.01 * (last_action[0] - thrust_hover) ** 2 - 0.01 * jnp.sum(last_action[1:] ** 2)

        # 5. Action Smoothness (jerk penalty)
        r_action_smooth = -0.01 * jnp.sum((last_action - prev_action) ** 2)

        # 6. Soft Obstacle Clearance (minimum clearance in LiDAR scan)
        max_scan_val = jnp.max(scan)
        min_dist_to_obs = self.cutoff_dist - max_scan_val
        r_clearance = -5.0 * jax.nn.relu(1.5 - min_dist_to_obs) ** 2

        # 7. Dynamic Obstacles Risk (using explicit positions)
        dobs_state = next_state.dobs_state
        dists_to_dobs_xy = jnp.sqrt(jnp.sum((dobs_state.pos_xy - pos[:2]) ** 2, axis=1) + 1e-8)
        dists_to_dobs = dists_to_dobs_xy - dobs_state.radius
        r_dobs_risk = -5.0 * jnp.sum(jax.nn.relu(1.5 - dists_to_dobs) ** 2)

        return r_goal_progress + r_goal_dist + r_speed + r_height + r_action_mag + r_action_smooth + r_clearance + r_dobs_risk

    def compute_p2m_reward(
        self, last_state: DynamicAvoidanceEnvState, next_state: DynamicAvoidanceEnvState, scan: jax.Array
    ) -> Dict[str, jax.Array]:
        """Compute the exact P2M aligned reward components for logging or evaluation."""
        pos = next_state.quadrotor_state.p
        prev_pos = last_state.quadrotor_state.p
        vel = next_state.quadrotor_state.v
        target = next_state.target_pos
        
        action = next_state.last_actions[-1]
        prev_action = last_state.last_actions[-1]

        # 1. Distances to goal
        dist_to_goal = jnp.sqrt(jnp.sum((target - pos) ** 2) + 1e-8)
        prev_dist_to_goal = jnp.sqrt(jnp.sum((target - prev_pos) ** 2) + 1e-8)
        touch_goal = dist_to_goal <= 3.0

        # 2. Velocity reward
        vel_mag = jnp.sqrt(jnp.sum(vel ** 2) + 1e-8)
        r_vel = jnp.log(jnp.exp(- 2.0 * jnp.maximum(vel_mag - 5.0, 0.0)) + 1.0)

        # 3. Acceleration reward
        acc_mag = jnp.sqrt(jnp.sum(next_state.quadrotor_state.acc ** 2) + 1e-8)
        r_acc = jnp.log(jnp.exp(- 5.0 * jnp.maximum(acc_mag - 5.0, 0.0)) + 1.0)

        # 4. Jerk reward
        r_jerk = 1.0 / (1.0 + jnp.sqrt(jnp.sum((action - prev_action) ** 2) + 1e-8))

        # 5. Height reward
        r_height = jnp.log(jnp.exp(- 2.0 * (jnp.maximum(1.5 - pos[2], 0.0) + jnp.maximum(pos[2] - 2.5, 0.0))) + 1.0)

        # 6. Goal reward
        vel_direction = (target - pos) / dist_to_goal
        r_goal_dir = jnp.minimum(jnp.sum(vel * vel_direction), 2.0)
        r_goal_dis = jnp.where(touch_goal, 0.0, (jnp.exp(prev_dist_to_goal - dist_to_goal) - 1.0) * 10.0)
        r_goal = r_goal_dir + r_goal_dis

        # 7. Safety reward
        distances = self.cutoff_dist - scan
        distances_flat = distances.flatten()
        distances_clip = jnp.maximum(distances_flat - 1.0, 0.0)
        obs_mask = distances_flat <= 1.0
        obs_count = jnp.sum(obs_mask)
        
        obs_dist = jax.lax.cond(
            obs_count > 0,
            lambda _: jnp.sum(distances_clip * obs_mask) / obs_count,
            lambda _: jnp.min(distances_clip),
            None
        )
        r_safety = jnp.maximum(jnp.log(obs_dist + 1e-8), -5.0)

        # 8. Dynamic obstacle reward (dobs)
        dobs_state = next_state.dobs_state
        drone_pos_2d = pos[:2]
        drone_vel_2d = vel[:2]

        def compute_single_dobs(pos_obs, vel_obs, rad):
            r = pos_obs - drone_pos_2d
            obstacle_vel_drone = vel_obs - drone_vel_2d
            dot_product = jnp.sum(r * obstacle_vel_drone)
            r_norm = jnp.linalg.norm(r)
            v_norm = jnp.linalg.norm(obstacle_vel_drone)
            cos_theta = jnp.clip(dot_product / (r_norm * v_norm + 1e-8), -1.0, 1.0)
            theta = jnp.arccos(cos_theta)
            coll_mask = theta < (jnp.pi / 2.0)
            vel_magnitude = jnp.linalg.norm(vel_obs)
            dist = r_norm - rad
            
            unit_velocity = vel_obs / (vel_magnitude + 1e-6)
            speed_line_distance = jnp.abs(r[0] * unit_velocity[1] - r[1] * unit_velocity[0])
            
            fov_mask = dist <= 7.5
            k_v = v_norm
            k_theta = 1.0 - (2.0 * theta / jnp.pi)
            k_d = jnp.exp(1.0 / (1.0 + speed_line_distance))
            k_total = jnp.where(coll_mask, 1.0 + k_v * k_theta * k_d, 1.0)
            
            r_d_zoom = jnp.maximum(dist - 1.0, 0.0) / (k_total + 1e-8)
            r_d = jnp.maximum(jnp.log(r_d_zoom + 1e-8), -5.0)
            return r_d, fov_mask

        r_ds, fov_masks = jax.vmap(compute_single_dobs)(dobs_state.pos_xy, dobs_state.vel_xy, dobs_state.radius)
        obs_count_dobs = jnp.maximum(jnp.sum(fov_masks), 1.0)
        r_dobs = jnp.sum(r_ds * fov_masks) / obs_count_dobs

        total = (1.2 * r_vel + 0.6 * r_acc + 0.2 * r_jerk + 0.3 * r_height +
                 0.8 * r_goal + 1.0 * r_safety + 0.6 * r_dobs)

        return {
            "reward_velocity": r_vel,
            "reward_acceleration": r_acc,
            "reward_jerk": r_jerk,
            "reward_height": r_height,
            "reward_goal": r_goal,
            "reward_safety": r_safety,
            "reward_dobs": r_dobs,
            "reward_total": total
        }


    @property
    def action_space(self) -> spaces.Box:
        low = jnp.concatenate(
            [jnp.array([self.thrust_min * 4.0]), self.omega_min]
        )
        high = jnp.concatenate(
            [jnp.array([self.thrust_max * 4.0]), self.omega_max]
        )
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        return ObservationBuilder.get_observation_space(self.action_space, self.cutoff_dist)
