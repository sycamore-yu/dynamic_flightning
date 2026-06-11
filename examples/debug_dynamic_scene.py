#!/usr/bin/env python3
"""Debug script: visualize the dynamic avoidance environment with rerun.

Usage:
    python examples/debug_dynamic_scene.py                     # save to .rrd
    python examples/debug_dynamic_scene.py --serve             # live streaming (headless server)
    python examples/debug_dynamic_scene.py --show              # live viewer (local display)
    python examples/debug_dynamic_scene.py --serve --output rec.rrd  # live + save
    python examples/debug_dynamic_scene.py --scan-mode mid360_livox  # Mid-360 LiDAR
    python examples/debug_dynamic_scene.py --steps 500 --seed 42
"""

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
from flightning.modules.dynamic_obstacle_field import DynamicObstacleField
from flightning.visualization.rerun_dynamic_avoidance import RerunVizAdapter, HAS_RERUN

if HAS_RERUN:
    import rerun as rr


class DebugRerunAdapter(RerunVizAdapter):
    """Extends RerunVizAdapter with spawn/serve modes for live viewing."""

    def __init__(self, spawn: bool = False, serve: bool = False, **kwargs):
        self.dobs_height = kwargs.get("dobs_height", 4.0)
        self.cutoff_dist = kwargs.get("cutoff_dist", 10.0)
        self.save_path = kwargs.get("save_path", None)
        self.history_positions = []
        self.initialized = False

        if not HAS_RERUN:
            print("Warning: rerun-sdk is not installed. Visualization is disabled.")
            return

        rr.init("debug_dynamic_scene", spawn=spawn)

        if serve:
            uri = rr.serve_grpc()
            print(f"gRPC server started at: {uri}")
            print("Connect a viewer with:")
            print(f"  rerun {uri}")

        if self.save_path is not None:
            parent_dir = os.path.dirname(self.save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            rr.save(self.save_path)


class Mid360Raycaster:
    """Mid-360 Livox LiDAR raycaster using LivoxGenerator + MjLidarJax."""

    def __init__(self, cutoff_dist: float = 10.0, dobs_height: float = 4.0):
        from mujoco_lidar.scan_gen import LivoxGenerator
        from mujoco_lidar.core_jax.mjlidar_jax import MjLidarJax
        import mujoco

        self.cutoff_dist = cutoff_dist
        self.dobs_height = dobs_height

        self.generator = LivoxGenerator("mid360")
        self.n_rays = self.generator.samples  # 24000

        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "flightning", "sensors", "avoidance_arena.xml"
        )
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_lidar_jax = MjLidarJax(self.mj_model)
        self.box_ids = self.mj_lidar_jax.box_ids
        self.cylinder_ids = self.mj_lidar_jax.cylinder_ids

        self.wall_xpos = jnp.array(self.mj_model.geom_pos[self.box_ids])
        quats = self.mj_model.geom_quat[self.box_ids]
        mats = []
        for q in quats:
            w, x, y, z = q
            x2, y2, z2 = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            r = np.array([
                [1.0 - 2.0*(y2+z2), 2.0*(xy-wz), 2.0*(xz+wy)],
                [2.0*(xy+wz), 1.0 - 2.0*(x2+z2), 2.0*(yz-wx)],
                [2.0*(xz-wy), 2.0*(yz+wx), 1.0 - 2.0*(x2+y2)]
            ])
            mats.append(r)
        self.wall_xmat = jnp.array(np.stack(mats, axis=0)) if mats else jnp.zeros((0, 3, 3))

    def get_scan_and_hits(self, drone_pos, drone_rot, dobs_pos_xy):
        """Run mid360 raycast. Returns (scan_image_36x6, hit_points_3d, n_valid_hits)."""
        theta_np, phi_np = self.generator.sample_ray_angles()
        ray_theta = jnp.array(theta_np)
        ray_phi = jnp.array(phi_np)

        norm = jnp.sqrt(drone_rot[0, 0]**2 + drone_rot[1, 0]**2 + 1e-8)
        yaw_cos = drone_rot[0, 0] / norm
        yaw_sin = drone_rot[1, 0] / norm
        sensor_mat = jnp.array([
            [yaw_cos, -yaw_sin, 0.0],
            [yaw_sin, yaw_cos, 0.0],
            [0.0, 0.0, 1.0]
        ])
        sensor_pos = drone_pos

        cylinder_xpos, cylinder_xmat = DynamicObstacleField.get_cylinder_poses(
            dobs_pos_xy, self.dobs_height
        )
        geom_xpos = jnp.zeros((self.mj_model.ngeom, 3))
        geom_xmat = jnp.tile(jnp.eye(3)[None, :, :], (self.mj_model.ngeom, 1, 1))
        geom_xpos = geom_xpos.at[self.box_ids].set(self.wall_xpos)
        geom_xmat = geom_xmat.at[self.box_ids].set(self.wall_xmat)
        geom_xpos = geom_xpos.at[self.cylinder_ids].set(cylinder_xpos)
        geom_xmat = geom_xmat.at[self.cylinder_ids].set(cylinder_xmat)

        distances, local_rays = self.mj_lidar_jax.trace_rays(
            geom_xpos, geom_xmat, sensor_pos, sensor_mat, ray_theta, ray_phi
        )

        distances_processed = jnp.where(distances <= 0.0, self.cutoff_dist, distances)
        world_rays = local_rays @ sensor_mat.T
        hit_z = sensor_pos[2] + distances_processed * world_rays[:, 2]
        valid_z = (hit_z >= 0.0) & (hit_z <= self.dobs_height)
        distances_processed = jnp.where(valid_z, distances_processed, self.cutoff_dist)

        valid_hit = distances_processed < (self.cutoff_dist - 0.01)
        hit_points_np = np.array(sensor_pos + distances_processed[:, None] * world_rays)
        valid_mask = np.array(valid_hit)
        hit_points_3d = hit_points_np[valid_mask]

        h_bins, v_bins = 36, 6
        h_edges = np.linspace(0.0, 2.0 * np.pi, h_bins + 1)
        v_edges = np.linspace(np.deg2rad(-7.0), np.deg2rad(52.0), v_bins + 1)
        theta_np_mod = theta_np % (2.0 * np.pi)
        dist_np = np.array(distances_processed)

        grid = np.full((h_bins, v_bins), self.cutoff_dist)
        h_idx = np.clip(np.digitize(theta_np_mod, h_edges) - 1, 0, h_bins - 1)
        v_idx = np.clip(np.digitize(phi_np, v_edges) - 1, 0, v_bins - 1)
        np.minimum.at(grid, (h_idx, v_idx), dist_np)

        occupancy = self.cutoff_dist - grid
        scan_image = occupancy[None, :, :].astype(np.float32)

        return scan_image, hit_points_3d, int(valid_mask.sum())


def parse_args():
    parser = argparse.ArgumentParser(description="Debug: visualize dynamic avoidance scene with rerun")
    parser.add_argument("--steps", type=int, default=300, help="Total simulation steps (default: 300)")
    parser.add_argument("--show", action="store_true", help="Spawn live rerun viewer (requires display)")
    parser.add_argument("--serve", action="store_true", help="Start gRPC server for live streaming (headless server)")
    parser.add_argument("--output", type=str, default=None, help="Also save to .rrd file (can combine with --serve/--show)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--print-every", type=int, default=50, help="Print status every N steps (default: 50)")
    parser.add_argument(
        "--scan-mode", type=str, default="p2m_oversample",
        choices=["p2m_oversample", "mid360_livox"],
        help="LiDAR scan mode (default: p2m_oversample)"
    )
    return parser.parse_args()


def simple_policy(state, obs, env):
    """Proportional controller: fly toward target while maintaining hover."""
    hover_thrust = 9.81 * env.quadrotor._mass

    target_dir = obs[216:219]
    vel_norm = obs[219:222]
    vel = vel_norm * 5.0

    pos = np.array(state.quadrotor_state.p)
    R = np.array(state.quadrotor_state.R)

    kp = 1.5
    desired_vel = target_dir * kp
    vel_error_world = desired_vel - vel
    vel_error_body = R.T @ vel_error_world

    kp_attitude = 0.8
    omega_x = -kp_attitude * vel_error_body[1]
    omega_y = kp_attitude * vel_error_body[0]
    omega_z = 0.0

    target_height = 2.0
    height_error = target_height - pos[2]
    thrust = hover_thrust + 2.0 * height_error

    max_omega = 3.0
    omega_x = float(np.clip(omega_x, -max_omega, max_omega))
    omega_y = float(np.clip(omega_y, -max_omega, max_omega))

    return jnp.array([thrust, omega_x, omega_y, omega_z])


def main():
    args = parse_args()

    if not HAS_RERUN:
        print("Error: rerun-sdk is not installed. Install with: pip install rerun-sdk")
        return

    if not args.serve and not args.show and args.output is None:
        args.output = "debug_dynamic_scene.rrd"

    env = DynamicAvoidanceEnv()

    mid360 = None
    if args.scan_mode == "mid360_livox":
        print("Initializing Mid-360 Livox raycaster...")
        mid360 = Mid360Raycaster(
            cutoff_dist=env.cutoff_dist,
            dobs_height=env.dobs_height,
        )

    save_path = args.output
    if args.show:
        viz = DebugRerunAdapter(
            spawn=True,
            dobs_height=env.dobs_height,
            cutoff_dist=env.cutoff_dist,
            save_path=save_path,
        )
    elif args.serve:
        viz = DebugRerunAdapter(
            serve=True,
            dobs_height=env.dobs_height,
            cutoff_dist=env.cutoff_dist,
            save_path=save_path,
        )
    else:
        viz = RerunVizAdapter(
            save_path=save_path,
            dobs_height=env.dobs_height,
            cutoff_dist=env.cutoff_dist,
        )

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)

    print("Warming up JIT compilation...")
    state, obs = env.reset(reset_key)

    if mid360 is not None:
        print("Warming up mid360 raycast JIT...")
        _scan, _hits, _n = mid360.get_scan_and_hits(
            state.quadrotor_state.p, state.quadrotor_state.R, state.dobs_state.pos_xy
        )

    print(f"Starting debug visualization for {args.steps} steps...")
    print(f"Scan mode: {args.scan_mode}")
    if mid360:
        print(f"  Mid-360 rays per frame: {mid360.n_rays}")
    print(f"Hover thrust: {9.81 * env.quadrotor._mass:.4f} N")
    print(f"Arena: ±{env.termination_xy_limit:.1f}m XY, 0.5-3.5m Z")
    print(f"Dynamic obstacles: 40 cylinders")
    print("-" * 70)

    episode_count = 0

    for step in range(args.steps):
        action = simple_policy(state, obs, env)

        key, step_key = jax.random.split(key)
        transition = env._step(state, action, step_key)

        state = transition.state
        obs = transition.obs
        terminated = bool(transition.terminated)
        truncated = bool(transition.truncated)

        if mid360 is not None:
            scan_image, hit_points_3d, n_hits = mid360.get_scan_and_hits(
                state.quadrotor_state.p, state.quadrotor_state.R, state.dobs_state.pos_xy
            )
            viz.log_state(state, scan_image, step_idx=step)
            if n_hits > 0:
                rr.log("world/lidar_hits", rr.Points3D(hit_points_3d, radii=[0.05], colors=[[255, 0, 0]]))
            else:
                rr.log("world/lidar_hits", rr.Clear(recursive=False))
        else:
            scan_image = obs[:216].reshape(1, 36, 6)
            viz.log_state(state, scan_image, step_idx=step)

        if step % args.print_every == 0:
            pos = np.array(state.quadrotor_state.p)
            target = np.array(state.target_pos)
            dist_to_target = float(np.linalg.norm(target - pos))
            dobs_pos = np.array(state.dobs_state.pos_xy)
            dobs_dists = np.sqrt(np.sum((dobs_pos - pos[:2]) ** 2, axis=1))
            nearby = int(np.sum(dobs_dists < 3.0))
            extra = f" | hits={n_hits}" if mid360 else ""
            print(
                f"  Step {step:4d} | pos=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f}) "
                f"| dist_target={dist_to_target:5.2f} | nearby_obs={nearby}"
                f"{extra} | episode={episode_count}"
            )

        if terminated or truncated:
            reason = "collision/OOB" if terminated else "truncated"
            print(f"  Episode ended at step {step} ({reason}). Resetting...")
            episode_count += 1
            key, reset_key = jax.random.split(key)
            state, obs = env.reset(reset_key)
            viz.history_positions = []

    print("-" * 70)
    print(f"Done. {episode_count + 1} episode(s) completed.")
    if args.serve:
        print("gRPC server stopped.")
    if save_path:
        abs_path = os.path.abspath(save_path)
        print(f"Recording saved to: {abs_path}")
        print(f"View with: rerun {abs_path}")


if __name__ == "__main__":
    main()
