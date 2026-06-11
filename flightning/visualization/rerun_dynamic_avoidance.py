import os
import numpy as np
import jax.numpy as jnp

try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

class RerunVizAdapter:
    """Rerun visualization adapter for dynamic avoidance environment."""

    def __init__(
        self,
        save_path: str = None,
        dobs_height: float = 4.0,
        cutoff_dist: float = 10.0
    ):
        self.dobs_height = dobs_height
        self.cutoff_dist = cutoff_dist
        self.save_path = save_path
        self.history_positions = []
        self.initialized = False

        if not HAS_RERUN:
            print("Warning: rerun-sdk is not installed. Visualization is disabled.")
            return

        # Initialize Rerun
        rr.init("flightning_dynamic_avoidance", spawn=False)
        if save_path is not None:
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            rr.save(save_path)

    def log_static_scene(self):
        if not HAS_RERUN:
            return

        # 1. Log static walls
        # East, West, North, South
        wall_centers = np.array([
            [20.0, 0.0, 2.0],
            [-20.0, 0.0, 2.0],
            [0.0, 20.0, 2.0],
            [0.0, -20.0, 2.0]
        ])
        wall_half_sizes = np.array([
            [0.5, 20.0, 2.0],
            [0.5, 20.0, 2.0],
            [20.0, 0.5, 2.0],
            [20.0, 0.5, 2.0]
        ])
        rr.log("world/walls", rr.Boxes3D(half_sizes=wall_half_sizes, centers=wall_centers), static=True)

    def log_state(self, state, scan_image, step_idx: int = None):
        """Log the environment state and LiDAR scan to Rerun.

        :param state: DynamicAvoidanceEnvState
        :param scan_image: shape (1, 36, 6) or (36, 6)
        """
        if not HAS_RERUN:
            return

        if not self.initialized:
            self.log_static_scene()
            self.initialized = True

        time = float(state.time)
        step = int(state.step_idx) if step_idx is None else step_idx

        # Set timelines
        rr.set_time("stable_time", duration=time)
        rr.set_time("step", sequence=step)

        # 1. Log Drone pose
        drone_pos = np.array(state.quadrotor_state.p)
        drone_rot = np.array(state.quadrotor_state.R)
        rr.log("world/drone", rr.Transform3D(translation=drone_pos, mat3x3=drone_rot))

        # Log drone simple proxy
        # Body box
        rr.log("world/drone/body", rr.Boxes3D(half_sizes=[0.15, 0.15, 0.05], colors=[[0, 150, 255]]))

        # Log coordinate axes
        rr.log("world/drone/axes", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ))

        # 2. Log Trajectory
        self.history_positions.append(drone_pos)
        rr.log("world/drone/path", rr.LineStrips3D([self.history_positions], colors=[[0, 255, 255]]))

        # 3. Log Target position
        target_pos = np.array(state.target_pos)
        rr.log("world/target", rr.Points3D([target_pos], radii=[0.5], colors=[[0, 255, 0]]))

        # 4. Log Dynamic Obstacles
        dobs_pos_xy = np.array(state.dobs_state.pos_xy)
        dobs_radius = np.array(state.dobs_state.radius)
        num_obs = dobs_pos_xy.shape[0]

        centers = np.stack([
            dobs_pos_xy[:, 0],
            dobs_pos_xy[:, 1],
            np.ones(num_obs) * (self.dobs_height / 2.0)
        ], axis=1)

        rr.log(
            "world/obstacles",
            rr.Cylinders3D(
                lengths=np.ones(num_obs) * self.dobs_height,
                radii=dobs_radius,
                centers=centers,
                colors=[[255, 100, 0] for _ in range(num_obs)]
            )
        )

        # 5. Log LiDAR Scan Image
        scan_image_np = np.array(scan_image)
        if scan_image_np.ndim == 3:
            scan_image_np = scan_image_np[0]
        # Normalize to [0, 255] for visual occupancy (closer is brighter)
        scan_vis = (scan_image_np / self.cutoff_dist * 255.0).astype(np.uint8)
        rr.log("drone/lidar/scan_image", rr.Image(scan_vis))

        # 6. Log LiDAR hits in world frame by reconstructing rays
        h_num, v_num = 36, 6
        h_angles = np.linspace(0.0, 360.0, h_num + 1)[:-1] * np.pi / 180.0
        v_angles = np.linspace(-7.0, 52.0, v_num) * np.pi / 180.0
        theta, phi = np.meshgrid(h_angles, v_angles, indexing="ij")
        theta = theta.flatten()
        phi = phi.flatten()

        local_rays = np.stack([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi)
        ], axis=-1)

        norm = np.sqrt(drone_rot[0, 0]**2 + drone_rot[1, 0]**2 + 1e-8)
        yaw_cos = drone_rot[0, 0] / norm
        yaw_sin = drone_rot[1, 0] / norm
        sensor_mat = np.array([
            [yaw_cos, -yaw_sin, 0.0],
            [yaw_sin, yaw_cos, 0.0],
            [0.0, 0.0, 1.0]
        ])

        flat_scan = scan_image_np.flatten()
        distances = self.cutoff_dist - flat_scan

        valid_hit = distances < (self.cutoff_dist - 0.01)
        if np.any(valid_hit):
            world_rays = local_rays @ sensor_mat.T
            hit_points = drone_pos + distances[:, None] * world_rays
            valid_hit_points = hit_points[valid_hit]
            rr.log("world/drone/lidar/hits", rr.Points3D(valid_hit_points, radii=[0.1], colors=[[255, 0, 0]]))
        else:
            rr.log("world/drone/lidar/hits", rr.Clear(recursive=False))
