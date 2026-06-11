import os
import mujoco
import jax
import jax.numpy as jnp
from mujoco_lidar.core_jax.mjlidar_jax import MjLidarJax
from flightning.modules.dynamic_obstacle_field import DynamicObstacleField

class MujocoLidarSensor:
    """LiDAR Sensor path using MjLidarJax."""

    def __init__(
        self,
        *,
        scan_mode: str = "p2m_oversample",
        cutoff_dist: float = 10.0,
        dobs_height: float = 4.0
    ):
        self.scan_mode = scan_mode
        self.cutoff_dist = cutoff_dist
        self.dobs_height = dobs_height

        if scan_mode != "p2m_oversample":
            raise ValueError(f"Scan mode '{scan_mode}' is not supported in the first version.")

        # Load avoidance arena XML
        xml_path = os.path.join(os.path.dirname(__file__), "avoidance_arena.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        # Initialize MjLidarJax
        self.mj_lidar_jax = MjLidarJax(self.mj_model)

        self.box_ids = self.mj_lidar_jax.box_ids
        self.cylinder_ids = self.mj_lidar_jax.cylinder_ids

        # Fetch static wall positions and orientations
        self.wall_xpos = jnp.array(self.mj_model.geom_pos[self.box_ids])
        
        # Convert geom_quat to rotation matrices
        import numpy as np
        quats = self.mj_model.geom_quat[self.box_ids]  # shape (N, 4)
        mats = []
        for q in quats:
            # MuJoCo format: (w, x, y, z)
            w, x, y, z = q
            x2, y2, z2 = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            r = np.array([
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)]
            ])
            mats.append(r)
        
        if len(mats) > 0:
            self.wall_xmat = jnp.array(np.stack(mats, axis=0))
        else:
            self.wall_xmat = jnp.zeros((0, 3, 3))


        # Oversample scanning pattern: 108 x 18
        self.h_num = 108
        self.v_num = 18
        h_angles = jnp.linspace(0.0, 360.0, self.h_num + 1)[:-1] * jnp.pi / 180.0
        v_angles = jnp.linspace(-7.0, 52.0, self.v_num) * jnp.pi / 180.0
        theta, phi = jnp.meshgrid(h_angles, v_angles, indexing="ij")
        self.ray_theta = theta.flatten()
        self.ray_phi = phi.flatten()

    def get_scan(
        self,
        drone_pos: jax.Array,
        drone_rot: jax.Array,
        dobs_pos_xy: jax.Array,
        stop_lidar_grad: bool = False
    ) -> jax.Array:
        """Compute the default occupancy-like LiDAR distance image."""
        # 1. Mount yaw-only sensor
        norm = jnp.sqrt(drone_rot[0, 0]**2 + drone_rot[1, 0]**2 + 1e-8)
        yaw_cos = drone_rot[0, 0] / norm
        yaw_sin = drone_rot[1, 0] / norm

        sensor_mat = jnp.array([
            [yaw_cos, -yaw_sin, 0.0],
            [yaw_sin, yaw_cos, 0.0],
            [0.0, 0.0, 1.0]
        ])
        sensor_pos = drone_pos

        # 2. Derive obstacle cylinder poses
        cylinder_xpos, cylinder_xmat = DynamicObstacleField.get_cylinder_poses(dobs_pos_xy, self.dobs_height)

        # 3. Assemble geom poses
        geom_xpos = jnp.zeros((self.mj_model.ngeom, 3))
        geom_xmat = jnp.tile(jnp.eye(3)[None, :, :], (self.mj_model.ngeom, 1, 1))

        geom_xpos = geom_xpos.at[self.box_ids].set(self.wall_xpos)
        geom_xmat = geom_xmat.at[self.box_ids].set(self.wall_xmat)

        geom_xpos = geom_xpos.at[self.cylinder_ids].set(cylinder_xpos)
        geom_xmat = geom_xmat.at[self.cylinder_ids].set(cylinder_xmat)

        # 4. Stop gradients if requested (dual visual gradient mode)
        if stop_lidar_grad:
            geom_xpos = jax.lax.stop_gradient(geom_xpos)
            geom_xmat = jax.lax.stop_gradient(geom_xmat)
            sensor_pos = jax.lax.stop_gradient(sensor_pos)
            sensor_mat = jax.lax.stop_gradient(sensor_mat)

        # 5. Raycasting
        distances, local_rays = self.mj_lidar_jax.trace_rays(
            geom_xpos, geom_xmat, sensor_pos, sensor_mat, self.ray_theta, self.ray_phi
        )

        # 6. Postprocess
        # Replace 0.0 (no hit) with cutoff_dist
        distances_processed = jnp.where(distances <= 0.0, self.cutoff_dist, distances)

        # Z height filtering: set distances of out-of-height hits to cutoff_dist
        world_rays = local_rays @ sensor_mat.T
        hit_z = sensor_pos[2] + distances_processed * world_rays[:, 2]
        valid_z = (hit_z >= 0.0) & (hit_z <= self.dobs_height)
        distances_processed = jnp.where(valid_z, distances_processed, self.cutoff_dist)

        # 3x3 min-pooling from (108, 18) to (36, 6)
        grid = distances_processed.reshape(self.h_num, self.v_num)
        pooled = jnp.min(grid.reshape(36, 3, 6, 3), axis=(1, 3))

        # Occupancy inversion: cutoff_dist - distance
        occupancy = self.cutoff_dist - pooled

        return occupancy[None, :, :]
