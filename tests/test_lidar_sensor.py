import jax
import jax.numpy as jnp
from flightning.sensors.mujoco_lidar_sensor import MujocoLidarSensor

def test_lidar_sensor_basic():
    sensor = MujocoLidarSensor()
    drone_pos = jnp.array([0.0, 0.0, 1.0])
    drone_rot = jnp.eye(3)
    dobs_pos_xy = jnp.zeros((40, 2))

    # 1. Output shape test
    scan = sensor.get_scan(drone_pos, drone_rot, dobs_pos_xy, stop_lidar_grad=False)
    assert scan.shape == (1, 36, 6)

    # 2. Check yaw-only mounting
    # A roll/pitch rotation should yield the same sensor orientation (and therefore same scan)
    # if the yaw is 0.
    pitch_angle = 0.3
    pitch_rot = jnp.array([
        [jnp.cos(pitch_angle), 0.0, jnp.sin(pitch_angle)],
        [0.0, 1.0, 0.0],
        [-jnp.sin(pitch_angle), 0.0, jnp.cos(pitch_angle)]
    ])
    scan_identity = sensor.get_scan(drone_pos, jnp.eye(3), dobs_pos_xy)
    scan_pitch = sensor.get_scan(drone_pos, pitch_rot, dobs_pos_xy)
    assert jnp.allclose(scan_identity, scan_pitch)

def test_lidar_sensor_gradients():
    sensor = MujocoLidarSensor()
    dobs_pos_xy = jnp.zeros((40, 2))
    drone_rot = jnp.eye(3)

    def loss_fn(pos, stop_grad):
        scan = sensor.get_scan(pos, drone_rot, dobs_pos_xy, stop_lidar_grad=stop_grad)
        return jnp.sum(scan)

    grad_fn = jax.grad(loss_fn, argnums=0)

    # Under stop_lidar_grad=True, gradients must be exactly zero
    g_stop = grad_fn(jnp.array([0.0, 0.0, 1.0]), True)
    assert jnp.allclose(g_stop, 0.0)

    # Under stop_lidar_grad=False (analytic_lidar_grad), gradients should be non-zero
    # since changing position changes raycast distances to walls/cylinders
    g_active = grad_fn(jnp.array([1.0, 1.0, 1.0]), False)
    assert not jnp.allclose(g_active, 0.0)
