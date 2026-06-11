from .double_sphere_camera import DoubleSphereCamera, CameraNames, CameraState

__all__ = [
    "DoubleSphereCamera",
    "CameraNames",
    "CameraState",
    "MujocoLidarSensor",
]


def __getattr__(name):
    if name == "MujocoLidarSensor":
        from .mujoco_lidar_sensor import MujocoLidarSensor

        return MujocoLidarSensor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
