# mujoco-lidar-sensor Specification

## Purpose
TBD - created by archiving change p2m-dynamic-avoidance. Update Purpose after archive.
## Requirements
### Requirement: Direct JAX Backend Integration
`mujoco-lidar-sensor` SHALL initialize MuJoCo-LiDAR from a static `mujoco.MjModel` but SHALL call `MjLidarJax.trace_rays` or `trace_rays_batch` directly with Flightning-provided arrays during rollout.

#### Scenario: Rollout raycast does not use mjData
- **WHEN** the sensor computes LiDAR distances during `DynamicAvoidanceEnv._step`
- **THEN** it SHALL pass `geom_xpos`, `geom_xmat`, `sensor_pos`, `sensor_mat`, `ray_theta`, and `ray_phi` arrays to `MjLidarJax` and SHALL NOT call `MjLidarWrapper.trace_rays(mj_data, ...)`.

### Requirement: Supported Geometry Scope
The first version SHALL support geometry that MuJoCo-LiDAR's JAX backend can raycast and that Flightning state can pose directly.

#### Scenario: Default arena geometry
- **WHEN** the default dynamic avoidance arena is used
- **THEN** walls SHALL be represented as boxes and dynamic obstacles SHALL be represented as cylinders.

#### Scenario: Unsupported geometry is rejected
- **WHEN** a mesh or arbitrary MJCF kinematic object is requested for training-time LiDAR collision semantics
- **THEN** the sensor SHALL reject it or mark it unsupported for this change.

### Requirement: P2M Oversample Scan Mode
The default scan mode SHALL be `p2m_oversample`, reproducing P2M's regular oversampling pipeline without NeuFlow.

#### Scenario: P2M scan output
- **WHEN** `p2m_oversample` is active
- **THEN** the sensor SHALL cast `108 x 18 = 1944` rays, apply Z filtering, 3x3 min-pool downsampling, range inversion `lidar_range - distance`, and return a single-channel `(1, 36, 6)` distance image.

### Requirement: Mid360 Modes Are Non-Default Future Scope
`mid360_livox` and `mid360_binned` SHALL NOT be first-version acceptance targets for this change.

#### Scenario: Mid360 mode requested
- **WHEN** a Mid-360 scan mode is referenced by configuration or documentation
- **THEN** the implementation MAY expose placeholders or explicit unsupported errors, but tests for real Mid-360 Livox scan generation SHALL NOT be required for first-version acceptance.

### Requirement: Yaw-Only Sensor Mount
The sensor mount SHALL support a yaw-only attachment mode matching P2M/Isaac `attach_yaw_only=True` semantics.

#### Scenario: Drone roll or pitch changes
- **WHEN** the drone rolls or pitches while yaw remains unchanged
- **THEN** the default LiDAR ray directions SHALL remain yaw-aligned and SHALL NOT roll or pitch with the vehicle body.

