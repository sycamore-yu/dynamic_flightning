# observation-builder Specification

## Purpose
TBD - created by archiving change p2m-dynamic-avoidance. Update Purpose after archive.
## Requirements
### Requirement: Flat Observation Contract
`ObservationBuilder` SHALL return one flat `jax.Array` observation compatible with existing Flightning BPTT/SHAC policy calls.

#### Scenario: Observation consumed by existing algorithms
- **WHEN** BPTT or SHAC calls `actor.apply(params, obs)` with the environment observation
- **THEN** `obs` SHALL be a flat JAX array and SHALL NOT require dict observation support in `LogWrapper`, `VecEnv`, BPTT, or SHAC.

### Requirement: Stable Observation Layout
`ObservationBuilder` SHALL define a stable layout containing LiDAR image data, target direction, drone velocity, and last action.

#### Scenario: Default observation shape
- **WHEN** the default `p2m_oversample` LiDAR mode is used
- **THEN** the observation SHALL contain `lidar_flat(216)`, `target_dir(3)`, `velocity(3)`, and `last_action(4)` in documented order.

### Requirement: Single-Channel LiDAR Input
The first version SHALL use a single-channel LiDAR distance image and SHALL NOT include NeuFlow optical-flow channels.

#### Scenario: P2M three-channel reference exists
- **WHEN** comparing to P2M's original policy input with scan plus flow channels
- **THEN** the Flightning first version SHALL use only the single scan channel and SHALL document that NeuFlow migration is out of scope.

### Requirement: Observation Bounds
`ObservationBuilder` SHALL provide enough metadata for `observation_space` to expose a stable `spaces.Box` shape.

#### Scenario: Observation space queried
- **WHEN** examples or algorithms read `env.observation_space.shape[0]`
- **THEN** the returned value SHALL match the flat observation length.

