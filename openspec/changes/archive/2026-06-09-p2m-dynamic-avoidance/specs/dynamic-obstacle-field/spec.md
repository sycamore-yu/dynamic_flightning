## ADDED Requirements

### Requirement: P2M-Compatible Obstacle State
`DynamicObstacleField` SHALL represent dynamic obstacles with JAX arrays for `pos_xy (N, 2)`, `vel_xy (N, 2)`, `radius (N,)`, and `hit/state (N,)`.

#### Scenario: Reset samples P2M defaults
- **WHEN** the obstacle field is reset with default configuration
- **THEN** it SHALL sample `dynamic_obs_num=40`, `pos_xy` in `[-18, 18]^2`, velocity norm in `[1, 5]`, radius in `[0.25, 0.45]`, and obstacle height `4.0`.

### Requirement: P2M Boundary Trace Semantics
`DynamicObstacleField` SHALL reproduce P2M boundary behavior, including the unintuitive `trace_prob` implementation where `random >= trace_prob` selects tracing toward a valid drone position.

#### Scenario: Boundary contact with valid drone
- **WHEN** an obstacle touches the configured x/y boundary and at least one valid drone position is available
- **THEN** the update SHALL use `random >= trace_prob` to choose trace velocity toward a sampled valid drone position and SHALL otherwise use reflected velocity.

#### Scenario: Boundary contact without valid drone
- **WHEN** an obstacle touches the configured x/y boundary and no valid drone position is available
- **THEN** the update SHALL reflect the obstacle velocity.

### Requirement: JAX PRNG Ownership
`DynamicObstacleField` SHALL use explicit JAX PRNG keys for all stochastic choices.

#### Scenario: Deterministic replay from keys
- **WHEN** reset and update are called with the same state, configuration, and PRNG keys
- **THEN** obstacle positions, velocities, radii, and trace choices SHALL be reproducible.

### Requirement: Obstacle Pose for Runtime Systems
`DynamicObstacleField` SHALL expose obstacle cylinder poses derived from `pos_xy`, fixed height, and radius for LiDAR geometry and Rerun visualization.

#### Scenario: Cylinder pose construction
- **WHEN** LiDAR geometry or visualization requests dynamic obstacle poses
- **THEN** each obstacle SHALL be represented as a vertical cylinder centered at `[pos_x, pos_y, dobs_height / 2]`.
