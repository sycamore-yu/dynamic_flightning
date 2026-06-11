## ADDED Requirements

### Requirement: CNN LiDAR Actor
`cnn-lidar-policy` SHALL provide a Flax/JAX policy module that accepts the default flat observation layout and outputs a 4D Flightning low-level action.

#### Scenario: Policy output shape
- **WHEN** the policy receives a default dynamic avoidance observation
- **THEN** it SHALL output `[collective_thrust, body_rate_x, body_rate_y, body_rate_z]` with shape `(4,)`.

### Requirement: LiDAR Feature Encoding
The policy SHALL reshape the LiDAR slice into a channel-first `(1, 36, 6)` image before convolutional encoding.

#### Scenario: Observation split
- **WHEN** the policy processes the default observation
- **THEN** it SHALL split the first 216 values as LiDAR image data and the remaining values as state features.

### Requirement: State Feature Fusion
The policy SHALL fuse LiDAR CNN features with target direction, velocity, and last action features before the action head.

#### Scenario: State features present
- **WHEN** target direction, velocity, and last action are present in the observation
- **THEN** the policy SHALL concatenate their encoded features with LiDAR features before producing an action.

### Requirement: No DVA Gradient Semantics
The policy SHALL NOT implement D.VA-specific actor-observation stop-gradient semantics.

#### Scenario: Used with BPTT or SHAC
- **WHEN** the policy is trained with BPTT or SHAC
- **THEN** the policy SHALL allow normal JAX gradient flow according to the environment and sensor gradient configuration.
