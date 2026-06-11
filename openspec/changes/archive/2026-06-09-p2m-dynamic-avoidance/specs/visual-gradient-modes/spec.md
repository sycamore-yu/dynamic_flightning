## ADDED Requirements

### Requirement: Analytic LiDAR Gradient Default
`analytic_lidar_grad` SHALL be the default mode for this change and SHALL allow gradients from LiDAR observation computations through `MjLidarJax` and sensor geometry into Flightning `EnvState` where JAX defines derivatives.

#### Scenario: BPTT LiDAR gradient smoke
- **WHEN** a BPTT smoke loss depends on the LiDAR observation
- **THEN** gradients with respect to policy parameters SHALL be finite and nonzero, and a targeted sensor-state gradient check SHALL compile successfully.

### Requirement: Stop LiDAR Gradient Control
`stop_lidar_grad` SHALL be an explicit stability or ablation mode that applies `jax.lax.stop_gradient` to LiDAR observation output or selected sensor outputs.

#### Scenario: Stop mode enabled
- **WHEN** `stop_lidar_grad` is configured
- **THEN** policy gradients through dynamics and non-LiDAR reward terms SHALL remain available, while LiDAR sensor-to-state gradients SHALL be blocked.

### Requirement: DVA Exclusion
This change SHALL NOT implement D.VA algorithm semantics.

#### Scenario: DVA requested
- **WHEN** D.VA training behavior such as actor observation stop-gradient is needed
- **THEN** it SHALL be handled by the separate `add-dva-algorithm` OpenSpec change, not by `p2m-dynamic-avoidance`.

### Requirement: Piecewise Differentiability Disclosure
The sensor gradient mode documentation SHALL state that raycasting through geometry intersections, nearest-hit selection, min-pooling, and hit/no-hit branches is piecewise differentiable and may be discontinuous at visibility or contact boundaries.

#### Scenario: Gradient mode documented
- **WHEN** users inspect the dynamic avoidance sensor configuration
- **THEN** they SHALL see that analytic LiDAR gradients are JAX-traceable but not globally smooth.
