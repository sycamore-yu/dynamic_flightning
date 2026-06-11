## ADDED Requirements

### Requirement: EnvState-Based Visualization
`rerun-debug-visualization` SHALL visualize only Flightning `EnvState` and sensor outputs, not MuJoCo `mjData`, MJCF kinematics, or Rerun-owned simulation state.

#### Scenario: Visualizing a rollout
- **WHEN** a dynamic avoidance rollout is logged
- **THEN** the visualization SHALL derive drone pose, obstacles, walls, target, trajectory, LiDAR rays, hit points, and scan image from Flightning state and sensor outputs.

### Requirement: Debug-Only Semantics
Rerun visualization SHALL be debug-only and SHALL NOT define physics, collision, reward, LiDAR collision geometry, or training state.

#### Scenario: Visualization disabled
- **WHEN** Rerun logging is disabled
- **THEN** training dynamics, reward, termination, and LiDAR computation SHALL remain unchanged.

### Requirement: Headless Export
The first version SHALL provide a headless Rerun export path for remote server workflows.

#### Scenario: Export on headless server
- **WHEN** a user runs the dynamic avoidance visualization export without a viewer
- **THEN** the system SHALL write a `.rrd` recording that can be opened offline.

### Requirement: Smoke Test Without Viewer
Visualization tests SHALL NOT require launching a graphical Rerun viewer.

#### Scenario: CI or remote test
- **WHEN** the Rerun visualization smoke test runs
- **THEN** it SHALL verify logging/export calls without requiring an interactive display.
