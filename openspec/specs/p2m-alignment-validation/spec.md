# p2m-alignment-validation Specification

## Purpose
TBD - created by archiving change p2m-dynamic-avoidance. Update Purpose after archive.
## Requirements
### Requirement: P2M Alignment Layer
`p2m-alignment-validation` SHALL be a validation and evaluation layer, not a training-runtime ownership layer.

#### Scenario: Alignment tests run
- **WHEN** alignment tests compare Flightning outputs to P2M references
- **THEN** they SHALL use fixed states and deterministic inputs to compare outputs without changing training runtime semantics.

### Requirement: Dynamic Obstacle Alignment
The validation suite SHALL test P2M-compatible dynamic obstacle initialization and update behavior.

#### Scenario: Trace probability edge case
- **WHEN** an obstacle touches a boundary under a fixed random value and valid drone position
- **THEN** the test SHALL verify P2M's `random >= trace_prob` trace-selection behavior.

### Requirement: LiDAR Scan Alignment
The validation suite SHALL test the `p2m_oversample` LiDAR postprocessing pipeline against P2M reference behavior.

#### Scenario: Fixed geometry scan
- **WHEN** fixed drone, wall, and obstacle geometry are provided
- **THEN** the test SHALL compare the resulting `(1, 36, 6)` distance image against a P2M-derived reference within documented tolerance.

### Requirement: P2M Reward Evaluation Metrics
The validation suite SHALL include exact or near-exact P2M reward component calculations as evaluation metrics and logs.

#### Scenario: P2M reward logged
- **WHEN** a rollout is evaluated for P2M alignment
- **THEN** P2M reward components SHALL be computed and logged separately from the default differentiable training reward.

### Requirement: Training Smoke Validation
The validation suite SHALL include BPTT and SHAC smoke checks for the dynamic avoidance environment.

#### Scenario: BPTT and SHAC smoke
- **WHEN** the smoke tests run with a small rollout
- **THEN** BPTT and SHAC SHALL compile, step the environment, compute loss/reward, and produce finite metrics without requiring D.VA.

