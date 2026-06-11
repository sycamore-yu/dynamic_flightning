## ADDED Requirements

### Requirement: Flightning Env API
`DynamicAvoidanceEnv` SHALL implement the existing Flightning `Env` API with JAX pytree state, `reset(key, state=None)`, `_step(state, action, key)`, `step(state, action, key)`, `action_space`, and `observation_space`.

#### Scenario: Reset returns state and observation
- **WHEN** `DynamicAvoidanceEnv.reset(key)` is called
- **THEN** it SHALL return a `DynamicAvoidanceEnvState` pytree and one flat `jax.Array` observation compatible with `LogWrapper`, `VecEnv`, BPTT, and SHAC.

#### Scenario: Step runs inside scan
- **WHEN** `_step` is called inside `jax.lax.scan`
- **THEN** it SHALL update environment state without Python-side mutable runtime state, MuJoCo `mjData`, or MuJoCo stepping.

### Requirement: Runtime State Ownership
`DynamicAvoidanceEnvState` SHALL be the runtime source of truth for drone state, target state, dynamic obstacle state, wall state, last actions, time, and step index.

#### Scenario: Runtime state does not depend on MuJoCo
- **WHEN** the environment computes dynamics, reward, termination, observation, or LiDAR poses during rollout
- **THEN** it SHALL use Flightning JAX state and SHALL NOT read or write MuJoCo `mjData`.

### Requirement: Flightning Action Semantics
`DynamicAvoidanceEnv` SHALL use Flightning low-level action semantics: a 4D action containing collective thrust and 3D body rates.

#### Scenario: Action is clipped and applied
- **WHEN** `_step` receives an action outside `action_space`
- **THEN** it SHALL clip the action to the environment action bounds before updating `last_actions` and quadrotor dynamics.

#### Scenario: P2M acceleration controller excluded
- **WHEN** P2M reference behavior uses 3D acceleration commands
- **THEN** the first version SHALL NOT migrate P2M's acceleration controller and SHALL NOT expose a 3D acceleration action space.

### Requirement: Differentiable Training Reward
`DynamicAvoidanceEnv` SHALL provide a default differentiable surrogate reward for BPTT/SHAC training, separate from exact P2M reward alignment metrics.

#### Scenario: Default reward is smooth training signal
- **WHEN** `_step` computes the default training reward
- **THEN** it SHALL combine differentiable goal progress, speed band, height band, soft clearance, dynamic obstacle risk, action magnitude, and action smoothness terms.

#### Scenario: P2M reward is not default BPTT objective
- **WHEN** P2M reward components are computed for alignment or logging
- **THEN** they SHALL NOT replace the default differentiable training reward unless explicitly configured as an evaluation or compatibility mode.

### Requirement: Termination and Truncation
`DynamicAvoidanceEnv` SHALL compute termination and truncation as episode management signals while preserving continuous reward terms before failure boundaries.

#### Scenario: Safety violation terminates episode
- **WHEN** the drone violates wall or bounds limits, height limits, excessive velocity limits, obstacle proximity limits, NaN state checks, or time horizon
- **THEN** the environment SHALL return `terminated` or `truncated` according to the violation type.
