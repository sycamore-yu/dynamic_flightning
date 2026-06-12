## ADDED Requirements

### Requirement: Dynamic Avoidance Training Uses D.VA Validation
Dynamic avoidance full training acceptance SHALL use a D.VA-based headless validation path instead of relying on legacy notebook execution.

#### Scenario: Headless D.VA validation command succeeds
- **WHEN** dynamic avoidance training acceptance is evaluated on a remote headless server
- **THEN** the validation SHALL run from script or test commands in the `flightning` conda environment.
- **AND** it SHALL report finite D.VA actor and critic metrics.

#### Scenario: Notebook execution is not required
- **WHEN** dynamic avoidance training acceptance is evaluated
- **THEN** notebook execution SHALL NOT be required as the primary acceptance path.

#### Scenario: Migration reuses generic D.VA capability
- **WHEN** the migration is implemented
- **THEN** it SHALL reuse the generic D.VA algorithm and observation adapter contract from `add-dva-algorithm` rather than creating a dynamic-avoidance-only algorithm fork.

### Requirement: Dynamic Avoidance D.VA Uses Privileged Critic Observation
Dynamic avoidance D.VA validation SHALL use a critic observation derived from training-only privileged environment state rather than falling back to the actor-visible LiDAR observation.

#### Scenario: LiDAR actor observation has privileged critic observation
- **WHEN** D.VA trains on `DynamicAvoidanceEnv` with actor-visible LiDAR observations
- **THEN** the actor observation SHALL remain the actor-visible dynamic avoidance observation.
- **AND** the critic observation SHALL be derived from `DynamicAvoidanceEnvState` as a deterministic, scaled privileged state observation.

#### Scenario: Missing privileged critic observation fails loudly
- **WHEN** a visual or LiDAR D.VA training path cannot provide a privileged critic observation
- **THEN** validation SHALL fail with an explicit error instead of silently evaluating the critic on actor observations.

#### Scenario: State-only D.VA is not treated as fallback
- **WHEN** D.VA trains on a state-only task where actor and critic intentionally use the same state observation
- **THEN** that SHALL be treated as state-only D.VA, not as a visual/LiDAR privileged-state fallback.

### Requirement: Dynamic Avoidance D.VA Handles Done Bootstrap Correctly
Dynamic avoidance D.VA validation SHALL avoid using auto-reset observations as terminal bootstrap observations.

#### Scenario: Early termination does not bootstrap from reset observation
- **WHEN** a dynamic avoidance episode terminates before the time limit
- **THEN** D.VA critic targets SHALL bootstrap with zero for that terminated environment.
- **AND** it SHALL NOT use the reset observation from the next episode as the terminal bootstrap value.

#### Scenario: Time-limit truncation uses pre-reset terminal critic observation
- **WHEN** a dynamic avoidance episode reaches the configured time limit
- **THEN** D.VA critic targets MAY bootstrap from the pre-reset terminal privileged critic observation.
- **AND** it SHALL NOT use the auto-reset observation from the next episode as the terminal bootstrap value.

### Requirement: Dynamic Avoidance D.VA Declares Optimization and Scaling Policy
Dynamic avoidance D.VA validation SHALL declare the optimization and observation scaling choices used by the headless script.

#### Scenario: Optimizer policy is explicit
- **WHEN** the dynamic avoidance D.VA validation script constructs actor and critic train states
- **THEN** it SHALL explicitly define actor and critic optimizer policies, including whether fixed learning rates or Optax schedules are used.

#### Scenario: Privileged critic observation uses deterministic scaling
- **WHEN** privileged critic observations are constructed from `DynamicAvoidanceEnvState`
- **THEN** their components SHALL use deterministic scaling based on environment or action bounds.
- **AND** validation SHALL NOT claim long-training convergence from finite smoke metrics alone.

### Requirement: Dynamic Avoidance Training Reward Uses Smooth Differentiable Proxies
Dynamic avoidance D.VA validation SHALL use a default training reward that separates hard episode events from differentiable LiDAR clearance and object-free motion/TTC learning signals.

#### Scenario: Smooth safety proxy replaces hard-margin-only obstacle penalty
- **WHEN** `_get_reward_jit` computes the default dynamic avoidance training reward
- **THEN** LiDAR clearance terms SHALL remain the current geometry safety signal for nearest raycast hits.
- **AND** clearance terms SHALL use smooth differentiable proxy penalties.
- **AND** they SHALL provide an anticipatory avoidance signal outside the immediate collision margin.
- **AND** they SHOULD use numerically stable primitives such as `jax.nn.softplus` rather than hand-written `log(exp(x) + 1)` expressions.

#### Scenario: Hard termination is not the differentiable collision proxy
- **WHEN** dynamic avoidance collision, bounds, height, velocity, or NaN predicates are computed
- **THEN** those predicates SHALL be used for termination and bootstrap semantics.
- **AND** they SHALL NOT replace continuous clearance and object-free motion/TTC reward proxies as the primary differentiable collision signal.
- **AND** any fixed event penalty included in reward SHALL use a stopped-gradient event indicator.

#### Scenario: Object-free motion and TTC proxy replaces object-level dobs risk
- **WHEN** the default training reward computes dynamic clutter risk
- **THEN** it SHALL remove object-level `dobs_risk` from the main validation reward.
- **AND** it SHALL compute `motion_risk` and/or `ttc_risk` from object-free clearance motion fields such as sector-wise `delta_d = (d_t - d_{t-1}) / dt`.
- **AND** if TTC is used, it SHALL compute a clipped and normalized sector-wise value such as `ttc = d_t / max(-delta_d, eps)`.
- **AND** the main validation reward SHALL NOT depend on dynamic obstacle object identity, object ordering, object radius fields, or object velocity labels.

#### Scenario: Reward proxy is numerically stable
- **WHEN** reward proxy calculations divide by norms, use clipping, or use inverse-trigonometric fallbacks
- **THEN** they SHALL guard denominators with epsilons and keep inverse-trigonometric inputs inside stable valid domains if such fallbacks are used.
- **AND** focused tests SHALL assert finite reward values and finite gradients in representative near-collision and safe-clearance scenes.
