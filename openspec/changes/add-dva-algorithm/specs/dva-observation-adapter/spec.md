## ADDED Requirements

### Requirement: D.VA Observation Adapter Boundary
D.VA SHALL use a caller-provided observation adapter to map a Flightning environment observation into actor and critic inputs.

#### Scenario: 默认 adapter 支持 state-only 环境
- **WHEN** the environment returns a flat `jax.Array` observation and no custom adapter is provided
- **THEN** D.VA SHALL treat that observation as both `actor_obs` and `critic_obs`.

#### Scenario: 自定义 adapter 拆分 actor/critic 输入
- **WHEN** a custom adapter receives an environment observation
- **THEN** it SHALL return a structured D.VA observation containing `actor_obs` and `critic_obs`.
- **AND** the adapter SHALL define only observation mapping, not environment state semantics, reward semantics, or dynamics execution.

### Requirement: Adapter JAX Compatibility
D.VA observation adapters SHALL be compatible with JAX transformations used by the training loop.

#### Scenario: adapter 在 rollout 中运行
- **WHEN** D.VA calls the adapter during reset, rollout, actor update, or critic update
- **THEN** the adapter output SHALL be a pytree of JAX arrays that can be used under `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

### Requirement: Critic Observation Fallback
D.VA SHALL allow the critic observation to fall back to the actor observation when a task has no privileged state observation.

#### Scenario: 无 privileged state
- **WHEN** a state-only or feature-only task does not provide a separate `critic_obs`
- **THEN** D.VA SHALL remain usable by evaluating actor and critic on the same adapted observation.

### Requirement: Observation Ownership Separation
The observation adapter SHALL be the ownership boundary between task-specific observation layout and generic D.VA algorithm mechanics.

#### Scenario: LiDAR/vision layout 不进入算法核心
- **WHEN** a LiDAR, vision, or feature task needs channel reshaping, slicing, or concatenation
- **THEN** that mapping SHALL live in the adapter or example/task module, not in generic `dva.train`.
