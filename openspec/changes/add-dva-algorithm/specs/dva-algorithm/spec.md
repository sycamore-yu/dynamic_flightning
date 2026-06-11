## ADDED Requirements

### Requirement: D.VA 独立算法入口
Flightning SHALL provide D.VA as an independent algorithm module, exposed from `flightning/algos/dva.py`, without changing the public APIs of BPTT, SHAC, or PPO.

#### Scenario: 调用通用 D.VA 训练入口
- **WHEN** a caller invokes `dva.train(env, actor_state, critic_state, observation_adapter=..., config=...)`
- **THEN** the algorithm SHALL use the caller-provided environment and Flax `TrainState` objects instead of constructing environments, networks, or optimizers internally.
- **AND** the return value SHALL include a final runner state and finite training metrics compatible with existing Flightning algorithm conventions.

#### Scenario: 现有算法 API 保持不变
- **WHEN** D.VA is added to `flightning.algos`
- **THEN** `bptt.train`, `shac.train`, and `ppo.train` SHALL remain callable with their existing public signatures and semantics.

### Requirement: D.VA Actor Gradient Semantics
D.VA SHALL stop gradients through actor observations inside the algorithm while preserving gradients from actor parameters through action, differentiable environment dynamics, and reward.

#### Scenario: actor observation 被 stop-gradient
- **WHEN** D.VA computes the actor loss from an environment observation
- **THEN** it SHALL apply `jax.lax.stop_gradient` to the actor observation before actor forward evaluation.
- **AND** gradients SHALL NOT require differentiating through visual, LiDAR, or feature observation construction.

#### Scenario: action 到 reward 的梯度仍保留
- **WHEN** the actor output action is passed to `env.step`
- **THEN** the actor loss SHALL preserve differentiable paths from actor parameters through action-dependent dynamics and rewards where the environment provides them.

### Requirement: D.VA Critic Training
D.VA SHALL train a critic from adapter-provided critic observations and SHALL maintain a target critic for stable bootstrapping.

#### Scenario: critic 使用 state/privileged observation
- **WHEN** the observation adapter returns a `critic_obs`
- **THEN** D.VA SHALL evaluate the critic and target critic on that `critic_obs`.

#### Scenario: critic target 有限
- **WHEN** D.VA computes one-step or TD-lambda critic targets for a small rollout
- **THEN** critic loss, actor loss, and reported metrics SHALL be finite.

### Requirement: D.VA 配置
D.VA SHALL expose a small typed configuration object for rollout horizon behavior, discounting, target critic EMA, critic iterations, batching, gradient clipping, and logging.

#### Scenario: 默认配置可运行
- **WHEN** a caller omits the config argument
- **THEN** D.VA SHALL use defaults suitable for a tiny smoke rollout without requiring project-specific YAML or external configuration files.

#### Scenario: 梯度裁剪可配置
- **WHEN** `max_grad_norm` is set in the D.VA config
- **THEN** actor and critic gradient updates SHALL respect that clipping setting.
