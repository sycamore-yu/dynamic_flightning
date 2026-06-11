## Why

Flightning 当前已有 `bptt.train(env, train_state, ...)` 和 `shac.train(env, actor_state, critic_state, ...)` 这类项目风格算法入口，但还没有 D.VA（Decoupled Visual-Based Analytical Policy Gradient，解耦视觉解析策略梯度）算法模块。D.VA 应作为独立算法加入 `flightning/algos/`，用于支持状态观测和视觉/传感器观测两类任务，而不是嵌入或修改现有 BPTT/SHAC。

参考 D.VA 代码事实：原版 `algorithms/dva.py` 同时支持 `vis_obs=True/False`；环境观测协议包含 `state_obs` 和可选 `vis_obs`；actor 输入会 detach，而 critic 使用 `state_obs`。Flightning 侧应保留项目现有风格，由用户传入 env、actor `TrainState`、critic `TrainState`，并通过 `observation_adapter` 显式定义观测拆分。

## What Changes

- 新增 `flightning/algos/dva.py`，提供 `train(env, actor_state, critic_state, observation_adapter=...)` 风格的 D.VA 训练入口。
- 新增 D.VA 配置结构，覆盖 rollout horizon、discount、critic method、TD-lambda、target critic EMA、critic iterations、batching、gradient clipping、normalization 和 logging。
- 新增 observation adapter 协议，将现有 Flightning `obs: jax.Array` 显式拆分为 actor observation、critic/state observation，以及可选视觉/传感器 observation。
- D.VA actor loss MUST 在算法内部对 actor observation 使用 `jax.lax.stop_gradient`，保留 `action -> differentiable env -> reward` 的梯度路径。
- Critic 默认使用 adapter 提供的 `state_obs`；如果 adapter 未提供 privileged state，则允许回退到 actor observation，以支持现有 state/feature 示例。
- 新增 state-only 和 vision/feature smoke examples，验证 D.VA 能覆盖类似 `examples/train_bptt_state.ipynb` 和 `examples/train_bptt_vision.ipynb` 的两类任务。
- 不修改 `flightning/algos/bptt.py`、`flightning/algos/shac.py` 的公共 API。
- 不把 D.VA 纳入 `p2m-dynamic-avoidance` change；P2M 迁移继续专注 BPTT/SHAC 路线和可微 LiDAR 观测。

## Capabilities

### New Capabilities

- `dva-algorithm`: D.VA 训练算法入口、训练状态、actor/critic 更新、目标 critic、梯度裁剪和指标输出。
- `dva-observation-adapter`: 将 Flightning env observation 映射为 D.VA 所需的 actor observation 与 critic/state observation。
- `dva-examples`: D.VA state-only 与 vision/feature 示例，覆盖现有 Flightning state/vision 风格任务。

### Modified Capabilities

无。该 change 新增独立算法模块，不改变现有 BPTT、SHAC、PPO 或环境 API 的规格级行为。

## Impact

- **新增文件**：
  - `flightning/algos/dva.py` — D.VA 算法实现
  - `examples/train_dva_state.ipynb` 或等价脚本 — state-only D.VA 示例
  - `examples/train_dva_vision.ipynb` 或等价脚本 — vision/feature D.VA 示例
  - `tests/test_dva.py` — D.VA smoke、adapter 和梯度路径测试
- **受影响代码**：
  - `flightning/algos/__init__.py` 需要导出 D.VA 入口
  - `flightning/modules/` 可复用现有 actor/critic 网络，也可新增最小 D.VA 示例网络
- **不影响**：
  - 不修改 `bptt.train`、`shac.train`、`ppo.train`
  - 不改变现有 hovering env 或 wrapper 的行为
  - 不改变 `p2m-dynamic-avoidance` 的 BPTT/SHAC 可微 LiDAR 路线
