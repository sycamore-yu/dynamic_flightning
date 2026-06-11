## Context

D.VA（Decoupled Visual-Based Analytical Policy Gradient，解耦视觉解析策略梯度）参考实现位于 `/home/tong/tongworkspace/paperworkspace/reference/D.VA`。已核实的关键事实如下：

- 原版 `algorithms/dva.py` 是独立算法类，支持 `vis_obs=True/False` 两类训练。
- 原版环境返回固定观测协议：`{"state_obs": ..., "vis_obs": ...}`，其中 `vis_obs` 可选。
- 原版 actor 可在 `vis_obs=True` 时用视觉 encoder 编码 `vis_obs`，在 `vis_obs=False` 时直接使用 `state_obs`。
- 原版 critic 始终使用 `state_obs`。
- 原版 D.VA actor loss 在 actor 输入处 detach：`vis_obs.detach().clone()` 或 `state_obs.detach()`，但动作仍进入可微环境，因此梯度路径保留为 `actor parameters -> action -> differentiable dynamics -> reward`。

Flightning 当前算法风格不同：`bptt.train` 和 `shac.train` 由调用方传入 env、Flax `TrainState` 和配置，算法内部只包装 `LogWrapper` 与 `VecEnv`。因此 Flightning 的 D.VA 不应照搬 `DVA(cfg)` 内部创建 env/network 的方式，而应采用项目风格 API。

## Goals / Non-Goals

**Goals:**

- 新增通用 D.VA 算法模块 `flightning/algos/dva.py`。
- API 采用 `train(env, actor_state, critic_state, observation_adapter=..., config=...)` 风格，与 BPTT/SHAC 一致。
- 支持 state-only 任务和视觉/传感器任务。
- 支持 adapter 从当前 Flightning flat observation 中提取 actor observation 和 critic/state observation。
- 在 D.VA 算法内部实现 actor observation 的 `stop_gradient`，不修改 policy 模块和现有 BPTT/SHAC。
- 提供目标 critic、critic 训练、one-step 或 TD-lambda target、梯度裁剪和基础训练指标。

**Non-Goals:**

- 不修改 `bptt.py`、`shac.py`、`ppo.py` 的公共 API。
- 不要求所有 Flightning env 立即返回 dict observation。
- 不在本 change 中迁移动动态避障完整训练验收；该迁移另开 OpenSpec change。
- 不实现 D.VA 原仓库的 Torch/DFlex/YAML 内部建模框架。

## API Design

```python
class DVAConfig(NamedTuple):
    gamma: float = 0.99
    lam: float = 0.95
    critic_method: str = "td-lambda"
    target_critic_alpha: float = 0.4
    critic_iterations: int = 16
    num_batches: int = 4
    max_grad_norm: float = 1.0
    logging_freq: int = 10
    logging: bool = True

class DVAObservation(NamedTuple):
    actor_obs: jax.Array
    critic_obs: jax.Array

def default_observation_adapter(obs: jax.Array) -> DVAObservation:
    return DVAObservation(actor_obs=obs, critic_obs=obs)

def train(
    env: Env,
    actor_state: TrainState,
    critic_state: TrainState,
    *,
    observation_adapter: Callable[[jax.Array], DVAObservation] = default_observation_adapter,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 50,
    num_envs: int = 64,
    key: chex.PRNGKey = jax.random.key(0),
    config: DVAConfig = DVAConfig(),
):
    ...
```

`observation_adapter`（观测适配器）是 D.VA 与具体 env 的边界。它只定义“如何从 env.obs 得到 actor/critic 输入”，不改变 env 状态语义。对于现有 state-only env，默认 adapter 即可；对于 LiDAR/vision env，adapter 可以把 flat observation 拆成 channel-first LiDAR image 和 state feature，或者拼接成 policy 已接受的 actor input。

## Gradient Semantics

D.VA actor 更新的核心语义：

1. 从 env 得到 `obs`。
2. 通过 `observation_adapter(obs)` 得到 `actor_obs` 与 `critic_obs`。
3. actor 使用 `jax.lax.stop_gradient(actor_obs)` 产生 action。
4. action 输入可微 env step。
5. reward 通过可微 dynamics 对 action 和 actor 参数反传。
6. critic 使用 `critic_obs` 学习 value target。

这与 BPTT/SHAC 的完整 analytical policy gradient 不同。BPTT/SHAC 继续允许观测模型梯度穿过 sensor/renderer 回到状态；D.VA 则在算法内部切断 actor observation 输入路径，以降低视觉/传感器链路导致的梯度噪声。

## Architecture

| Layer | Upstream Caller | Owned Responsibility | Module Location | Explicit Exclusions |
|-------|-----------------|----------------------|-----------------|---------------------|
| `dva.train` | examples/tests | D.VA rollout、actor loss、critic update、target critic update、metrics | `flightning/algos/dva.py` | 不创建 env/network；不修改 BPTT/SHAC |
| `DVAObservation` | `dva.train` | 承载 actor observation 与 critic observation | `flightning/algos/dva.py` | 不定义 env state |
| `observation_adapter` | caller-provided | 将 env obs 映射为 D.VA 输入 | examples 或 task module | 不执行 dynamics；不计算 reward |
| actor network | caller-provided `TrainState` | 根据 stop-gradient 后的 actor obs 输出 action | `flightning/modules/*` | 不决定 D.VA 梯度路径 |
| critic network | caller-provided `TrainState` | 根据 critic obs 估计 value | `flightning/modules/*` | 不读取 visual obs，除非 adapter 明确选择 |

## Decisions

### ADR-001: Implement D.VA as an Independent Algorithm

**Status:** Accepted

**Decision:** D.VA lives in `flightning/algos/dva.py` and exposes a `train(...)` function parallel to BPTT/SHAC.

**Rationale:** The reference D.VA implementation is a standalone algorithm, and the user confirmed it should not be embedded into BPTT/SHAC.

### ADR-002: Use Project-Style API Instead of `DVA(cfg)`

**Status:** Accepted

**Decision:** Flightning D.VA receives env, actor `TrainState`, critic `TrainState`, config, and an observation adapter from the caller.

**Rationale:** Existing Flightning algorithms use caller-owned env/network construction. Preserving this avoids introducing a second YAML-driven training framework.

### ADR-003: Use Adapter-Based Observation Splitting

**Status:** Accepted

**Decision:** D.VA does not require envs to return dict observations in the first version. It consumes the existing env observation and delegates splitting to `observation_adapter`.

**Rationale:** Current Flightning examples use flat `jax.Array` observations and `observation_space.shape[0]`. Adapter-based splitting supports existing tasks and future LiDAR/vision tasks without changing wrapper contracts.

### ADR-004: Stop Actor Observation Gradient Inside D.VA

**Status:** Accepted

**Decision:** D.VA applies `jax.lax.stop_gradient` to actor observation before actor forward.

**Rationale:** This matches the reference D.VA code behavior where `vis_obs` or `state_obs` is detached before actor forward. The gradient path through action, differentiable dynamics, and reward remains active.

### ADR-005: Keep Dynamic Avoidance Acceptance Migration Separate

**Status:** Accepted

**Decision:** `add-dva-algorithm` provides the generic D.VA algorithm only. Full dynamic avoidance training acceptance migration will be handled by a later change, recommended name `replace-avoidance-training-validation-with-dva`.

**Rationale:** This keeps algorithm ownership separate from P2M/dynamic-avoidance validation ownership. Existing dynamic obstacle, LiDAR, and reward alignment tests remain the numerical alignment layer; training success criteria can move to D.VA in a focused follow-up change.

### ADR-006: Prefer Headless Scripts for Acceptance

**Status:** Accepted

**Decision:** D.VA examples are script-first. Notebooks may exist as optional explanatory artifacts, but scripts and automated tests are the required validation path.

**Rationale:** The project is commonly validated on a remote headless server, and prior notebook-based training acceptance was blocked by resource requirements.

## Test Strategy

- State-only smoke test: train D.VA for a tiny rollout on `HoveringStateEnv` or an equivalent lightweight env with default adapter, asserting finite metrics and finite gradients.
- Vision/feature smoke test: train D.VA on `HoveringFeaturesEnv` or equivalent feature observation with adapter, asserting actor observation stop-gradient behavior and critic update.
- Adapter test: verify custom adapter returns expected actor/critic shapes and works under `jax.jit`/`jax.vmap`.
- Gradient path test: verify actor gradients are finite and nonzero through `action -> env.step -> reward`, while actor input observation gradients are stopped.
- Headless script validation: run state-only and feature/vision-style D.VA scripts in the `flightning` conda environment before presenting them as examples.

## Relationship to P2M Dynamic Avoidance

`p2m-dynamic-avoidance` remains focused on BPTT/SHAC with differentiable MuJoCo-LiDAR observations. It should allow LiDAR gradients through `MjLidarJax` into `EnvState` for analytic sensor-gradient experiments. D.VA is tracked separately by this change and can later be applied to the dynamic avoidance env after both changes mature.
