# Issue: 抽取 `flightning/algos/_common.py` 并对齐 bptt/ppo/shac 算法 API

## Parent

Sibling of `001-shac-jax-migration.md`（SHAC 迁移）。

## What to build

为 `flightning.algos` 建立共享基础设施层，并统一三个算法（bptt / ppo / shac）的对外 API，使调用者可以用一致的方式导入、配置、运行训练。

### 背景与动机

SHAC 迁移完成后，三个算法共享若干通用 building block（梯度裁剪、TD-λ 目标、EMA 更新、rollout 收集），但当时这些逻辑要么藏在 `shac.py` 内部作为私有函数，要么在各算法中重复实现（progress callback 写法、metrics 返回格式、`train()` 签名中 `config` 的位置）。本 issue 集中解决这些对称性债务。

### 范围

#### 1. 新建 `flightning/algos/_common.py`

算法无关的 RL 基础设施，全部公开命名（无 leading underscore）：

- `clip_grads(grads, max_norm)` — pytree L2 全局梯度裁剪。
- `td_lambda_targets(rewards, dones, next_values, gamma, lam)` — 前向 Ai/Bi 扫描计算 TD-λ 目标。
- `ema_update(params, target_params, alpha)` — `target = alpha*target + (1-alpha)*params`，target network 通用。
- `get_rollouts(env, policy, num_rollouts, key)` — `jax.vmap(rollout)` 包装，签名 `(obs, key) -> action`。

#### 2. `shac.py` 改用共享 helper

- 删除本地 `_clip_grads` / `td_lambda_targets` 定义。
- 从 `_common` 导入 `clip_grads` / `ema_update` / `td_lambda_targets`，并用 `ema_update` 替换内联 `tree_map` 的 target critic 更新。

#### 3. `flightning/algos/__init__.py` 统一导出

```python
from .bptt import train as train_bptt, Config as BPTTConfig
from .ppo  import train as train_ppo,  Config as PPOConfig
from .shac import train as train_shac, Config as SHACConfig
from ._common import (clip_grads, ema_update,
                      get_rollouts, td_lambda_targets)
```

用户一行即可拿到算法、Config、共享工具。

#### 4. `train()` 签名对齐

三个算法统一为：

```
train(env, train_state[s], ..., key, config: Config = Config())
```

- `bptt.train`：新增 `Config`（最小字段 `logging` / `logging_freq`），`config` 作为带默认值的 keyword arg。
- `ppo.train`：`config` 从 required positional 改为 `config: Config = Config()`，与 bptt/shac 一致。
- `shac.train`：签名已对齐（保持）。

#### 5. metrics 返回格式统一

所有算法 `train()` 返回 `{"runner_state": ..., "loss": ..., "metrics": ...}`：

- `bptt`：新增 `"loss"` 键，保留 `"metrics"` 作为向后兼容别名。
- `ppo`：当前返回 `metric = samples.info`，保持 `"metrics"` 键，必要时补 `"loss"`。
- `shac`：保持 `"metrics": {"actor_loss": ...}`；调用者按算法查对应键。

#### 6. notebook 改用共享 `get_rollouts`

`train_shac_state.ipynb` / `train_shac_vision.ipynb` 中 inline 的 `def get_rollouts(...)` 与 `jax.vmap(rollout, ...)` 调用替换为 `from flightning.algos import get_rollouts`。后续可在 bptt notebook 中做同样替换（不在本 issue 必需范围，但优先推荐）。

## Acceptance criteria

- [ ] `flightning/algos/_common.py` 存在并导出 `clip_grads` / `ema_update` / `get_rollouts` / `td_lambda_targets`。
- [ ] `flightning/algos/shac.py` 不再定义本地 `_clip_grads` 或 `td_lambda_targets`；target critic 更新使用 `ema_update`。
- [ ] `from flightning.algos import train_bptt, train_ppo, train_shac, BPTTConfig, PPOConfig, SHACConfig` 全部可解析。
- [ ] `bptt.train()` 和 `ppo.train()` 在不传 `config` 参数时可正常调用（使用默认 `Config()`）。
- [ ] `bptt.train()` 返回值 dict 包含 `"loss"` 键（`"metrics"` 作为别名保留）。
- [ ] `examples/train_shac_state.ipynb` 与 `examples/train_shac_vision.ipynb` 的评估 cell 使用 `from flightning.algos import get_rollouts`，不再有 inline `jax.vmap(rollout, ...)` 或本地 `def get_rollouts`。
- [ ] `python flightning/algos/shac.py`（`__main__` smoke test）仍然通过。

## Blocked by

- `001-shac-jax-migration.md` — SHAC 算法与 notebook 必须先就位，本 issue 在其上做 API 层统一。

## 关键设计决策（已锁定）

1. **`_common.py` 命名**：以 leading underscore 表示"algos 内部共享"，但内部符号（`clip_grads` 等）无 underscore、通过 `__init__.py` 公开。
2. **`config=Config()` 关键字参数**：所有算法统一；调用者可省略或传自定义 Config。
3. **保留 `metrics` 键做向后兼容**：bptt 已有 `res_dict["metrics"]` 用法（如 notebook 绘图），新增 `"loss"` 不删旧键。
4. **`get_rollouts` 位置**：放 `_common.py` 而非 `flightning/envs/env_base.py`，因为它是 RL 训练辅助而非 env 核心 API。
5. **不重写 `ActorCriticPPO` / `SHACActor` 来统一"网络前向返回约定"**——三种风格（raw tensor / `PiValue` / `SHACActionSample`）短期容忍；后续可考虑 `PolicyOutput` 协议（不在本 issue 范围）。

## 后续工作（不在本 issue 范围）

- `ppo.train()` 返回值 dict 补 `"loss"` 键（如需要）。
- bptt/ppo notebook 同样替换为 `from flightning.algos import get_rollouts`。
- 抽取 epoch/minibatch 扫描骨架为 `training_loop` helper。
