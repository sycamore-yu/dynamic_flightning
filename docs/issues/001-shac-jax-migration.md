# Issue: 迁移 SHAC 算法到 JAX 并提供 state/vision 训练 notebook

## Parent

无（本 issue 为首个顶层条目）。

## What to build

将 NVIDIA DiffRL（PyTorch）的 SHAC（Short-Horizon Actor Critic）算法移植到 `rpg_flightning` 仓库，以 JAX 重写，使其与现有 `bptt` / `ppo` 在生态、命名、调用方式上保持一致；并提供两个端到端 Jupyter notebook（state 版与 vision 版），流程与 `train_bptt_state.ipynb` / `train_bptt_vision.ipynb` 对齐。

### 背景与动机

- `flightning` 已具备 `bptt`（back-prop through time）和 `ppo` 两个算法，全部基于 JAX/Flax/optax 生态；环境仿真（`HoveringStateEnv` / `HoveringFeaturesEnv`）本身也是 JAX 可微的。
- SHAC 的核心思想——"通过短视野 rollout 的可微仿真反传梯度到 actor"——在 JAX 下可用 `jax.value_and_grad` 直接表达，比 PyTorch 版本更贴合本项目约定。
- 验收标准明确：用户应能像运行 `train_bptt_*.ipynb` 一样运行 `train_shac_*.ipynb`，无需切换框架栈。

### 范围

#### 1. 算法实现：`flightning/algos/shac.py`

JAX 版 SHAC，签名与 `bptt.train` / `ppo.train` 对齐：

```
train(env, actor_state, critic_state=None,
      num_epochs, num_steps_per_epoch, num_envs,
      key, config=Config()) -> {"runner_state": ..., "metrics": ...}
```

核心组件：

- **Actor rollout**：`jax.lax.scan` 驱动；每步通过 `SHACActor.sample_action`（vmap 化）采样动作；梯度穿透 `env.step` 回传到 actor 参数。
- **Actor loss**：前向累加折扣回报，遇 `done` 重置；在 horizon 末端用 `target_critic(next_obs)` 做 terminal bootstrap。
- **Critic update**：在 rollout buffer 上做 TD-λ 目标回归，使用固定 size 的 minibatch scan（避免 JAX tracer concretization 错误）。
- **Target critic**：EMA 更新（`alpha * target + (1-alpha) * critic`）稳定 bootstrap。
- **环境 wrap**：`LogWrapper` + `VecEnv` 在 `train()` 内部应用。

#### 2. 网络定义：`flightning/modules/mlp.py`

在 `ActorCriticPPO` 旁边新增两个类：

- **`SHACActor`**：高斯策略。`@compact` 的 `__call__` 构建 dense trunk（auto-named `Dense_0/Dense_1/...`，与 vision 预训练 predictor 命名一致以支持参数拷贝）；`setup()` 声明可学 `log_std`；`sample_action(obs, key, deterministic)` 返回 `SHACActionSample(action, mean, std)`。
- **`SHACCritic`**：标量值头。`setup()` 构建层；`__call__` 输出沿最后一轴 squeeze 为 `(num_envs,)`。

`flightning/modules/__init__.py` 同步导出 `SHACActor` / `SHACCritic` / `SHACActionSample`。

#### 3. 两个端到端 notebook

- **`examples/train_shac_state.ipynb`**：镜像 `train_bptt_state.ipynb`。使用 `MinMaxObservationWrapper`、`SHACActor` + `SHACCritic`、分离的 `TrainState`（actor/critic），最后用 `get_rollouts` 评估并 `env_eval.plot_trajectories`。
- **`examples/train_shac_vision.ipynb`**：镜像 `train_bptt_vision.ipynb`。保留"先做特征→状态监督预训练，再拷贝 `Dense_0`/`Dense_1` 到 actor"的两阶段流程，然后把训练切换到 SHAC。

#### 4. 不引入 PyTorch 依赖

原始 DiffRL 的 PyTorch 辅助模块（`utils/` / `models/`）不应被原样拷贝。所有算法逻辑、网络、工具函数用 JAX 重写并放入 `flightning` 已有目录。

## Acceptance criteria

- [ ] `flightning/algos/shac.py` 实现 `train(env, actor_state, critic_state=None, ..., config=Config())`，签名字段顺序与 `bptt.train` / `ppo.train` 一致。
- [ ] `SHACActor` / `SHACCritic` / `SHACActionSample` 在 `flightning/modules/mlp.py` 中定义并从 `flightning/modules/__init__.py` 导出。
- [ ] `python flightning/algos/shac.py`（`__main__` smoke test）在 `HoveringStateEnv` 上能跑通至少 2 个 epoch 并打印 `actor_loss`。
- [ ] `examples/train_shac_state.ipynb` 能端到端运行：环境构建 → 网络初始化 → SHAC 训练 → actor loss 绘图 → 评估 rollout → `plot_trajectories`。
- [ ] `examples/train_shac_vision.ipynb` 能端到端运行：特征→状态预训练 → `Dense_0`/`Dense_1` 拷贝到 actor → SHAC 训练 → 评估 rollout。
- [ ] `SHACActor` 的参数命名（`Dense_0`/`Dense_1`/`Dense_2` + `log_std`）与 vision notebook 中的 `MLP` predictor 命名一致，使得 `actor_params['params']['Dense_0'] = predictor_params['params']['Dense_0']` 可直接工作。
- [ ] 不引入任何 PyTorch 导入（`torch` / `tensorboardX` / `rl_games` 等）。
- [ ] 算法核心行为与 DiffRL 参考实现对齐：actor 用短视野 rollout 累加折扣回报作为 loss，critic 在 buffer 上做 TD-λ 回归，target critic 用 EMA 更新。

## Blocked by

None — 可立即开始。

## 关键设计决策（已锁定）

1. **JAX 重写**而非 PyTorch 迁移——匹配 `flightning` 生态，避免双栈维护。
2. **网络定义放 `flightning/modules/mlp.py`**——与 `ActorCriticPPO` 同位置，遵守项目边界划分（网络在 modules，算法在 algos）。
3. **`SHACActor` 使用 `@compact` + `setup()` 混合**：trunk 用 `@compact`（auto-named 支持预训练参数拷贝），`log_std` 在 `setup()` 声明供非 compact 的 `sample_action` 读取。
4. **`action_log_std` 不作为独立字段**——封装进 `SHACActor.params['log_std']`，由 `TrainState` 统一管理。
5. **不继承 `MLP`**——Flax 单 `@compact` 限制与 `sample_action` 需求冲突，~10 行 trunk 重复容忍。

## 后续工作（不在本 issue 范围）

- 见 sibling issue `SHARED_INFRA`（共享基础设施与 API 对齐）。
- Vision notebook 的预训练 6 个 cell 抽成 `examples/pretrain_vision.py`。
- 三算法 epoch/minibatch 骨架抽成 `training_loop`。

## 2026-06-08 调试与修复总结

### 结论

`examples/train_shac_state.ipynb` 不收敛的主因是 **SHAC 算法实现问题**，不是单纯的超参数问题。

本次确实也调整了一个参数：`SHACActor(initial_log_std=-2.0)`。该调整只降低 raw action 空间中的初始探索噪声，属于稳定性改进；它不能解释原实现中的 critic bootstrap 错位、actor loss 结算方式错误、TD-λ 目标方向错误等问题。

### 修复内容

1. **修正 critic bootstrap 对齐**

   文件：`flightning/algos/shac.py`

   原实现构造 `next_values` 时把 `samples.next_obs` 和 `final_obs` 拼接，导致 `td_lambda_targets()` 中的 `next_values[t + 1]` 实际偏到后一拍。修复后显式构造：

   - `next_values[0] = V(s_0)`
   - `next_values[t + 1] = V(s_{t+1})`

   这保证 critic target 中的 bootstrap value 与 rollout transition 对齐。

2. **修正 actor loss 的结算方式**

   文件：`flightning/algos/shac.py`

   参考 DiffRL 的 `compute_actor_loss()`，actor loss 应在 episode `done` 或短 horizon 末端结算累计折扣回报，并附加 `gamma * V(s_{t+1})` bootstrap。原实现把每个 prefix return 都平均进 actor loss，会改变 SHAC 的优化目标。修复后：

   - forward scan 累加 `rew_acc` 和折扣 `gamma`
   - 只在 `done` 或 horizon final step 产生 loss term
   - `done` 后重置累计回报和折扣

3. **修正 TD-λ target 的扫描方向**

   文件：`flightning/algos/_common.py`

   DiffRL 的 `compute_target_values()` 对 TD-λ 使用从后往前的 Ai/Bi trace。原 JAX 实现是正向 scan，critic 监督目标不等价。修复后使用 reverse scan，并把 horizon 最后一拍视为 terminal boundary。

4. **保留 full-horizon notebook 训练长度**

   文件：`examples/train_shac_state.ipynb`

   DiffRL 配置常用 `steps_num=32`，但在本 hovering state 任务上，100 epoch sanity test 显示 `num_steps_per_epoch=32` 反而退化；原 notebook 使用 `env.max_steps_in_episode` 的 full horizon 更稳定。因此没有把 notebook 改为 32-step SHAC。

5. **降低初始探索噪声**

   文件：`examples/train_shac_state.ipynb`

   增加 `initial_log_std=-2.0`。这是辅助稳定项，因为当前 actor action 直接处于真实 quadrotor action 空间，不是 DiffRL PyTorch 版中先输出再 `tanh` 的 normalized action。初始 `std=1` 对 thrust/angular-rate action 偏大。

### 验证结果

- `td_lambda_targets()` 与手写 DiffRL 风格反向循环对比：`maxerr 0.0`。
- GPU smoke test：`jax.devices()` 返回 `[cuda(id=0)]`，512-hidden actor/critic 路径可编译运行。
- 100 epoch sanity test（full horizon）：
  - deterministic mean return: `-38.98 -> -30.13`
  - value loss: `7.65 -> 1.95`
- 100 epoch sanity test（32-step horizon）：
  - deterministic mean return: `-38.98 -> -42.88`
  - 因此未采用 32-step horizon 作为 notebook 默认配置。

### 注意事项

- `flightning/algos/_common.py` 当前在工作树中是 untracked 文件，但 `flightning/algos/shac.py` 已依赖其中的 `clip_grads()`、`ema_update()`、`td_lambda_targets()`、`get_rollouts()`。提交本 issue 时必须包含该文件。
- `gitnexus detect-changes` 当前报告 medium risk，原因是工作树已有多处未提交变更；本次 SHAC 修复相关的符号影响分析为 LOW。

## 2026-06-08 Vision 对比与指标补充

### Notebook 修改

在 `examples/train_bptt_vision.ipynb` 和 `examples/train_shac_vision.ipynb` 的评估段新增 `eval_metrics` cell，用同一套指标比较两个算法：

- `mean_return` / `return_std`：最终评估 rollout 的平均回报与方差。
- `position_mse`：整段评估轨迹中当前位置到 `env_eval.goal` 的均方距离。
- `final_position_mse` / `final_position_rmse`：评估末端位置误差，避免只靠轨迹图主观判断。
- `collision_rate`：评估 episode 中出现 termination 的比例。
- `num_training_steps`：训练实际消耗的环境步数。
- `steps_per_second`：环境步吞吐量。
- `return_per_second`：按 wall-clock 归一化后的最终回报效率。

两个 notebook 的 metric cell 已通过 `python3 -m json.tool` 校验，并用 dummy transition 跑通过指标函数。

### Vision SHAC 与 BPTT 的当前结论

在本项目的 visual-feature hovering 任务上，SHAC 不如 BPTT 更像是任务与算法匹配度问题，而不是继续调一两个超参数就能根本解决的问题。

依据：

1. `paper/Learning Quadrotor Control From Visual Features Using Differentiable Simulation/2410.15979v3.html` 的核心对比是 BPTT/differentiable simulation 相对 PPO：论文明确写到 differentiable simulation 通过 dynamics model 反传梯度，提供 low-variance analytical policy gradients 和更高 sample efficiency；图注也写明 BPTT via differentiable simulation 在 state-based 与 vision-based control 上 outperform PPO。
2. 同一论文的 vision 任务并不是拿 SHAC 和 BPTT 做对比；它的正例算法就是 BPTT。该任务使用 visual features、可微 camera model、可微 quadrotor dynamics，并且依赖 pretraining 改善 vision policy 的收敛与最终性能。
3. `paper/Accelerating Visual-Policy Learning through Parallel Differentiable Simulation/2505.10646v2.html` 对 visual-SHAC 的描述更谨慎：SHAC 通过 short-horizon rollout 和 value function 缓解长轨迹优化噪声，但在视觉/渲染场景中会遇到更大的梯度范数、显存和计算开销；文中报告 3D visual-SHAC 可出现梯度范数快速超过 `1e15`，backward signal 退化为噪声。
4. `reference/DiffRL/README.md` 说明 SHAC 的定位是 GPU differentiable simulation + short-horizon actor-critic，用于可微仿真下的加速策略学习；它的优势不是在所有简单短 horizon 任务上压过 full-horizon BPTT。
5. `$graphify query` 在 `/home/tong/tongworkspace/paperworkspace` 的图谱里命中 `reference/DiffRL/algorithms/shac.py`、`reference/DiffRL/algorithms/bptt.py`、`reference/D.VA/algorithms/shac.py` 以及本项目 quadrotor/vision 节点，确认本次分析对应的本地代码与参考实现路径。

因此，本任务上 BPTT 更强是合理的：BPTT 直接沿完整可微轨迹优化 actor，没有 critic bootstrap bias，也没有 critic 拟合误差；当前任务 horizon 中等、状态转移和 visual feature observation 都可微且较平滑，正好适合 full-horizon analytical gradient。SHAC 的短 horizon + critic bootstrap 在这里引入了额外估计误差和优化耦合，优势没有充分发挥。

### 建议比较方式

后续不要只比较最终轨迹图。至少记录：

- final control quality：`position_mse`、`final_position_rmse`、`collision_rate`。
- sample efficiency：达到同一 `mean_return` 或 `final_position_rmse` 阈值所需的 `num_training_steps`。
- wall-clock efficiency：达到同一阈值所需秒数，以及 `steps_per_second`。
- stability：多 seed 的 `return_std`、收敛失败率、gradient norm。
- SHAC-specific health：critic loss、critic target MSE/value calibration、actor/critic loss 是否出现同步震荡。
- resource cost：GPU peak memory、JIT 编译时间、训练主循环时间。
