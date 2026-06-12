# D.VA 动态避障收敛调试档案

更新时间：2026-06-12

## 目的

本文档整理当前 `examples/train_dva_avoidance.py` 的训练状态、Rerun 输出语义、已知缺口和下一阶段调试入口。下一阶段目标是完成真实 D.VA actor policy 的闭环动态避障收敛：策略需要在避开动态障碍物的同时到达目标。

## 当前结论

当前实现已经完成 D.VA headless smoke validation：脚本可在远程 headless `flightning` conda 环境运行，D.VA rollout、privileged critic observation、done/truncated bootstrap 和 reward proxy 数值稳定路径已打通。

真实 D.VA actor policy 仍处于收敛缺口状态。最近一次真实 actor rollout 在第 30 步结束，`reached_target=False`，最小目标距离仍为 `21.371 m`，总回报为 `-357.7752`。这说明当前训练指标只能支持“路径可运行、loss 有限、critic loss 下降”的结论，策略行为仍需继续调参和诊断。

## 最近一次训练输出

命令：

```bash
conda run -n flightning python examples/train_dva_avoidance.py
```

关键输出：

```text
Final D.VA actor loss: 7.0813, value loss: 280.3751
Convergence check: actor plateau ref=8.3834, actor tail=7.8433, value initial=2887.1697, value tail=358.7524
Episode ended at step 30. Reason: Collision/Out of Bounds
D.VA policy evaluation: reached_target=False, min_target_dist=21.371 m, total_reward=-357.7752
```

当前 `examples/outputs/dynamic_avoidance_dva_rerun.rrd` 文件大小约 `202 KiB`，mtime 为 `2026-06-12 21:59:56 +0800`。该 RRD 来自真实 D.VA actor policy rollout。只有约 30 帧的原因是 episode 提前结束。

## 术语边界

| 术语 | 本仓库含义 | 当前状态 |
| --- | --- | --- |
| D.VA headless smoke validation | 验证脚本、rollout、privileged critic schema、bootstrap 语义和 finite metrics | 已完成 |
| Dynamic avoidance policy convergence | 用真实 actor policy 闭环评估证明到达目标、避开碰撞、成功率达标和轨迹可信 | 当前缺口 |
| Goal-directed reference controller | 手写目标导向控制器，用于验证环境动力学和 Rerun 可视化链路 | 只作为参考轨迹 |
| Real D.VA actor rollout | `trained_state.apply_fn(trained_state.params, obs)` 直接输出动作并驱动环境 | 当前评估主证据 |

## 当前脚本配置

来源：`examples/train_dva_avoidance.py`

### 环境参数

```python
DynamicAvoidanceConfig(
    stop_lidar_grad=True,
    clearance_weight=0.5,
    motion_risk_weight=0.05,
    barrier_temperature=0.75,
)
```

### 网络和优化器

```python
actor_model = CNNLidarActor(feature_list=[442, 64, 64, action_dim])
critic_model = SHACCritic(feature_list=[127, 64, 64, 1])
actor_lr = 3e-5
critic_lr = 1e-3
```

当前 actor 直接输出 4 维低层动作，动作参数化仍待加入 `env.hovering_action` 中心化和显式 `action_scale`。低层动作语义是 collective thrust + 3 维 body rates。

### D.VA 训练参数

```python
num_epochs = 80
num_steps_per_epoch = 20
num_envs = 8
DVAConfig(
    logging=True,
    logging_freq=10,
    critic_iterations=2,
    num_batches=2,
    critic_method="td-lambda",
    gamma=0.9,
    lam=0.9,
    max_grad_norm=0.5,
)
```

### 评估参数

```python
key_eval = jax.random.PRNGKey(123)
total_steps = 150
success_radius = 3.0
rrd_path = "examples/outputs/dynamic_avoidance_dva_rerun.rrd"
```

评估动作来源：

```python
action = trained_state.apply_fn(trained_state.params, obs)
```

## Rerun 输出解释

`examples/outputs/dynamic_avoidance_dva_rerun.rrd` 表示真实 D.VA actor policy rollout。当前文件短，是训练后策略在固定 seed `123` 下提前触发 `Collision/Out of Bounds`。

之前出现过 seed `1000`、约第 `349` 步到达 `2.982 m` 成功半径的轨迹。该轨迹来自手写目标导向控制器，适合用来检查环境动力学、目标半径和可视化链路。真实收敛证据应以 D.VA actor rollout 的成功率、episode 长度、终止率、最小目标距离和轨迹避障行为为准。

## 已知调试线索

1. Action parameterization 是最高优先级。当前 `CNNLidarActor` 直接输出物理动作，初期训练容易落入坠落、直飞或过大姿态扰动。下一步应把 actor 输出中心设到 `env.hovering_action` 附近，并使用有界 `action_scale` 控制 body-rate 和 thrust 扰动幅度。
2. `stop_lidar_grad=True` 是稳定对照。JAX `jax.lax.stop_gradient` 的官方语义是让梯度计算忽略被标记的依赖；在这里保留前向 LiDAR 观测，同时切断 LiDAR raycasting 几何到状态的反向依赖。下一步应对比 `stop_lidar_grad=True` 和解析 LiDAR 梯度模式的成功率与轨迹。
3. Reward 分量缺少训练时诊断。当前总 reward 能保持有限，但缺少 goal progress、distance、clearance、motion/TTC、height、action penalty 的分量日志，导致直飞、保守悬停和碰撞失败的根因难以区分。
4. 评估指标偏单 seed。当前脚本只导出一个固定 seed 的 RRD。下一步需要批量 seed 评估，至少记录 success rate、termination rate、final/min target distance、episode length 和 return。
5. Critic loss 下降说明 value fitting 有进展，但 actor loss plateau 只能证明有限和相对稳定。行为收敛需要真实闭环 rollout 证明。

## 下一阶段实验顺序

1. **动作中心化实验**
   - 修改 `CNNLidarActor` 初始化或调用方式，使策略输出围绕 `env.hovering_action`。
   - 使用显式 `action_scale`，先限制 body rates 和 thrust 扰动。
   - 成功标准：rollout episode 长度增加，collision/out-of-bounds rate 下降，最小目标距离改善。

2. **批量评估指标**
   - 在训练后跑多个 eval seeds。
   - 输出 success rate、termination rate、mean/min final distance、mean episode length、mean return。
   - RRD 只保留指定 seed 的可视化样本，批量指标写入 stdout 或 JSON。

3. **Reward 分量日志**
   - 在 `_get_reward_jit` 或环境 info 中暴露关键 reward terms。
   - 对失败 rollout 记录每步分量均值和末端分量。
   - 用分量日志判断策略是在追目标、避障、控高还是被动作惩罚主导。

4. **LiDAR 梯度对照**
   - 固定训练预算，分别运行 `stop_lidar_grad=True` 与解析 LiDAR 梯度。
   - 对比 finite metrics、success rate、trajectory behavior。
   - 依据：Context7 查询的 JAX 文档中，`stop_gradient` 会让梯度计算忽略标记值的依赖；该事实直接影响视觉几何路径的 actor 梯度。

5. **训练预算和尺度微调**
   - 在动作参数化和评估指标稳定后，再扩大 `num_epochs`、`num_steps_per_epoch` 或调整 `gamma/lam`。
   - 同步观察 critic target scale、value loss tail 和 actor rollout 行为。

## 推荐验收命令

基础测试：

```bash
conda run -n flightning pytest tests/test_dva.py tests/test_dynamic_avoidance_env.py -q
```

训练与单 seed RRD：

```bash
conda run -n flightning python examples/train_dva_avoidance.py
```

OpenSpec 校验：

```bash
conda run -n flightning openspec validate replace-avoidance-training-validation-with-dva --strict
```

## 关键源码入口

- `examples/train_dva_avoidance.py`：当前 D.VA 训练脚本、actor/critic 初始化、评估 rollout 和 RRD 导出。
- `flightning/algos/dva.py`：D.VA rollout、adapter、actor/critic update 和 bootstrap 逻辑。
- `flightning/envs/dynamic_avoidance_env.py`：动态避障环境状态、reward proxy、termination、privileged critic observation。
- `flightning/modules/cnn_lidar_policy.py`：LiDAR actor 网络和动作输出参数化入口。
- `flightning/sensors/mujoco_lidar_sensor.py`：JAX LiDAR raycasting 和梯度路径。
- `docs/issues/005-dva-policy-convergence-gap.md`：收敛缺口 issue 和验收项。
- `CONTEXT.md`：D.VA smoke validation 与 dynamic avoidance policy convergence 的全局术语边界。
- `openspec/changes/replace-avoidance-training-validation-with-dva/design.md`：本轮 OpenSpec 的设计边界和风险。
- `openspec/changes/replace-avoidance-training-validation-with-dva/tasks.md`：本轮 OpenSpec 已完成项和 follow-up 指向。

## 接手建议

下一步从 action parameterization 和批量 policy evaluation 入手。先让真实 D.VA actor policy 的 episode 活得更长、终止率下降，再用 reward 分量和 LiDAR 梯度对照判断是否具备障碍物感知和绕行动机。
