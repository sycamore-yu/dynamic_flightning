# D.VA 动态避障收敛调试档案

更新时间：2026-06-13

## 目的

本文档整理当前 `examples/train_dva_avoidance.py` 的训练状态、Rerun 输出语义、已验证改动和下一阶段调试入口。当前目标已经推进到第一阶段短程动态课程收敛：真实 `CNNLidarActor` 参数闭环 rollout 在固定多 seed 评估中达到目标。

## 当前结论

当前实现已经完成 D.VA headless smoke validation，并在短程动态避障课程中得到可复现实验结果。脚本可在远程 headless `flightning` conda 环境运行，D.VA rollout、privileged critic observation、done/truncated bootstrap、reward proxy 数值稳定路径和多 seed policy evaluation 均已打通。

最新可复现结果来自真实 `CNNLidarActor` rollout：固定 seeds `(123, 124, 125, 126, 127)` 的 `success_rate=0.80`、`termination_rate=0.20`。seed `123` 在第 `198` 步进入 `3.0 m` 成功半径，最小目标距离为 `2.995 m`。当前结论限定在第一阶段课程：短程目标、低速/小半径动态障碍、起点 obstacle clearance、goal-directed warm start、D.VA critic validation 和冻结 actor。

## 2026-06-13 训练输出

命令：

```bash
conda run -n flightning python examples/train_dva_avoidance.py
```

关键输出：

```text
Warm start actor imitation: samples=6105, final_mse=0.069710
Final D.VA actor loss: 1.6756, value loss: 1085.4073
Convergence check: actor plateau ref=2.2802, actor tail=2.2685, value initial=3323.0039, value tail=1116.3563
D.VA policy evaluation seed=123: reached_target=True, min_target_dist=2.995 m, final_target_dist=2.995 m, episode_len=198, total_reward=-644.2815
D.VA policy evaluation summary: success_rate=0.80, termination_rate=0.20, mean_episode_len=143.4, mean_min_target_dist=3.606 m, mean_final_target_dist=4.040 m, mean_return=-1014.4397
```

当前 `examples/outputs/dynamic_avoidance_dva_rerun.rrd` 文件大小约 `1.1 MiB`，mtime 为 `2026-06-13 00:37 +0800`。该 RRD 来自真实 `CNNLidarActor` rollout 的 seed `123` 轨迹。

## 术语边界

| 术语 | 本仓库含义 | 当前状态 |
| --- | --- | --- |
| D.VA headless smoke validation | 验证脚本、rollout、privileged critic schema、bootstrap 语义和 finite metrics | 已完成 |
| D.VA 第一阶段课程收敛 | 短程动态课程中真实 actor rollout success rate 达标 | 已完成 |
| Dynamic avoidance policy convergence | 完整动态避障任务中用真实 actor policy 闭环评估证明到达目标、避开碰撞、成功率达标和轨迹可信 | 后续工作 |
| Goal-directed reference controller | 手写目标导向控制器，用于验证环境动力学和 Rerun 可视化链路 | 只作为参考轨迹 |
| Real D.VA actor rollout | `trained_state.apply_fn(trained_state.params, actor_obs)` 直接输出动作并驱动环境 | 当前评估主证据 |
| Actor-visible schema | 226 维归一化 LiDAR + body-frame target/velocity + height + last body rates | 已更新 |
| Privileged critic schema | 127 维 state-derived critic observation | 维度和语义保持稳定 |

## 当前脚本配置

来源：`examples/train_dva_avoidance.py`

### 环境参数

```python
DynamicAvoidanceConfig(
    trace_prob=1.0,
    stop_lidar_grad=True,
    clearance_weight=2.0,
    motion_risk_weight=0.2,
    barrier_temperature=0.5,
    dobs_vel_range=(0.3, 1.0),
    dobs_radius_range=(0.15, 0.25),
    reset_obstacle_clearance=5.0,
    reset_target_offset=28.0,
)
```

### 网络和优化器

```python
actor_model = CNNLidarActor(
    feature_list=[442, 64, 64, action_dim],
    action_bias=env.hovering_action,
    action_scale=jnp.array([2.5, 3.0, 3.0, 2.0]),
    initial_scale=0.1,
    initial_log_std=-2.0,
    min_std=0.05,
)
critic_model = SHACCritic(feature_list=[127, 64, 64, 1])
actor_lr = 0.0
critic_lr = 1e-3
```

当前 actor 通过 goal-directed reference actions warm start 获得初始参数，D.VA 阶段冻结 actor 并训练 critic。低层动作语义是 collective thrust + 3 维 body rates。`actor_lr=0.0` 是第一阶段课程收敛的稳定配置。

### D.VA 训练参数

```python
num_epochs = 80
num_steps_per_epoch = 150
num_envs = 8
DVAConfig(
    logging=True,
    logging_freq=10,
    critic_iterations=4,
    num_batches=2,
    critic_method="td-lambda",
    gamma=0.97,
    lam=0.92,
    max_grad_norm=0.5,
)
```

### 评估参数

```python
eval_seeds = (123, 124, 125, 126, 127)
total_steps = 400
success_radius = 3.0
rrd_path = "examples/outputs/dynamic_avoidance_dva_rerun.rrd"
```

评估动作来源：

```python
actor_obs = dynamic_avoidance_dva_adapter(obs, state).actor_obs
action = trained_state.apply_fn(trained_state.params, actor_obs)
```

## Rerun 输出解释

`examples/outputs/dynamic_avoidance_dva_rerun.rrd` 表示真实 `CNNLidarActor` policy rollout。当前文件记录 seed `123` 达到目标的轨迹。

之前出现过 seed `1000`、约第 `349` 步到达 `2.982 m` 成功半径的轨迹。该轨迹来自手写目标导向控制器，适合用来检查环境动力学、目标半径和可视化链路。真实收敛证据应以 D.VA actor rollout 的成功率、episode 长度、终止率、最小目标距离和轨迹避障行为为准。

## 已验证调试结论

1. 动作中心化和有界 `action_scale` 是必要改动。直接输出低层动作的 baseline 在 seed `123` 约第 30 步终止；加入 `env.hovering_action` 和 `action_scale` 后 episode length 与目标距离持续改善。
2. actor 原始观测缺少低层 body-rate action 所需的 body-frame 语义和高度。`dynamic_avoidance_dva_adapter` 已把 actor-visible schema 调整为归一化 LiDAR、body-frame target/velocity、height 和 last body rates。
3. 课程难度需要显式配置。默认 40 个动态障碍、速度 1-5 m/s、无起点 clearance 的环境会产生早碰撞和高方差失败；第一阶段课程使用低速/小半径动态障碍、起点 clearance 和短程目标。
4. 当前 actor-gradient D.VA 微调会破坏 warm-start 控高行为。第一阶段稳定配置冻结 actor，仅训练 critic 并执行 D.VA rollout/metrics validation。
5. 多 seed 评估已经纳入脚本，输出 success rate、termination rate、episode length、min/final distance 和 return。

## 下一阶段实验顺序

1. **Reward 分量日志**
   - 在 `_get_reward_jit` 或环境 info 中暴露关键 reward terms。
   - 对失败 rollout 记录每步分量均值和末端分量。
   - 用分量日志判断策略是在追目标、避障、控高还是被动作惩罚主导。

2. **LiDAR 梯度对照**
   - 固定训练预算，分别运行 `stop_lidar_grad=True` 与解析 LiDAR 梯度。
   - 对比 finite metrics、success rate、trajectory behavior。
   - 依据：Context7 查询的 JAX 文档中，`stop_gradient` 会让梯度计算忽略标记值的依赖；该事实直接影响视觉几何路径的 actor 梯度。

3. **解冻 actor 的 D.VA 微调**
   - 从当前课程收敛配置出发，逐步恢复 actor learning rate。
   - 同步观察高度终止、速度终止、collision 和 success rate。
   - 需要配合 reward 分量日志判断 actor-gradient 更新的主导信号。

4. **课程扩展**
   - 从 `reset_target_offset=28.0` 的短程目标逐步回到更长路径。
   - 逐步提高 `dobs_vel_range`、`dobs_radius_range` 和追踪难度。
   - 每个阶段保持固定 seed batch 和 RRD 样本输出。

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

下一步从 reward 分量日志、解析 LiDAR 梯度对照和解冻 actor 微调入手。当前课程已经证明真实 actor rollout 可以在短程动态场景中到达目标；完整动态避障收敛仍需要把 actor-gradient、reward 语义和课程难度逐步接上。
