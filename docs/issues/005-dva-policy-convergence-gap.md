# Issue: 补齐 D.VA 动态避障策略收敛与真实避障评估

## Parent

`openspec/changes/replace-avoidance-training-validation-with-dva`

## Type

HITL + AFK

## What to build

把当前 D.VA headless smoke validation 推进到真实 actor policy 的闭环动态避障收敛验证。当前脚本和测试已经证明 D.VA 动态避障路径可运行、privileged critic schema 可构造、actor/critic metrics 有限，并在第一阶段短程动态课程中证明真实 `CNNLidarActor` rollout 可以到达目标。

## Current evidence

- `examples/train_dva_avoidance.py` 的 D.VA training metrics 有限，critic loss 会下降。
- 2026-06-13 的第一阶段短程动态课程结果：固定 seeds `(123, 124, 125, 126, 127)` 的真实 actor rollout 达到 `success_rate=0.80`、`termination_rate=0.20`。
- seed `123` 的真实 actor rollout 在第 `198` 步达到 `2.995m` 成功半径，`examples/outputs/dynamic_avoidance_dva_rerun.rrd` 记录该轨迹。
- 手写目标导向控制器现在只用于 warm start 数据采集和环境 sanity check；验收输出使用 `CNNLidarActor` 参数闭环 rollout。
- `stop_lidar_grad=True` 会阻断 LiDAR 渲染几何对环境状态的反向依赖；JAX 官方 `jax.lax.stop_gradient` 语义是让梯度计算忽略被标记的依赖。该模式适合作为 D.VA-style 稳定对照。

## Likely causes

1. **完整任务课程仍需扩展**：当前收敛结果限定在短程目标、低速/小半径动态障碍、起点 obstacle clearance 和冻结 actor 配置。
2. **actor-gradient 微调仍不稳定**：解冻 actor 后容易出现高度越界、超速或 collision，说明 D.VA actor-gradient 与 reward 尺度还需要继续诊断。
3. **reward 分量缺少日志**：当前 reward proxy 包含目标进度、距离、速度、高度、动作、clearance 和 TTC 风险，但训练日志仍缺少 goal progress、clearance risk、motion/TTC risk、height/action penalties 的分项输出。
4. **避障梯度路径仍需对照**：`stop_lidar_grad=True` 是当前稳定课程配置；解析 LiDAR 梯度模式的成功率和轨迹仍需单独报告。
5. **privileged critic schema 当前保持稳定**：critic observation 仍是 127 维 state-derived schema；本轮主要调整 actor-visible schema 和课程配置。

## Acceptance criteria

- [x] `examples/train_dva_avoidance.py` 默认导出的 `dynamic_avoidance_dva_rerun.rrd` 只记录真实 D.VA actor policy rollout。
- [x] 新增 policy evaluation summary，至少包含 success rate、collision/termination rate、mean/min final distance、mean episode length、mean return。
- [ ] 新增一个明确命名的 reference rollout 脚本或输出路径，例如 `dynamic_avoidance_goal_directed_reference.rrd`，用于检查环境和可视化，而不是训练收敛。
- [x] 训练 actor 使用与环境动作空间匹配的参数化策略，例如 `action_bias=env.hovering_action` 和有界 `action_scale`，并在文档中说明动作语义。
- [ ] 对比 `analytic_lidar_grad` 与 `stop_lidar_grad` 两种模式的 D.VA/BPTT 训练结果，至少报告 finite metrics、success rate 和避障轨迹差异。
- [ ] 增加 reward component logging，能区分 goal progress、clearance risk、motion/TTC risk、height/action penalties 对训练的贡献。
- [x] 至少一个固定 seed 的真实 D.VA actor policy rollout 达到 `dist_to_goal <= 3.0m` 且未触发 collision/out-of-bounds termination。
- [x] 批量 seed 评估中真实 D.VA actor policy success rate 达到后续实验设定的阈值，并记录阈值依据。
- [ ] 解冻 actor 的 D.VA 微调达到与冻结 actor 第一阶段课程相同或更高的 success rate。
- [ ] 将第一阶段课程逐步扩展到更长目标距离、更高动态障碍速度和默认半径范围。

## Verification commands

```bash
conda run -n flightning pytest tests/test_dva.py tests/test_dynamic_avoidance_env.py -q
conda run -n flightning python examples/train_dva_avoidance.py
```

## Notes

本 issue 已完成第一阶段课程内的“策略行为可信”验收。后续结论应区分三层：headless smoke validation、第一阶段课程收敛、完整动态避障收敛。当前代码达到第二层，第三层仍需要 reward 分量日志、LiDAR 梯度对照和解冻 actor 微调。
