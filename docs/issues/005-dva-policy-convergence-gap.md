# Issue: 补齐 D.VA 动态避障策略收敛与真实避障评估

## Parent

`openspec/changes/replace-avoidance-training-validation-with-dva`

## Type

HITL + AFK

## What to build

把当前 D.VA headless smoke validation 推进到真实 actor policy 的闭环动态避障收敛验证。当前脚本和测试能证明 D.VA 动态避障路径可运行、privileged critic schema 可构造、actor/critic metrics 有限，但还不能证明训练出的 actor policy 会有效避障、绕开动态障碍物并到达目标。

## Current evidence

- `examples/train_dva_avoidance.py` 的 D.VA training metrics 有限，critic loss 会下降。
- 真实 D.VA actor policy 在评估 rollout 中会较早结束，批量 seed 搜索未观察到 `dist_to_goal <= 3.0m` 的成功 episode。
- 手写目标导向控制器能在 seed `1000` 上约第 `349` 步到达 `2.982m` 成功半径，但该轨迹只证明环境动力学和可视化链路可运行，不能作为 D.VA actor policy 收敛证据。
- `stop_lidar_grad=True` 会阻断 LiDAR 渲染几何对环境状态的反向依赖；JAX 官方 `jax.lax.stop_gradient` 语义是让梯度计算忽略被标记的依赖。该模式适合作为 D.VA-style 稳定对照，但它削弱了纯视觉避障 reward 对 actor 的直接几何梯度。

## Likely causes

1. **验收目标偏 smoke**：当前 OpenSpec 只要求 finite metrics 和 headless command 成功，缺少真实 policy success-rate 验收。
2. **action parameterization 弱**：`CNNLidarActor` 默认输出未围绕 `env.hovering_action` 和动作边界归一化；低层动作是 collective thrust + body rates，未中心化时策略容易先学到坠落、直飞或过大姿态扰动。
3. **避障梯度弱**：`stop_lidar_grad=True` 保留前向 LiDAR 观测，但会切断 raycasting 几何到 state 的反向路径；当前 object-free clearance/TTC reward 主要证明数值稳定，尚未证明能驱动绕障策略形成。
4. **reward 目标不完整**：当前 reward proxy 包含目标进度、距离、速度、高度、动作、clearance 和 TTC 风险，但缺少成功率驱动的训练评估闭环、reward 分量日志和针对绕障行为的失败诊断。
5. **评估混淆风险**：目标导向参考控制器生成的 Rerun 轨迹可能被误读成训练 actor 的行为，D.VA policy Rerun 和 reference controller Rerun 必须分开命名。

## Acceptance criteria

- [ ] `examples/train_dva_avoidance.py` 默认导出的 `dynamic_avoidance_dva_rerun.rrd` 只记录真实 D.VA actor policy rollout。
- [ ] 新增 policy evaluation summary，至少包含 success rate、collision/termination rate、mean/min final distance、mean episode length、mean return。
- [ ] 新增一个明确命名的 reference rollout 脚本或输出路径，例如 `dynamic_avoidance_goal_directed_reference.rrd`，用于检查环境和可视化，而不是训练收敛。
- [ ] 训练 actor 使用与环境动作空间匹配的参数化策略，例如 `action_bias=env.hovering_action` 和有界 `action_scale`，并在文档中说明动作语义。
- [ ] 对比 `analytic_lidar_grad` 与 `stop_lidar_grad` 两种模式的 D.VA/BPTT 训练结果，至少报告 finite metrics、success rate 和避障轨迹差异。
- [ ] 增加 reward component logging，能区分 goal progress、clearance risk、motion/TTC risk、height/action penalties 对训练的贡献。
- [ ] 至少一个固定 seed 的真实 D.VA actor policy rollout 达到 `dist_to_goal <= 3.0m` 且未触发 collision/out-of-bounds termination。
- [ ] 批量 seed 评估中真实 D.VA actor policy success rate 达到后续实验设定的阈值，并记录阈值依据。

## Verification commands

```bash
conda run -n flightning pytest tests/test_dva.py tests/test_dynamic_avoidance_env.py -q
conda run -n flightning python examples/train_dva_avoidance.py
```

## Notes

本 issue 的核心是把“验证路径可运行”升级为“策略行为可信”。在该 issue 完成前，D.VA dynamic avoidance 的结论应表述为 smoke validation passed，而不是 policy converged。
