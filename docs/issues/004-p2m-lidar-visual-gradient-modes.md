# Issue: 实现 P2M LiDAR 视觉梯度双模式并用 BPTT 首验收

## Parent

`docs/prd/p2m-dynamic-avoidance-migration.md`

## Type

AFK

## What to build

为 P2M 动态避障迁移定义并验证 LiDAR distance image 的两种视觉梯度模式：默认 `analytic_lidar_grad` 和稳定对照 `stop_lidar_grad`。

`analytic_lidar_grad` 使用 JAX 解析 raycasting 生成 LiDAR distance image，并允许梯度从观测经传感器几何回到环境状态。`stop_lidar_grad` 使用相同的前向观测，但通过 JAX `stop_gradient` 阻断 sensor-to-state 梯度，作为 D.VA-style 稳定性对照。第一条端到端训练验收使用 BPTT。

## Acceptance criteria

- [ ] `analytic_lidar_grad` 和 `stop_lidar_grad` 两种模式有明确配置入口，并且前向 LiDAR distance image 数值一致。
- [ ] `analytic_lidar_grad` 在非退化几何场景中，对相关环境状态产生有限且非零的 LiDAR observation 梯度。
- [ ] `stop_lidar_grad` 在相同场景中阻断 sensor-to-state 梯度，但不阻断 policy action 到 dynamics/reward 的训练梯度。
- [ ] 两种模式都兼容 JAX `jit`、`lax.scan` 和显式 PRNG 管理。
- [ ] 第一条端到端训练 smoke test 使用 BPTT，能够完成 compile、rollout、loss 计算和参数更新，并产生有限 metrics。
- [ ] 文档说明参考来源：DiffAero-style analytical differentiable raycasting、D.VA-style observation-gradient decoupling、JAX `grad` / `stop_gradient` / `lax.scan`。

## Blocked by

None - can start immediately

## Notes

该 issue 不要求实现完整 reward 公式，也不要求 SHAC 首验收。reward/termination 细化由 `docs/issues/003-p2m-safety-dobs-reward-semantics.md` 继续承接。
