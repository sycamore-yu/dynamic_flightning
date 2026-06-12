# Issue: 明确 P2M safety/dobs reward proxy 与 hard termination 语义

## Parent

`docs/prd/p2m-dynamic-avoidance-migration.md`

## Type

HITL

## What to build

明确迁移后的 P2M 动态避障任务中 reward 与 termination 的边界：hard collision/termination 保持离散事件，reward 使用连续或分段可微的 safety/dobs proxy，为 BPTT/SHAC 提供可用的避障学习信号。

本 issue 不要求实现完整环境，而是产出可执行的 reward/termination 设计合同，后续 AFK 实现 issue 必须按该合同编码和测试。

## Context

P2M 源任务同时包含 `reward_safety`、`reward_dobs` 和 hard termination/misbehave。Flightning 现有 hovering 环境则使用 hard collision termination，并将 collision penalty 通过 stop-gradient 从主梯度路径隔离。迁移方案已确认采用折中边界：termination 是离散 hard event，reward 使用连续或分段可微 proxy。

## Acceptance criteria

- [ ] 明确 hard termination 触发条件，包括 obstacle proximity、wall/bounds violation、height violation、velocity/acceleration violation、NaN state 和 time truncation。
- [ ] 明确 safety reward proxy 的输入、输出、符号、归一化、clip/log/smooth 处理，以及它如何从 LiDAR distance image 计算。
- [ ] 明确 dobs reward proxy 的输入、输出、符号、归一化、clip/log/smooth 处理，以及它如何使用 dynamic obstacle state、drone position 和 drone velocity。
- [ ] 明确哪些 reward 项允许分段可微，哪些 hard event 必须 stop-gradient 或只进入 termination。
- [ ] 明确 P2M 数值对齐用例：固定 drone state、obstacle state、wall/bounds state、LiDAR distance image 时，JAX proxy 与 P2M reference 的容差。
- [ ] 明确 BPTT/SHAC 梯度验收：在非碰撞、非边界样例中，reward 对 policy action 路径产生有限且非零梯度；在 hard termination 样例中不会产生 NaN 梯度。
- [ ] 将最终决策同步回 `CONTEXT.md` 和 `docs/prd/p2m-dynamic-avoidance-migration.md`。

## Blocked by

None - can start immediately

## Notes

该 issue 是 HITL，因为 reward/termination 边界会影响训练目标，且后续改动代价高。讨论时必须同时引用 P2M reward/termination 源实现和 Flightning 当前 collision/reward 实现。
