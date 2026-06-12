# PRD: P2M 动态避障任务迁移到 Flightning 可微分训练栈

## Problem Statement

用户希望将 P2M 中“无人机基于 LiDAR 图像观测躲避动态障碍物并飞向目标”的任务语义迁移到 Flightning，并使用 Flightning 的 JAX 可微分强化学习算法 BPTT 或 SHAC 训练，而不是迁移 P2M 的 Torch、ROS、NeuFlow 或 PPO 训练机制。

当前问题是：P2M 的任务语义、观测、动态障碍物、LiDAR 几何、reward 和终止条件与 Flightning 的 JAX Env API、BPTT/SHAC rollout、现有 quadrotor 低层动作接口之间还没有明确的组件边界和验收标准。

## Solution

实现一个 JAX/JIT/scan 兼容的动态避障任务环境，以 Flightning Env API 作为唯一训练接口。迁移 P2M 的任务语义，包括 LiDAR distance image、目标方向、速度、last_action、动态障碍物、wall/bounds、目标导航、reward 与 termination，并对 P2M 的关键数值行为做对齐验证。

第一版不迁移 P2M 的 PPO 训练、Torch tensor、ROS raycast、NeuFlow 推理或 acceleration controller。策略动作采用 Flightning 现有低层动作接口：collective thrust + 3 维 body rates。观测中的 last_action 是上一时刻实际进入 Flightning 环境状态的 4 维低层动作，在迁移中承担 P2M `acc_input` 的“上一控制输入反馈”角色。视觉梯度默认采用解析 LiDAR 梯度，同时保留阻断 LiDAR 梯度作为稳定性对照。

## User Stories

1. As a differentiable RL researcher, I want a P2M-style dynamic avoidance task in Flightning, so that I can train it with BPTT or SHAC.
2. As a researcher, I want the migrated task to use Flightning Env API, so that BPTT and SHAC can call it without special adapters.
3. As a researcher, I want the environment state to be a JAX pytree, so that rollouts can be jitted, scanned, vectorized, and differentiated.
4. As a researcher, I want LiDAR distance images generated in JAX, so that the training path does not depend on Torch, ROS, or external inference.
5. As a researcher, I want P2M default LiDAR parameters migrated, so that comparisons against P2M are meaningful.
6. As a researcher, I want dynamic obstacles represented in environment state, so that obstacle motion participates in reset, step, observation, and reward.
7. As a researcher, I want wall and bounds handling included in the first version, so that the task matches the P2M arena semantics rather than an easier subset.
8. As a researcher, I want LiDAR ray directions implemented as a sensor-level component, so that ray generation is reusable and not coupled to a single task.
9. As a researcher, I want the policy action semantics to match Flightning's low-level quadrotor interface, so that the existing differentiable dynamics path is reused.
10. As a researcher, I want numerical alignment tests against P2M, so that the JAX migration can be validated before training conclusions are trusted.
11. As a researcher, I want reward terms split by meaning, so that goal progress, safety, dynamic obstacle risk, smoothness, and control cost can be inspected independently.
12. As a researcher, I want termination and reward semantics documented separately, so that collision handling does not silently change the optimization objective.
13. As a researcher, I want a CNN LiDAR policy that works with the first observation shape, so that BPTT/SHAC training can use the image-like structure of LiDAR distance observations.
14. As a maintainer, I want deep modules for LiDAR geometry and obstacle dynamics, so that they can be tested independently from full training.
15. As a maintainer, I want P2M-specific compatibility decisions isolated in the task PRD, so that global project terminology remains stable in CONTEXT.
16. As a researcher, I want both analytic and stop-gradient LiDAR observation modes, so that I can compare full visual feedback gradients against a stable decoupled baseline.
17. As a researcher, I want the first end-to-end acceptance to use BPTT, so that the shortest differentiable rollout path is validated before adding SHAC-specific critic complexity.

## Implementation Decisions

- Build a Dynamic Avoidance Env as the single task-level environment exposed to training.
- Build a Dynamic Avoidance Env State that carries quadrotor state, target state, dynamic obstacle state, wall/bounds state, LiDAR-related history if needed, last actions, time, and step index.
- Build a Dynamic Obstacle Field module responsible for obstacle initialization and per-step obstacle motion.
- Build a LiDAR Ray Direction Generator in the sensor layer.
- Build a LiDAR Distance Image module that computes dynamic obstacle hits, wall hits, bounds hits, merges nearest hits, and produces a dense LiDAR distance image.
- Build an Observation Builder that emits LiDAR distance image + target direction + velocity + last_action.
- Build Reward and Termination logic that preserves P2M task semantics while remaining JAX-compatible: hard collision/termination remains discrete, while reward uses continuous or piecewise-differentiable safety/dynamic-obstacle proxies.
- Build a CNN LiDAR policy module that encodes the LiDAR distance image and fuses it with target direction, velocity, and last_action before producing the low-level action.
- Build two visual-gradient modes for the LiDAR observation path: analytic_lidar_grad, which allows gradients through JAX analytical raycasting, and stop_lidar_grad, which uses the same forward observation while stopping sensor-to-state gradients.
- Use Flightning Env API as the only training interface.
- Use the low-level Flightning action interface: collective thrust + 3D body rates.
- Use P2M default LiDAR training parameters: range 10, horizontal resolution 36, vertical resolution 6, horizontal sample 3, vertical sample 3, horizontal FOV 360, vertical FOV [-7, 52].
- Include dynamic obstacles, walls, and bounds in the first implementation rather than splitting them into later phases.
- Migrate P2M's dynamic-obstacle motion semantics as boundary bounce plus trace_prob-driven drone tracing, implemented with explicit JAX PRNG for jit/scan compatibility.
- Preserve user-confirmed dynamic-obstacle details: dobs_state state machine, asymmetric z-velocity handling during bounce/tracing, post-bounce state[2] flip to avoid lock-in, and boundary-touch position perturbation as a small nudge rather than full resampling.
- Use CNN as the first policy architecture, following P2M's image-like LiDAR encoder pattern rather than a flattened LiDAR MLP.
- Use analytic_lidar_grad as the default visual-gradient mode and keep stop_lidar_grad as an explicit stability/ablation mode. Reference sources: DiffAero-style analytical differentiable raycasting, D.VA-style observation-gradient decoupling, and JAX's `grad` / `stop_gradient` / `lax.scan` mechanisms.
- Use BPTT as the first end-to-end training acceptance target; SHAC remains in scope but is not the first closed-loop proof.
- Do not migrate PPO as the training mechanism.
- Do not use Torch, ROS, Python deques, or NeuFlow inference in the training path.
- Do not migrate P2M's acceleration controller into the first training interface.
- Use P2M numerical alignment as the main validation strategy before treating training results as meaningful.

## 传感器、动态障碍物与可视化架构设计 (Sensor, Obstacle, and Visualization Architecture)

结合 P2M 的数学逻辑与 Flightning 的 JAX/JIT 约束，本迁移方案对 LiDAR 传感器、动态障碍物交互以及可视化制定了以下具体架构设计：

### 1. 障碍物与碰撞的 JAX 表示
*   **状态表示 (State Representation)**：在 JAX 的环境状态树 (`EnvState`) 中，动态障碍物被统一建模为垂直圆柱体 (Cylinders)。其状态信息被稠密地存储在 Tensor 中，包括当前坐标 `dobs_pos (x, y)`、运动速度 `dobs_vel (vx, vy)`、半径 `r` 以及高度信息。
*   **碰撞判定 (Collision Detection)**：为了保持 JAX 编译的高效性，碰撞不依赖任何外部引擎。无人机（坐标 `drone_pos`）与障碍物的碰撞判定退化为连续的几何距离计算：
    $\text{Distance}_{xy} = \Vert \text{drone\_pos}_{xy} - \text{dobs\_pos}_{xy} \Vert_2$
    当 $\text{Distance}_{xy} < (r_{obs} + \text{error\_tolerance})$ 且高度坐标落入障碍物高度区间内时，触发碰撞判定。
    这一连续的解析表示使得不仅可以产生离散的 `terminated` 信号，还能直接衍生出可微的碰撞惩罚 (Safety Proxy Reward)，允许策略通过梯度安全地推开障碍物。

### 2. LiDAR 与障碍物的解析交互 (Analytic Raycasting)
P2M 源码中采用的是纯数学代数方程来计算 LiDAR 命中距离。在 JAX 迁移中，我们将严格保留这一“求交”本质，这为实现完全可微的**解析 LiDAR 梯度 (analytic_lidar_grad)** 奠定了基础：
*   **射线方向矩阵**：根据 LiDAR 的水平与垂直 FOV 及分辨率，利用 `jnp.meshgrid` 和三角函数预先计算并缓存全局固定的射线方向向量矩阵。
*   **代数求交 (Algebraic Intersection)**：对于每条射线与每一个圆柱形障碍物，JAX 在并行维度上解 2D 射线的二次方程。若判别式大于零且射线参数 $t > 0$（即在前方），同时交点的 $Z$ 坐标在圆柱体内，则视为有效命中，计算出命中距离 $t$。
*   **遮挡与最小值合并 (Occlusion Merging)**：对于单条射线可能穿透多个障碍物或墙壁的情况，利用 `jnp.min(hits, axis=-1)` 提取所有有效交点中的最短距离。未命中任何物体的射线填充最大量程 `lidar_range`。
*   **梯度反传**：因为上述射线生成、解方程和最小值选择过程仅包含连续的代数运算，JAX 的 `jax.grad` 能够无缝穿越这一过程。当策略损失反向传播时，误差能够通过 LiDAR 图像的每一个像素，反向流入环境的 `drone_pos` 和 `dobs_pos`，为可微强化学习提供强大的解析视觉梯度。
*   **梯度阻断 (stop_lidar_grad)**：为作稳定性对照，可通过 `jax.lax.stop_gradient(env_state)` 在前向求解 LiDAR 距离之前切断无人机/障碍物状态的计算图。这使得策略依旧能看见深度图，但剔除了环境动力学的视觉梯度回流。

### 3. 可视化方案 (Visualization Strategy)
为了摆脱对 ROS 和 Torch 的依赖，并适应 BPTT / SHAC rollout 产生的庞大 Tensor 轨迹，设计以下纯 Python/JAX 的轻量化呈现策略：
*   **LiDAR 观测热力图**：利用 `matplotlib.pyplot.imshow` 配合伪彩图 (`cmap='jet'`)，直接将 `(v_num, h_num)` 的 LiDAR 距离矩阵渲染为 2D 深度图。这对于校验死角、传感器盲区以及代码解析正确性极其直观。
*   **3D 场景与动态回放 (集成 Rerun-SDK)**：
    *   取消传统的 matplotlib 3D/动画绘制。鉴于 JAX 轨迹数据为纯 Numpy/DeviceArray，我们直接将全周期的轨迹数据（如 `drone_pos`, `dobs_pos`）以及从 LiDAR 深度图还原出的 **3D 点云 (3D Point Cloud)** 发送给 `rerun-sdk`。
    *   Rerun-SDK 提供原生支持可旋转、可交互的高帧率 3D 渲染 GUI，只需数行代码即可实现复杂的 3D 图元推流与时序回放，极大地提升复杂动态避障策略的分析与调试效率。

## Testing Decisions

- Good tests should verify external behavior and numerical task semantics, not implementation details.
- Test LiDAR ray direction generation for shape, FOV coverage, normalization, and P2M default parameter compatibility.
- Test dynamic obstacle initialization against P2M distributions for position, velocity, radius, and count.
- Test dynamic obstacle update behavior against P2M semantics, including boundary bounce, trace_prob tracing, dobs_state transitions, z-velocity asymmetry, anti-lock flip behavior, and boundary-touch perturbation.
- Test wall and bounds geometry by comparing expected hit distances in controlled scenes.
- Test LiDAR distance image generation against P2M reference outputs on fixed drone, obstacle, and wall states.
- Test analytic_lidar_grad for finite, nonzero gradients from LiDAR distance image through sensor geometry to relevant environment state in non-degenerate scenes.
- Test stop_lidar_grad for identical forward observations and intentionally absent sensor-to-state gradients, while preserving policy-action-to-dynamics/reward gradients.
- Test observation builder shape and normalization for LiDAR distance image + target direction + velocity + last_action.
- Test reward components separately: goal progress, safety proxy, dynamic-obstacle proxy, smoothness/control cost, and termination-triggered behavior.
- Test termination behavior for wall/bounds violation, height violation, excessive velocity or acceleration, obstacle proximity, NaN state, and time truncation.
- Test BPTT and SHAC smoke training for compile success, finite metrics, and nonzero policy gradients through the environment.
- Treat BPTT as the first required end-to-end training smoke test; SHAC smoke tests can follow after the BPTT path is validated.
- Prior art in the codebase includes existing environment reset/step patterns, sensor projection logic, BPTT/SHAC training loops, and module-level policy networks.

## Out of Scope

- Migrating P2M PPO training logic.
- Migrating ROS or C++ raycast into the training path.
- Migrating NeuFlow as an external inference model into the training path.
- Migrating P2M's acceleration controller as the first action interface.
- Using flatten LiDAR + MLP as the first policy architecture.
- Treating stop_lidar_grad as the only visual-gradient strategy.
- Requiring SHAC as the first end-to-end training acceptance path.
- Claiming real-world transfer or final performance superiority before numerical alignment and training validation pass.
- Introducing a parallel P2M-specific training API outside Flightning Env API.

## Further Notes

- PRD scope is proposal-specific. Global terminology and resolved ambiguity belong in CONTEXT.
- Collision reward handling is now a follow-up design issue: hard collision/termination remains discrete, while reward uses continuous or piecewise-differentiable safety/dobs proxies; exact proxy formulas require a dedicated review.
- Observation shape is confirmed as LiDAR distance image + target direction + velocity + last_action. last_action means the previous Flightning low-level action, not P2M's acceleration command.
- Dynamic obstacle strategy is confirmed as boundary bounce plus trace_prob drone tracing with explicit JAX PRNG. The user-confirmed dobs_state state machine and related anti-lock details must be source-aligned before implementation.
- Policy architecture is confirmed as CNN LiDAR policy. P2M uses convolution over the LiDAR observation before fusing state features; this is the selected first-version architecture.
- Visual gradient strategy is confirmed as dual-mode: default analytic_lidar_grad with stop_lidar_grad as a stability control. The analytic path follows the documented DiffAero-style analytical raycasting direction; the stop-gradient control follows the documented D.VA-style decoupling direction and JAX `stop_gradient`.
- dobs_state state machine details are intentionally source-pending. The user will provide the source later; until then, they are user-confirmed target semantics, not current P2M main-branch code facts.
nalytic_lidar_grad with stop_lidar_grad as a stability control. The analytic path follows the documented DiffAero-style analytical raycasting direction; the stop-gradient control follows the documented D.VA-style decoupling direction and JAX `stop_gradient`.
- dobs_state state machine details are intentionally source-pending. The user will provide the source later; until then, they are user-confirmed target semantics, not current P2M main-branch code facts.
