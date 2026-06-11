## Why

P2M 的无人机动态避障任务（基于 LiDAR 图像观测躲避动态障碍物并飞向目标）目前运行在 Isaac Sim + Torch + ROS + NeuFlow 的训练栈上，无法利用 Flightning 的 JAX 可微分仿真器进行 BPTT/SHAC 端到端梯度训练。将 P2M 的任务语义迁移到 Flightning 的纯 JAX/JIT/scan 环境，可以解锁可微分强化学习在动态避障场景中的应用，同时摆脱对 Torch、ROS 和 NeuFlow 外部推理的依赖。

本 change 的核心原则是：**Flightning EnvState 是运行时状态真相**。无人机动力学、动态障碍物位置/速度更新、reward/termination 都由 Flightning 纯 JAX 代码定义。MuJoCo 只用于给 MuJoCo-LiDAR 的 JAX backend 提供静态几何 schema 和 geom metadata；不接入 MuJoCo 的 `mjData` 作为训练运行时状态，也不使用 MuJoCo 正向运动学或 `mj_step`。物理世界里无人机和障碍物的运动积分全部由 Flightning 自己用 JAX 执行，例如 `dobs_pos = dobs_pos + vel * dt`。

由于当前无人机和障碍物没有 MJCF/Isaac USD 模型，本 change 将新增 Rerun 调试级仿真可视化：直接从 Flightning EnvState 渲染无人机姿态、动态障碍物、墙体、目标、轨迹和 LiDAR 射线/命中点，用于观察“一个无人机在障碍物中飞行”的训练状态，而不把 Rerun 可视化资产作为动力学或传感器语义来源。

## What Changes

- **新增动态避障环境**：实现 `DynamicAvoidanceEnv` 继承 Flightning `Env` 基类，包含 JAX pytree 兼容的 `EnvState`、`reset`/`_step` 接口、quadrotor 动力学集成、reward 和 termination 逻辑
- **新增动态障碍物场模块**：严格复刻 P2M 动态障碍物语义。障碍物状态为位置、速度、半径和命中/状态槽；初始化采样位置、速度模长/方向和半径；每步先按 P2M 的触边规则更新速度，再执行 `pos_xy = pos_xy + vel_xy * dt`，高度固定为 `dobs_height / 2`。`trace_prob` 行为严格按 P2M 源码复刻，即触边时使用 `random >= trace_prob` 选择追踪无人机方向，否则反向/反射；文档会标注该字段名和行为存在直觉差异。
- **新增 MuJoCo-LiDAR 传感器集成**：在 `flightning/sensors` 下直接使用 `MjLidarJax` 的数组接口作为纯 JAX 解析 raycasting 引擎。传感器通过 JAX 手动计算世界位姿挂载在无人机上（`sensor_pos = drone.p + R_yaw @ offset_B`），射线方向仅跟随 yaw 旋转（`attach_yaw_only=True`，与 P2M 一致，避免 roll/pitch 导致 LiDAR 图像抖动）。传感器不调用 `MjLidarWrapper.trace_rays(mj_data, ...)` 的运行时路径，而是由 Flightning state 构造 `geom_xpos`、`geom_xmat`、`sensor_pos`、`sensor_mat` 后直接调用 JAX backend。提供三种扫描模式：
  1. **`p2m_oversample`（默认，照搬 P2M）**：108×18 = 1944 条超分射线 → MjLidarJax 求交 → Z 过滤 → 3×3 min-pool 降采样 → 36×6 distance image → 反转（`lidar_range - raw`）→ `(1, 36, 6)` occupancy-like 输出
  2. **`mid360_livox`（真实 Mid-360 扫描模式）**：使用 `LivoxGenerator` 从 `mid360.npy` 预计算扫描模式中采样 ~24000 条非重复 Lissajous 射线 → MjLidarJax 求交 → 投射到规则 bin 网格 → 输出 `(1, 36, 6)` 或更密网格。更接近真实硬件但计算量更大
  3. **`mid360_binned`（Mid-360 FOV + 规则网格）**：使用 Mid-360 的 FOV 参数（360° H, [-7°, 52°] V）但保持规则超分网格（同模式 1），兼顾真实 FOV 和 CNN 友好的图像结构。可作为模式 1 和模式 2 之间的折中
- **新增观测构建器**：融合 LiDAR distance image `(1, 36, 6)`（单通道 scan，不含 NeuFlow 光流）+ 目标相对方向 `(3,)` + 速度 `(3,)` + `last_action` `(4,)` 为策略观测
- **新增 CNN LiDAR 策略网络**：卷积编码 LiDAR distance image，与目标方向、速度、last_action 状态特征融合后输出 collective thrust + 3D body rates
- **新增 LiDAR 解析梯度路径**：`analytic_lidar_grad` 为默认训练路径，允许梯度从 LiDAR 观测经 `MjLidarJax` 和传感器几何回到 Flightning `EnvState`，用于承接 BPTT/SHAC 等传统可微仿真训练算法。`stop_lidar_grad` 仅作为稳定性对照配置，不代表 D.VA 训练路线。
- **新增 P2M 数值对齐验证**：在固定 drone/obstacle/wall 状态下，验证 JAX LiDAR distance image、动态障碍物更新、reward 分量与 P2M 参考输出的数值容差
- **新增 BPTT 端到端 smoke test**：验证 DynamicAvoidanceEnv 可通过 BPTT 完成 compile、rollout、loss 计算和参数更新，产生有限且非零梯度
- **新增 Arena XML 场景描述**：在 `flightning/sensors` 下创建 `avoidance_arena.xml`，声明 LiDAR raycast 需要的静态 geom schema。第一版只承诺 MuJoCo-LiDAR JAX backend 已支持且能由 Flightning state 直接提供 pose 的几何体，默认覆盖墙体 box 和动态障碍物 cylinder。HFIELD 可作为 backend 已有能力保留但不作为本 change 验收目标；mesh 和任意 MJCF kinematics 不纳入本 change。
- **新增 Rerun 调试级仿真可视化**：从 EnvState 直接记录无人机姿态、机体坐标轴/简化四旋翼代理、动态障碍物 cylinder、墙体 box、目标点、轨迹和 LiDAR rays/hits。Rerun 可视化资产只用于观察和调试，不参与训练、不参与 LiDAR 碰撞语义、不作为状态真相。
- **不迁移**：P2M 的 PPO 训练逻辑、Torch tensor 操作、ROS raycast C++ 节点、NeuFlow 外部推理、acceleration controller、flatten LiDAR + MLP 策略、MuJoCo `mjData` 运行时同步、MuJoCo/Isaac 级 photorealistic renderer

## Capabilities

### New Capabilities

- `dynamic-avoidance-env`: 动态避障环境主体，实现 Flightning Env API（reset/_step/step），管理 EnvState pytree（drone/target/obstacle/wall 状态、last_actions、time、step_idx），集成 quadrotor 动力学，定义 reward（goal progress + safety proxy + dobs proxy + smoothness/control）和 termination（wall/bounds violation、height violation、excessive velocity、obstacle proximity、NaN state、time truncation）
- `dynamic-obstacle-field`: 动态障碍物场模块，负责障碍物初始化和每步运动更新。状态结构为 `pos_xy (N, 2)`、`vel_xy (N, 2)`、`radius (N,)`、`hit/state (N,)`；默认复刻 P2M 的 `dynamic_obs_num=40`、`pos_xy ∈ [-18, 18]^2`、`vel_norm ∈ [1, 5]`、`radius ∈ [0.25, 0.45]`、`dobs_height=4.0`。触边时按 P2M 的 `random >= trace_prob` 逻辑选择追踪有效无人机位置或反射速度，使用 JAX PRNG 显式驱动。
- `mujoco-lidar-sensor`: LiDAR 传感器集成层。传感器挂载在无人机上，通过 JAX 计算世界位姿（`sensor_pos = drone.p + R_yaw @ offset_B`），射线方向仅跟随 yaw 旋转（`attach_yaw_only=True`，提取 yaw 角构建旋转矩阵）。直接调用 `MjLidarJax.trace_rays` / `trace_rays_batch` 的数组接口，输入由 Flightning state 构造的 `geom_xpos`、`geom_xmat`、`sensor_pos`、`sensor_mat`。提供三种扫描模式：(1) `p2m_oversample`（默认）— 照搬 P2M 的 1944 射线超分管线：108×18 射线 → MjLidarJax cylinder/box raycasting → Z 高度过滤 → 3×3 min-pool → 36×6 → 反转 → `(1, 36, 6)`；(2) `mid360_livox` — 使用 LivoxGenerator 真实 Mid-360 非重复扫描模式（~24000 射线 + bin 投射）；(3) `mid360_binned` — Mid-360 FOV + 规则超分网格折中。第一版只承诺 JAX backend 已支持且能由 Flightning state 直接提供 pose 的几何体。支持 `analytic_lidar_grad`/`stop_lidar_grad` 双梯度模式。
- `observation-builder`: 观测构建器，将 LiDAR distance image、目标相对方向、自身速度和 last_action 融合为策略观测向量
- `cnn-lidar-policy`: CNN LiDAR 策略网络，接收 `(1, 36, 6)` 单通道 LiDAR distance image（第一版不含 NeuFlow 光流，区别于 P2M 的 3 通道输入），经卷积编码后与目标方向、速度、last_action 等状态特征融合，输出 low-level action（collective thrust + 3D body rates）
- `visual-gradient-modes`: 视觉梯度控制模式，实现 analytic_lidar_grad（默认，JAX 解析可微 raycasting）和 stop_lidar_grad（stop_gradient 阻断 sensor-to-state 梯度）两种配置，服务于 BPTT/SHAC 可微仿真路线。D.VA 算法作为独立 OpenSpec change 处理，不属于本 change 第一版范围。
- `rerun-debug-visualization`: Rerun 调试级仿真可视化，将 EnvState 转换为可视化 primitives。无人机使用简化四旋翼代理、body axes 和轨迹；动态障碍物使用 cylinder 代理；墙体使用 box 代理；目标使用 point/sphere 代理；LiDAR 可显示 rays、hit points 和 scan image。该能力只用于观察 Flightning state，不提供 physics、kinematics 或 LiDAR 碰撞语义。
- `p2m-alignment-validation`: P2M 数值对齐验证框架，包括 LiDAR distance image 对齐、动态障碍物更新规则对齐、reward 分量对齐，以及 BPTT smoke training 验收

### Modified Capabilities

无现有 spec 需要修改。本迁移为全新能力引入，不影响 Flightning 现有的 hovering 环境或训练算法（bptt/ppo/shac）的已有行为。

## Impact

- **新增文件**：
  - `flightning/envs/dynamic_avoidance_env.py` — 环境主体
  - `flightning/sensors/mujoco_lidar_sensor.py` — LiDAR 传感器包装
  - `flightning/sensors/avoidance_arena.xml` — Arena 场景描述
  - `flightning/modules/dynamic_obstacle_field.py` — 动态障碍物场
  - `flightning/modules/observation_builder.py` — 观测构建器
  - `flightning/modules/cnn_lidar_policy.py` — CNN LiDAR 策略网络
  - `flightning/visualization/rerun_dynamic_avoidance.py` — Rerun 调试级仿真可视化
  - `tests/test_p2m_alignment.py` — P2M 数值对齐测试
  - `tests/test_dynamic_avoidance_env.py` — 环境单元测试
  - `tests/test_lidar_sensor.py` — LiDAR 传感器测试
  - `tests/test_rerun_visualization.py` — Rerun 可视化适配器 smoke test（不要求启动 viewer）
  - `examples/train_bptt_avoidance.ipynb` — BPTT 训练 notebook
- **新增依赖**：`mujoco-lidar` (MuJoCo-LiDAR JAX 后端)、`mujoco`（仅用于 XML 解析和 geom 参数提取）、`rerun-sdk`（可视化，可选）
- **受影响的现有代码**：无破坏性变更。新增环境独立于现有 hovering 环境，不修改 `env_base.py`、`hovering_state_env.py` 或 `algos/` 模块
- **训练算法兼容性**：新环境通过 Flightning Env API 暴露，BPTT/SHAC/PPO 可直接调用 `env.reset`/`env.step`，无需特殊适配器
- **计算资源**：JAX JIT 编译 + vmap 批量并行，GPU 上大批量 rollout 的预期吞吐与现有 hovering 环境相当
