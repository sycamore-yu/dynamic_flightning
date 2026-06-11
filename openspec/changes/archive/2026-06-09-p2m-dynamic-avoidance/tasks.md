## 1. 基线与工程入口

- [x] 1.1 核对 `flightning/envs`、`flightning/modules`、`flightning/sensors`、`flightning/algos`、`examples` 和 `tests` 的现有导出方式，记录本次新增 file 需要接入的位置。
- [x] 1.2 在修改任何现有函数、类、方法或导出符号前运行 GitNexus upstream impact analysis，并只对本 change 需要的最小导出点做改动。
- [x] 1.3 添加 `mujoco-lidar`、`mujoco` 和可选 `rerun-sdk` 依赖入口；Rerun 缺失时训练路径不得失败。
- [x] 1.4 建立 P2M 参考 fixture，固定动态障碍物、LiDAR 后处理和 reward 分量的确定性输入/输出。
- [x] 1.5 新增 `avoidance_arena.xml`，只声明第一版训练 LiDAR 需要的墙体 box 和动态障碍物 cylinder 静态几何 schema。

## 2. 动态障碍物场

- [x] 2.1 新增 `flightning/modules/dynamic_obstacle_field.py`，定义 JAX pytree 兼容的配置和状态：`pos_xy`、`vel_xy`、`radius`、`hit/state`。
- [x] 2.2 实现 reset 采样，默认匹配 P2M：`dynamic_obs_num=40`、`pos_xy in [-18,18]^2`、`vel_norm in [1,5]`、`radius in [0.25,0.45]`、`dobs_height=4.0`。
- [x] 2.3 实现每步更新：边界触碰时先按 P2M `random >= trace_prob` 选择追踪或反射速度，再执行 `pos_xy += vel_xy * dt`。
- [x] 2.4 暴露动态障碍物 cylinder pose 构造接口，中心高度为 `dobs_height / 2`，供 LiDAR 几何和 Rerun 可视化复用。
- [x] 2.5 添加动态障碍物单元测试，覆盖确定性 PRNG、边界反射、`trace_prob` 反直觉语义和 cylinder pose。

## 3. MuJoCo-LiDAR 传感器路径

- [x] 3.1 新增 `flightning/sensors/mujoco_lidar_sensor.py`，从静态 `mujoco.MjModel` 初始化 `MjLidarJax` metadata，但 rollout 中不读取或写入 `mjData`。
- [x] 3.2 实现 `LidarGeometryAdapter`，从 Flightning EnvState 派生墙体 box 与障碍物 cylinder 的 `geom_xpos` 和 `geom_xmat`。
- [x] 3.3 实现 `LidarMount`，从无人机位置和 yaw-only 姿态构造 `sensor_pos` 和 `sensor_mat`，默认不随 roll/pitch 旋转。
- [x] 3.4 实现默认 `p2m_oversample` 扫描：`108 x 18` 射线、Z 过滤、`3x3` min-pool、`lidar_range - distance`，输出 `(1, 36, 6)`。
- [x] 3.5 实现 `analytic_lidar_grad` 和 `stop_lidar_grad` 配置；默认允许 JAX 梯度经 `MjLidarJax` 和传感器几何回到 EnvState。
- [x] 3.6 对 `mid360_livox` 和 `mid360_binned` 提供显式 future-scope 行为，第一版可报 unsupported，不纳入验收测试。
- [x] 3.7 添加 LiDAR 测试，覆盖无 `mjData` 调用、yaw-only 挂载、输出 shape、P2M 后处理和 targeted gradient smoke。

## 4. 观测构建与策略网络

- [x] 4.1 新增 `flightning/modules/observation_builder.py`，返回 flat `jax.Array`，默认布局固定为 `lidar_flat(216) + target_dir(3) + velocity(3) + last_action(4)`。
- [x] 4.2 为 `observation_space` 提供稳定 shape metadata，保证 `LogWrapper`、`VecEnv`、BPTT 和 SHAC 不需要 dict observation 支持。
- [x] 4.3 新增 `flightning/modules/cnn_lidar_policy.py`，将前 216 维 reshape 为 channel-first `(1, 36, 6)` 后做 CNN 编码。
- [x] 4.4 将 LiDAR CNN feature 与目标方向、速度、last_action 特征融合，输出 Flightning 4D action：collective thrust + 3D body rates。
- [x] 4.5 添加观测和策略测试，覆盖 observation split、flat shape、policy output shape 和正常 BPTT/SHAC 梯度语义；不得加入 D.VA stop-gradient 行为。

## 5. DynamicAvoidanceEnv

- [x] 5.1 新增 `flightning/envs/dynamic_avoidance_env.py`，定义 `DynamicAvoidanceEnvState`，以 Flightning EnvState 作为 drone、target、obstacle、wall、last_action、time、step_idx 的运行时真相。
- [x] 5.2 实现 `reset(key, state=None)`，初始化无人机、目标、动态障碍物、墙体、last_action，并返回 state 和 flat observation。
- [x] 5.3 实现 `_step(state, action, key)`，裁剪 Flightning 4D action，更新 last_action，调用现有 quadrotor JAX dynamics，并更新动态障碍物。
- [x] 5.4 在 `_step` 内调用 LiDAR sensor 和 ObservationBuilder，且不得使用 MuJoCo `mjData`、`mj_step` 或 Python-side mutable runtime state。
- [x] 5.5 实现默认可微训练 reward，包含 goal progress、speed band、height band、soft clearance、dynamic obstacle risk、action magnitude 和 action smoothness。
- [x] 5.6 实现 termination/truncation：wall/bounds、高度、速度、障碍物 proximity、NaN state 和 time horizon。
- [x] 5.7 添加环境测试，覆盖 `reset`/`step` API、JIT、vmap、`lax.scan`、action clipping、termination 和有限 reward。

## 6. Rerun 调试可视化

- [x] 6.1 新增 `flightning/visualization/rerun_dynamic_avoidance.py`，把 EnvState 和 sensor outputs 转换为 Rerun primitives。
- [x] 6.2 记录无人机简化代理、body axes、动态障碍物 cylinder、墙体 box、目标点、轨迹、LiDAR rays/hits 和 scan image。
- [x] 6.3 提供 headless `.rrd` 导出路径，适配远程服务器无 viewer 工作流。
- [x] 6.4 添加 Rerun smoke test，验证 logging/export 调用，不要求启动图形 viewer；Rerun 禁用或缺失时训练行为保持不变。

## 7. P2M 对齐验证

- [x] 7.1 新增 `tests/test_p2m_alignment.py`，使用固定状态比较动态障碍物初始化和更新语义。
- [x] 7.2 添加 `trace_prob` 边界样例，明确验证 P2M 的 `random >= trace_prob` trace-selection 行为。
- [x] 7.3 添加固定几何 LiDAR scan 对齐测试，比较 `(1, 36, 6)` `p2m_oversample` 输出和 P2M-derived reference 的容差。
- [x] 7.4 实现 P2M reward component 计算作为 alignment/evaluation logging，不替代默认 BPTT/SHAC 可微训练 reward。
- [x] 7.5 添加 BPTT 和 SHAC smoke validation，确认小 rollout 可 compile、step、loss/reward 计算，并产生有限 metrics；不要求 D.VA。

## 8. 示例、文档与最终校验

- [x] 8.1 新增 `examples/train_bptt_avoidance.ipynb` 或等价 headless 脚本，演示默认 `analytic_lidar_grad` 动态避障训练。
- [x] 8.2 新增可选 Rerun `.rrd` 导出示例，便于远程服务器离线查看 rollout。
- [x] 8.3 在配置或文档中说明 `analytic_lidar_grad` 是默认路线，`stop_lidar_grad` 是稳定性/消融对照，D.VA 属于 `add-dva-algorithm` change。
- [x] 8.4 运行动态避障相关测试、BPTT/SHAC smoke、`openspec validate p2m-dynamic-avoidance --strict`，记录任何跳过项和原因。
- [x] 8.5 运行 `gitnexus_detect_changes()` 检查影响范围，确认本 change 未意外修改现有 hovering 环境或 BPTT/SHAC/PPO 公共 API。
