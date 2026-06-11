# Flightning 可微分无人机训练

本文档定义将 LiDAR 图像动态避障任务迁移到 Flightning 的 JAX 可微分仿真训练栈时使用的稳定术语和边界。

## 语言

**可微分强化学习**:
通过可微分仿真器 rollout 反向传播来优化策略参数的训练方式，本项目中具体指 BPTT 和 SHAC。
_避免_: 用 DiffRL 指代 PPO 或泛化的强化学习。

**BPTT**:
通过完整或配置长度的 rollout horizon 反向传播来优化策略参数的可微分训练算法。
_避免_: model-free RL。

**SHAC**:
通过短 horizon 仿真 rollout 反传 actor 梯度，并训练 critic 做 bootstrap 的可微分 actor-critic 算法。
_避免_: 不依赖仿真器梯度的 PPO 式 policy-gradient 采样。

**PPO**:
一种 model-free 强化学习基线，在迁移讨论中必须与可微分强化学习分开处理。
_避免_: DiffRL、可微分算法。

**P2M 动态避障任务**:
源任务族，含义是无人机使用 LiDAR 派生的图像观测，在飞向目标的同时躲避动态障碍物。
_避免_: 把迁移理解为只替换 controller。

**LiDAR 图像观测**:
由 LiDAR ray 距离或 distance-map 特征派生出的稠密图像式观测。
_避免_: raw point cloud、camera RGB image。

**分段可微 LiDAR 几何**:
LiDAR ray hit 几何可以包含 min、clip、有效性 mask、碰撞阈值等分段定义，但训练主路径必须保持 JAX/JIT 兼容并尽量保留 action 到动力学和主要 reward 的梯度。
_避免_: Torch/ROS/外部模型参与训练主路径。

**任务语义迁移**:
迁移 P2M 动态避障任务的目标、观测、障碍物、奖励、终止和碰撞语义，而不是迁移源仓库的训练机制。
_避免_: 复制 Torch/ROS/NeuFlow 训练链路。

**Flightning Env API**:
迁移后任务对训练算法暴露的唯一接口，语义由环境定义，训练算法只通过 `reset`、`_step`、`step`、观测空间和动作空间交互。
_避免_: 为 P2M 单独引入并行训练接口。

**低层动作接口**:
迁移后第一版策略动作采用 Flightning 现有低层控制语义，即 4 维 action 表示 collective thrust 和 3 维 body rates。
_避免_: 第一版同时迁移 P2M 的 acceleration controller。

**last_action**:
上一时刻实际进入 Flightning 环境状态的 4 维低层动作，即 collective thrust 和 3 维 body rates；它在迁移观测中承担 P2M `acc_input` 的“上一控制输入反馈”角色。
_避免_: P2M acceleration command、当前时刻尚未执行的 policy 输出。

**LiDAR Ray 方向生成器**:
根据 LiDAR FOV、分辨率和采样倍率生成固定 ray direction 网格的传感器组件。
_避免_: 放在训练算法或任务 reward 内部。

**动态障碍物状态机**:
动态障碍物用于表达边界反弹、trace_prob 追踪无人机、触边小扰动、防锁死翻转和命中状态的离散模式。
_避免_: 把触边位置小扰动理解为完整重采样。

**CNN LiDAR 策略**:
先用卷积编码 LiDAR distance image，再与目标方向、速度和 last_action 等状态特征融合后输出低层动作的策略网络。
_避免_: 第一版使用 flatten LiDAR + MLP 作为主策略结构。

**解析 LiDAR 梯度**:
LiDAR distance image 由 JAX 解析 raycasting 生成，并允许梯度从观测经传感器几何回到环境状态。
_避免_: 把深度图只当作不可导的外部输入。

**阻断 LiDAR 梯度**:
LiDAR distance image 仍在前向传播中生成并输入策略，但使用 `stop_gradient` 阻断观测模型对环境状态的反向依赖，作为稳定性对照。
_避免_: 误以为阻断 LiDAR 梯度会阻断 policy action 到动力学和 reward 的所有梯度。

**PRD**:
某个具体提案的需求合同，描述目标、范围、组件、决策、测试和未决项。
_避免_: 把 PRD 当成全局术语表。

**CONTEXT**:
项目长期有效的领域语言和已解决歧义，供所有后续提案复用。
_避免_: 把一次提案的临时任务清单写成全局事实。

## 关系

- **P2M 动态避障任务**为策略输入产生**LiDAR 图像观测**。
- **可微分强化学习**在本项目中包括 **BPTT** 和 **SHAC**。
- **PPO**是独立基线，不定义可微分训练接口。
- **任务语义迁移**保留 P2M 的避障任务含义，但不保留 Torch、ROS 或 NeuFlow 训练机制。
- **Flightning Env API**是迁移后任务供 BPTT、SHAC 和其他算法调用的唯一训练接口。
- **分段可微 LiDAR 几何**服务于**可微分强化学习**，但不要求碰撞判定、遮挡选择或最近 hit 选择在所有边界处处光滑。
- **低层动作接口**由 Flightning 的 quadrotor 动力学执行，**P2M 动态避障任务**只定义目标、障碍物、观测和奖励语义。
- **last_action**是**低层动作接口**中的上一时刻动作，在观测中与 **LiDAR 图像观测**、目标方向和速度一起提供给策略。
- **LiDAR Ray 方向生成器**属于传感器层，应放入 Flightning sensors 领域，而不是作为某个训练算法私有逻辑。
- **动态障碍物状态机**属于任务语义，必须由 JAX PRNG 显式驱动以兼容 `jit` 和 `scan`。
- **CNN LiDAR 策略**消费 **LiDAR 图像观测**和状态特征，输出**低层动作接口**动作。
- **解析 LiDAR 梯度**是第一版默认视觉梯度策略；**阻断 LiDAR 梯度**是稳定性对照。
- **CONTEXT**是全局语言；**PRD**是单个迁移提案的需求与验收范围。

## 示例对话

> **开发者:** "可以通过把 P2M 的 PPO policy 接到 Flightning 里完成迁移吗？"
> **领域专家:** "不可以。迁移后的任务必须通过 Flightning Env API 暴露 JAX 兼容的状态、观测、奖励和动力学，让 BPTT 或 SHAC 能够穿过 rollout 反传梯度。"

## 已标记歧义

- "DiffRL" 可能指 NVlabs 仓库、广义研究方向，或本项目的可微分算法。本文档中统一使用**可微分强化学习**指 BPTT/SHAC 这类依赖仿真器梯度的训练方式，不包括 PPO。
- "迁移 P2M" 可能指复制源实现，也可能指迁移任务语义。已决议：迁移**任务语义**，不迁移 P2M 的 Torch/ROS/NeuFlow 训练机制。
- "任务文档" 可能指全局上下文或具体提案。已决议：长期术语和边界写入 **CONTEXT**，P2M 动态避障迁移的范围和验收写入 **PRD**。

## 工作纪律

- 每次提出迁移设计问题前，必须先基于现有代码或 GitNexus 查询给出代码依据；不能只凭记忆或抽象最佳实践提问。
- 当现有代码和迁移设想冲突时，先指出冲突，再给推荐答案。
- 给出选项时必须同时说明每个选项的来源、优点、代价和代码依据；不能只给推荐项。

## 已确认迁移决策

- 第一版动作语义采用**低层动作接口**：collective thrust + 3 维 body rates。
- 第一版观测采用 LiDAR distance image + target direction + velocity + **last_action**；其中 **last_action** 是上一时刻低层动作，而不是 P2M acceleration command。
- **LiDAR Ray 方向生成器**放在 Flightning sensors 领域。
- LiDAR 参数按 P2M 默认训练配置迁移：range 10、水平分辨率 36、垂直分辨率 6、水平采样 3、垂直采样 3、水平 FOV 360、垂直 FOV [-7, 52]。
- 第一版同时实现 dynamic obstacles、walls 和 bounds，不拆成多个阶段。
- collision/termination 保持离散 hard event；reward 使用连续或分段可微的 safety/dobs proxy，并作为后续 issue 继续细化。
- 第一版策略结构采用 **CNN LiDAR 策略**，不采用 flatten LiDAR + MLP 作为主结构。
- 视觉梯度默认采用**解析 LiDAR 梯度**，同时保留**阻断 LiDAR 梯度**作为稳定对照；参考来源是 DiffAero 的解析可微 raycasting 思路、D.VA 的观测梯度解耦思路，以及 JAX `grad`/`stop_gradient`/`lax.scan` 官方机制。
- 动态障碍物迁移 P2M 的“边界反弹 + trace_prob 追踪无人机”语义，使用 JAX PRNG 显式实现，并保留 dobs_state 状态机、z 速度在反弹/追踪中的非对称处理、反弹后 state[2] 翻转的防锁死逻辑、触边位置小扰动语义。
- 迁移验证采用 P2M 数值对齐作为主要验收方式。
- 第一条端到端训练验收使用 BPTT；SHAC 后续仍需支持，但不是首个闭环验收目标。
- dobs_state 状态机等补充细节的具体源码来源暂缓，等待用户后续提供；在提供前不得把这些细节宣称为当前 P2M 主分支源码事实。
