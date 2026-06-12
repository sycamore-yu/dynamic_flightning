## ADDED Requirements

### Requirement: Privileged Critic Observation Uses Ego-centric Object-free Clearance-Motion Schema
Dynamic avoidance D.VA validation 的 privileged critic observation SHALL 使用 ego-centric、object-free 的 clearance-motion schema。该 schema SHALL 表达空间格子的距离、距离变化和粗略 time-to-collision，而不是表达仿真器对象列表中的第几个动态障碍物位置、速度或半径。

该 critic schema 的目标是支持通用 dynamic clutter 避障：critic 可以使用 training-only processed state，但该 state 的语义必须贴近 actor 可通过 LiDAR 感知到的空间变化，避免通过 `DynamicObstacleFieldState` 的对象级标签给 actor update 引入过强的仿真器 object shortcut。

#### Scenario: Ego-centric base state
- **WHEN** constructing the critic observation
- **THEN** it SHALL include a 10-dimensional `ego_state` containing normalized height, body-frame linear velocity, body-frame heading/yaw encoding, and normalized angular velocity.
- **AND** it SHALL include a 4-dimensional `goal_state` containing the body-frame unit vector to the target and the normalized/clipped goal distance.

#### Scenario: Object-free clearance field
- **WHEN** building the spatial obstacle representation
- **THEN** it SHALL construct a 36-dimensional `clearance_field` from the current LiDAR/geometry field using `12` horizontal sectors × `3` height bands.
- **AND** each cell SHALL represent the nearest obstacle clearance in that direction/height band, without classifying the hit as static or dynamic.
- **AND** it SHOULD use smooth minimum aggregation such as log-sum-exp or softmin within sectors to preserve spatial gradients and stable clearance semantics.

#### Scenario: Object-free clearance motion field
- **WHEN** building the temporal obstacle representation
- **THEN** it SHALL construct a 36-dimensional `clearance_delta_field` over the same sectors as `clearance_field`.
- **AND** each cell SHALL represent normalized clearance change `delta_d = (d_t - d_{t-1}) / dt`.
- **AND** negative `delta_d` SHALL indicate that the nearest occupied geometry in that cell is approaching, while positive `delta_d` SHALL indicate that the cell is becoming more open.
- **AND** this representation SHALL NOT depend on dynamic obstacle object identity, object ordering, object radius fields, or object velocity labels.

#### Scenario: Optional object-free time-to-collision field
- **WHEN** `ttc_field` is enabled for the first validation schema
- **THEN** it SHALL construct a 36-dimensional `ttc_field` using `ttc = d_t / max(-delta_d, eps)`.
- **AND** `ttc` SHALL be clipped to a fixed horizon such as `[0, 3s]` and deterministically normalized.
- **AND** cells with non-approaching motion SHALL map to the maximum clipped horizon rather than an infinite or NaN value.

#### Scenario: Full schema assembly
- **WHEN** assembling the final `critic_obs`
- **THEN** the main validation schema SHALL concatenate `ego_state` (10), `goal_state` (4), `clearance_field` (36), `clearance_delta_field` (36), `ttc_field` (36), `last_action` (4), and episode `progress` (1).
- **AND** the resulting main schema SHALL be a 127-dimensional vector.
- **AND** if `ttc_field` is explicitly disabled for an ablation, the schema SHALL remain explicitly named and SHALL NOT be confused with the main validation schema.

#### Scenario: Ablation configurations
- **WHEN** running experiments to validate the components of the privileged schema
- **THEN** ablation configurations SHALL be supported to evaluate the impact of these features.
- **AND** ablations MAY include an explicitly named non-acceptance baseline without privileged states (`critic_obs = actor_obs`) and intermediate configurations without `clearance_delta_field` or without `ttc_field`.
- **AND** the main dynamic avoidance D.VA validation path SHALL NOT treat `critic_obs = actor_obs` as a fallback when privileged critic observation construction is missing.
