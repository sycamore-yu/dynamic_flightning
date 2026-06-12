## Context

Dynamic avoidance training acceptance is being moved from the old notebook/Rerun-oriented path to a script-first D.VA validation path that runs on the remote headless `flightning` conda environment.

The previous `add-dva-algorithm` change added a generic D.VA algorithm, but the generic adapter currently maps only an environment observation into actor and critic inputs. That is not sufficient for dynamic avoidance D.VA because the actor-visible observation is a LiDAR/task feature vector, while the critic needs training-only privileged state.

Project terms used by this design:

- `actor-visible observation`: the observation available to the actor and deployment path. For current dynamic avoidance this is the flat LiDAR observation produced by `ObservationBuilder`.
- `privileged critic observation`: a deterministic, scaled observation derived from `DynamicAvoidanceEnvState` that is available during training for critic learning but not available to the actor/deployment path.
- `pre-reset terminal critic observation`: the privileged critic observation derived from the state produced by `_step` before `Env.step` auto-reset replaces done environments with reset observations.

Reference D.VA uses separate state and visual observations: actor input is detached visual or state observation, while the critic uses state observation. A visual/LiDAR D.VA path that silently uses actor-visible observations as critic observations would compromise that design. State-only D.VA remains valid when actor and critic intentionally use the same state observation.

## Goals / Non-Goals

**Goals:**

- Provide a complete dynamic avoidance D.VA validation design with script-first, headless commands.
- Require dynamic avoidance LiDAR D.VA to use privileged critic observations instead of visual/LiDAR fallback.
- Extend the D.VA adapter boundary so task-specific adapters can derive critic observations from `EnvState` without changing the actor observation layout.
- Fix D.VA done/truncated bootstrap handling so critic targets do not use auto-reset observations.
- Define that dynamic avoidance privileged critic observations use deterministic scaling for finite smoke validation.
- Align the default dynamic avoidance training reward with D.VA/BPTT needs by using smooth differentiable LiDAR clearance and object-free motion/TTC risk proxy terms while keeping hard termination separate.
- Keep optimizer policy explicit in the training script while preserving caller-owned `TrainState` construction.

**Non-Goals:**

- Do not productize `mid360_livox` or any other LiDAR scan mode in this change.
- Do not require notebook execution as the primary acceptance path.
- Do not claim long-training convergence from finite smoke metrics.
- Do not claim true D.VA actor policy convergence, successful dynamic obstacle avoidance, or goal-reaching behavior from the headless finite-metrics validation.
- Do not add generic `RunningMeanStd`, return normalization, or reward scaling to D.VA core in this change.
- Do not replace `compute_p2m_reward` as the P2M alignment/logging function; this change affects the default differentiable training reward used by `_get_reward_jit`.
- Do not change BPTT, SHAC, or PPO public APIs except where a shared bootstrap bug fix is intentionally factored later.

## Current Validation Boundary

This change completes D.VA headless smoke validation for dynamic avoidance: script-first execution, finite actor/critic metrics, privileged critic observations, done/truncated bootstrap handling, and numerically stable reward proxy tests.

The validation does not prove that the trained actor policy has learned effective dynamic obstacle avoidance or goal-reaching. The observed trained-policy rollouts still terminate early or fly largely along the direct goal path without demonstrated obstacle-aware maneuvering. The policy convergence gap is tracked in `docs/issues/005-dva-policy-convergence-gap.md`.

## Decisions

### Decision 1: Dynamic avoidance D.VA requires privileged critic observations

**Decision:** Dynamic avoidance visual/LiDAR D.VA SHALL fail loudly if no privileged critic observation is provided.

**Rationale:** Reference D.VA uses `vis_obs` or `state_obs` for the actor but always trains the critic from `state_obs`. Falling back from a LiDAR actor observation to a LiDAR critic observation would turn the method into a different algorithm. The only same-observation case allowed by this design is state-only D.VA, where both actor and critic intentionally use state observations.

**Alternatives considered:**

- Allow critic fallback to actor observation: rejected because it compromises D.VA's asymmetric actor/critic design.
- Require all Flightning environments to return structured `{"actor_obs", "critic_obs"}` observations: rejected because it would disrupt existing actor-visible observation contracts.

### Decision 2: Extend the adapter boundary instead of changing `DynamicAvoidanceEnv` actor observations

**Decision:** D.VA's adapter contract SHALL be extended so task adapters can inspect the current observation, environment state, and step/reset metadata when constructing `DVAObservation`.

The intended conceptual contract is:

```python
adapter(obs, env_state, info) -> DVAObservation(actor_obs=..., critic_obs=...)
```

The exact implementation may preserve compatibility for state-only callers, but dynamic avoidance adapters must be able to derive `critic_obs` from `DynamicAvoidanceEnvState`.

**Rationale:** Privileged state lives in `DynamicAvoidanceEnvState`, not in the 226-dimensional actor-visible observation. The adapter is the correct ownership boundary: it defines task-specific observation mapping while D.VA core owns rollout, actor gradient semantics, critic targets, and metrics.

**Alternatives considered:**

- Return structured observations from `DynamicAvoidanceEnv`: rejected because existing BPTT/SHAC/CNN policy paths expect the current flat actor observation.
- Store critic observations only in `EnvTransition.info`: rejected as the primary API because it makes the critic contract implicit and fragile across wrappers.

### Decision 3: D.VA rollout must preserve pre-reset terminal critic observations

**Decision:** Dynamic avoidance D.VA rollout SHALL not rely on `Env.step`'s post-reset observation for bootstrap values. It must preserve pre-reset terminal state or pre-reset terminal critic observation before applying reset selection.

Early termination bootstrap policy:

- If `terminated=True` before the time limit, bootstrap value is zero.

Time-limit truncation bootstrap policy:

- If `truncated=True` because the horizon/time limit is reached, bootstrap may use the pre-reset terminal privileged critic observation.
- D.VA SHALL primarily trust the environment-provided `truncated` flag and MAY also use `step_idx >= max_steps_in_episode` as a dynamic-avoidance-specific guard when constructing tests or terminal bootstrap metadata.

**Rationale:** `Env.step` auto-resets done environments and returns reset observations. Using those observations in TD-lambda or one-step targets leaks a new episode's initial value into the previous episode's target. Reference D.VA avoids this by using `state_obs_before_reset` and explicitly handling early termination versus time-limit endings.

**Alternatives considered:**

- Keep using `trans.obs` and mask all done bootstraps to zero: rejected because it loses valid time-limit bootstrap behavior.
- Change `Env.step` globally to return pre-reset observations: rejected as too broad for this migration.

### Decision 4: Use deterministic scaling for first dynamic avoidance privileged critic observations

**Decision:** The first dynamic avoidance privileged critic observation SHALL use deterministic scaling based on known environment and action bounds.

The schema SHALL avoid object-level dynamic obstacle lists. Candidate components include:

- ego-centric quadrotor state, including normalized height, body-frame velocity, yaw/heading encoding, and normalized angular velocity;
- target information, preferably relative to the drone where appropriate;
- object-free clearance fields derived from LiDAR/geometry sectors;
- object-free clearance motion fields such as sector-wise clearance delta and optional TTC;
- last action and episode progress where useful for value estimation.

The first main schema is the 127-dimensional ego-centric object-free clearance-motion critic observation defined in the capability spec.

`clearance_delta_field` history SHALL be owned by `DynamicAvoidanceEnvState`, not by the D.VA adapter. Reset initializes previous clearance from the reset scan so the first delta is zero. `_step` computes motion/TTC fields from the previous clearance and current clearance, then stores the current clearance into the next state for the following step.

**Rationale:** Reference D.VA supports running observation and return normalization, but implementing generic online RMS and return scaling is larger than the validation migration. Dynamic avoidance still needs a stable scale for critic inputs, so deterministic scaling is the smallest explicit contract that can support finite headless validation.

**Alternatives considered:**

- Add generic RunningMeanStd and return normalization now: rejected as a separate long-training stability feature.
- Use raw privileged state: rejected because field scales vary widely and would make finite smoke failures harder to interpret.

### Decision 5: Optimizer policy stays caller-owned

**Decision:** D.VA core SHALL NOT own learning rate schedule policy in this change. Dynamic avoidance D.VA scripts SHALL explicitly construct actor and critic `TrainState` objects and declare whether they use fixed learning rates or Optax schedules.

**Rationale:** Flightning algorithms receive caller-created `TrainState` objects. This already permits independent actor/critic optimizers and schedules. Moving LR schedule into D.VA core would conflict with this API style.

**Alternatives considered:**

- Add `actor_lr`, `critic_lr`, and schedule fields to `DVAConfig`: rejected because it duplicates optimizer ownership already held by `TrainState`.

### Decision 6: Keep LiDAR scan mode productization separate

**Decision:** This change SHALL use the current supported dynamic avoidance LiDAR observation path. `mid360_livox` and related scan-mode productization belong in a separate OpenSpec change.

**Rationale:** Dynamic avoidance D.VA validation already expands scope to privileged critic observations and bootstrap semantics. Adding sensor-mode productization would mix algorithm validation with LiDAR backend/API work.

### Decision 7: Update the default training reward with smooth clearance and object-free motion-risk proxies

**Decision:** The default differentiable training reward in `_get_reward_jit` SHALL use smooth LiDAR clearance and object-free motion/TTC risk proxy terms suitable for D.VA/BPTT optimization. Hard termination remains an episode-management signal and SHALL NOT be the primary differentiable collision penalty.

Safety proxy policy:

- Preserve `clearance` as the current geometry safety term computed from the LiDAR/raycast field. This term is class-agnostic and covers the nearest hit regardless of whether it comes from walls, static clutter, or dynamic obstacles.
- Replace hard-margin ReLU-squared clearance penalties with smooth penalties, preferably using `jax.nn.softplus`-style barriers rather than hand-written `log(exp(x) + 1)` forms.
- Preserve a weak but nonzero safety gradient outside the immediate collision margin so the optimizer receives earlier avoidance signals.

Object-free motion-risk proxy policy:

- Remove object-level `dobs_risk` from the default reward.
- Add `motion_risk` and/or `ttc_risk` computed from object-free clearance motion fields, such as sector-wise `delta_d = (d_t - d_{t-1}) / dt` and `ttc = d_t / max(-delta_d, eps)`.
- Align the reward's motion-risk primitive with the privileged critic schema's object-free `clearance_delta_field` and `ttc_field`.
- Do not use dynamic obstacle object identity, object ordering, radius fields, or object velocity labels as reward inputs for the main validation path.
- First-version constants SHALL be caller-visible configuration fields rather than hard-coded locals. The conservative starting point is `clearance_margin = 1.5`, `barrier_temperature = 0.25`, `ttc_horizon = 3.0`, `clearance_weight = 5.0`, and `motion_risk_weight`/`ttc_risk_weight` in the same scale as the removed `dobs_risk` weight.

Hard event policy:

- `terminated` and collision predicates are used for environment truncation and value bootstrapping semantics.
- If a fixed collision event penalty is added to reward, its event indicator SHALL be wrapped with `jax.lax.stop_gradient`.

Numerical-stability policy:

- All norm divisions SHALL include epsilon guards.
- Any inverse-trigonometric fallback, if used, SHALL clip its input away from exact `-1.0` and `1.0`.
- Reward tests SHALL assert finite reward and finite gradients in representative near-boundary scenes.

**Rationale:** The PRD requires hard termination and reward semantics to be documented separately, and requires continuous or piecewise-differentiable safety/dynamic-clutter proxy rewards. The current ReLU-squared clearance terms give no obstacle-avoidance gradient outside the chosen margin. Object-level `dobs_risk` would train against simulator-provided dynamic obstacle labels that the actor cannot observe at deployment time. Object-free clearance motion and TTC fields preserve the P2M-style preference for pixel/point-level environmental change over object detection, tracking, and prediction shortcuts.

**Alternatives considered:**

- Copy `compute_p2m_reward` directly into `_get_reward_jit`: rejected because `compute_p2m_reward` is an alignment/logging function and currently contains angle logic that needs additional stability review before becoming the default training objective.
- Keep object-level `dobs_risk` based on dynamic obstacle positions, velocities, and radii: rejected for the main validation path because it encodes simulator object labels rather than object-free clutter motion.
- Use `arccos(cos_theta)` as the main collision-cone primitive: rejected for the first version because it is object-level and has difficult gradients near its domain boundaries.
- Keep current ReLU-squared margins: rejected because they provide no anticipatory obstacle signal outside the margin.

## Risks / Trade-offs

- Privileged critic schema may become too large or poorly scaled -> Mitigation: define a deterministic field order and scaling policy, then test shape, finiteness, and JAX transform compatibility.
- D.VA rollout changes may affect existing state-only D.VA tests -> Mitigation: preserve a compatibility path for state-only adapters and keep focused regression tests.
- Bootstrap semantics may interact with `LogWrapper`/`VecEnv` behavior -> Mitigation: add tests that distinguish early termination, time-limit truncation, pre-reset terminal critic obs, and auto-reset obs.
- Finite smoke metrics may be mistaken for training convergence -> Mitigation: specs and scripts state that this validates headless execution and finite metrics only.
- Smooth clearance penalties may over-penalize safe trajectories if margins or temperatures are poorly chosen -> Mitigation: expose constants in one local reward helper and test representative near/far obstacle scenes.
- Object-free motion/TTC risk terms may become noisy when sector minima switch between hits or when clearance delta is near zero -> Mitigation: use smooth sector aggregation where feasible, clip TTC to a fixed horizon, guard denominators with epsilons, and test finite gradients for near-zero approach rates.

## Migration Plan

1. Finalize the privileged critic observation schema and deterministic scaling policy.
2. Extend the D.VA adapter contract while preserving state-only compatibility.
3. Update D.VA rollout sampling to preserve pre-reset terminal critic observations for target construction.
4. Update the default dynamic avoidance training reward to use smooth LiDAR clearance and object-free motion/TTC risk proxies.
5. Add dynamic avoidance D.VA training script and tests for finite actor/critic metrics.
6. Add bootstrap-specific tests for early termination and time-limit truncation behavior.
7. Add reward proxy tests for finite values and finite gradients.
8. Run headless validation commands in the `flightning` conda environment.
9. Validate the OpenSpec change with `openspec validate replace-avoidance-training-validation-with-dva --strict`.

## Open Questions

No implementation-blocking design questions remain.
