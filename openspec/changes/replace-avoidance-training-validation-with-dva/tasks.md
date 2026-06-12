## 1. Migration Contract

- [x] 1.1 Identify the current dynamic avoidance training validation path that depends on legacy notebook execution.
- [x] 1.2 Define the D.VA-based headless validation command set for dynamic avoidance training.
- [x] 1.3 Define the dynamic avoidance privileged critic observation schema and deterministic scaling policy.
- [x] 1.4 Define the D.VA rollout done/truncated bootstrap contract, including early termination and time-limit truncation behavior.
- [x] 1.5 Define the default dynamic avoidance training reward proxy contract: smooth LiDAR clearance, object-free motion/TTC risk, hard-event separation, and numerical-stability guards.

## 2. Implementation

- [x] 2.1 Extend the D.VA observation adapter contract so dynamic avoidance adapters can derive privileged critic observations from `DynamicAvoidanceEnvState` without changing actor observation layout.
- [x] 2.2 Add dynamic avoidance privileged critic observation construction with deterministic scaling, including object-free `clearance_field`, `clearance_delta_field`, `ttc_field`, and `DynamicAvoidanceEnvState` ownership of the previous clearance field.
- [x] 2.3 Fix D.VA rollout sampling so critic targets use pre-reset terminal critic observations where required and never bootstrap from auto-reset observations.
- [x] 2.4 Update `_get_reward_jit` to keep clearance as the current geometry safety term and replace object-level `dobs_risk` with object-free `motion_risk` / `ttc_risk` based on the same clearance motion fields used by the privileged critic schema.
- [x] 2.5 Keep hard termination predicates separate from differentiable reward proxies; wrap any fixed event penalty indicator with stop-gradient.
- [x] 2.6 Add reward proxy numerical-stability tests covering finite reward values and finite gradients in near-collision, safe-clearance, and near-zero relative-velocity scenes.
- [x] 2.7 Add or update dynamic avoidance D.VA training scripts so they run in the `flightning` conda environment and explicitly declare actor/critic optimizer policy.
- [x] 2.8 Add focused automated validation that checks finite dynamic avoidance D.VA actor and critic metrics.
- [x] 2.9 Add focused validation for early termination and time-limit truncation bootstrap behavior.
- [x] 2.10 Remove dynamic avoidance full-training acceptance dependency on notebook execution.

## 3. Verification

- [x] 3.1 Run the dynamic avoidance D.VA validation commands in headless mode.
- [x] 3.2 Run focused D.VA tests covering adapter compatibility and done/truncated bootstrap semantics.
- [x] 3.3 Run focused dynamic avoidance reward proxy tests.
- [x] 3.4 Run `openspec validate replace-avoidance-training-validation-with-dva --strict`.

## 4. Follow-up Notes

True D.VA actor policy convergence, obstacle-aware maneuvering, and goal-reaching success are tracked in `docs/issues/005-dva-policy-convergence-gap.md`.
