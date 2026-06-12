# Change: Replace Avoidance Training Validation with D.VA

## Why
Dynamic avoidance training acceptance currently depends on legacy notebook-style validation. The project needs a script-first, headless validation path that can run in the `flightning` conda environment and reuse the D.VA algorithm added by `add-dva-algorithm`.

## What Changes
- Define a follow-up migration from legacy dynamic avoidance training validation to D.VA-based headless validation.
- Require script-first acceptance for dynamic avoidance training without relying on notebook execution.
- Require dynamic avoidance D.VA validation to use actor-visible LiDAR observations with a separate privileged critic observation derived from `DynamicAvoidanceEnvState`.
- Define the privileged critic observation as an ego-centric object-free clearance-motion schema rather than a simulator object-list schema.
- Replace object-level `dobs_risk` in the default validation reward with object-free motion/TTC risk while preserving LiDAR clearance as the current geometry safety term.
- Fix D.VA rollout bootstrap semantics so done/truncated handling uses pre-reset terminal critic observations rather than auto-reset observations.
- Keep LiDAR scan mode productization, including `mid360_livox`, out of this change; that belongs in a separate sensor-mode change.

## Impact
- Affected specs: `dynamic-avoidance-training-validation`, `dynamic-avoidance-privileged-critic-schema`
- Affected code in future implementation: D.VA adapter contract, dynamic avoidance privileged critic observation construction, clearance-motion field state ownership, D.VA rollout bootstrap handling, dynamic avoidance reward proxy, dynamic avoidance D.VA examples/tests, and any legacy notebook acceptance references.
