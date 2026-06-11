## 1. Specs and Scope

- [x] 1.1 Validate `add-dva-algorithm` with strict OpenSpec checks after delta specs are present.
- [x] 1.2 Keep D.VA scoped as a generic algorithm capability; do not migrate dynamic avoidance full training acceptance in this change.
- [x] 1.3 Create a follow-up OpenSpec change for dynamic avoidance training acceptance migration, recommended name: `replace-avoidance-training-validation-with-dva`.

## 2. Algorithm Implementation

- [x] 2.1 Add `flightning/algos/dva.py` with `DVAConfig`, `DVAObservation`, a default observation adapter, runner/sample state containers, and `train(...)`.
- [x] 2.2 Match existing Flightning algorithm conventions: caller-owned env/networks/optimizers, internal `LogWrapper` and `VecEnv`, JAX-compatible rollout, and `{"runner_state": ..., "metrics": ...}` output.
- [x] 2.3 Implement actor update semantics with `jax.lax.stop_gradient(actor_obs)` while preserving gradients through `actor -> action -> env.step -> reward`.
- [x] 2.4 Implement critic update using adapter-provided `critic_obs`, target critic EMA, finite one-step or TD-lambda targets, configurable critic iterations, batching, and gradient clipping.
- [x] 2.5 Export D.VA from `flightning/algos/__init__.py` without changing BPTT, SHAC, or PPO public APIs.

## 3. Tests

- [x] 3.1 Add a state-only D.VA smoke test that runs a tiny rollout and asserts finite metrics.
- [x] 3.2 Add an adapter test that verifies custom actor/critic observation shapes and JAX transformation compatibility.
- [x] 3.3 Add a gradient semantics test showing actor observation gradients are stopped while actor-parameter gradients through action-dependent reward remain finite and nonzero.
- [x] 3.4 Add a critic update smoke test that verifies actor loss, critic loss, target values, and gradient norms are finite.

## 4. Examples

- [x] 4.1 Add a headless state-only D.VA training script that can run in the `flightning` conda environment.
- [x] 4.2 Add a headless feature/vision-style D.VA script that exercises a non-default observation adapter.
- [x] 4.3 Treat notebooks as optional documentation or exploration artifacts, not as required validation for this change.

## 5. Verification

- [x] 5.1 Run `npx gitnexus analyze` before implementation work that edits code symbols.
- [x] 5.2 Run the focused D.VA tests in the `flightning` conda environment.
- [x] 5.3 Run the headless D.VA scripts once before presenting them as usable examples.
- [x] 5.4 Run `openspec validate add-dva-algorithm --strict`.
