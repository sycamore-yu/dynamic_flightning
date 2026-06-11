# Change: Replace Avoidance Training Validation with D.VA

## Why
Dynamic avoidance training acceptance currently depends on legacy notebook-style validation. The project needs a script-first, headless validation path that can run in the `flightning` conda environment and reuse the D.VA algorithm added by `add-dva-algorithm`.

## What Changes
- Define a follow-up migration from legacy dynamic avoidance training validation to D.VA-based headless validation.
- Require script-first acceptance for dynamic avoidance training without relying on notebook execution.
- Keep this change scoped to the migration plan and acceptance contract; implementation happens in a later task pass.

## Impact
- Affected specs: `dynamic-avoidance-training-validation`
- Affected code in future implementation: dynamic avoidance training examples/tests and any legacy notebook acceptance references.
