## 1. Migration Contract

- [ ] 1.1 Identify the current dynamic avoidance training validation path that depends on legacy notebook execution.
- [ ] 1.2 Define the D.VA-based headless validation command set for dynamic avoidance training.

## 2. Implementation

- [ ] 2.1 Add or update dynamic avoidance D.VA training scripts so they run in the `flightning` conda environment.
- [ ] 2.2 Add focused automated validation that checks finite dynamic avoidance D.VA training metrics.
- [ ] 2.3 Remove dynamic avoidance full-training acceptance dependency on notebook execution.

## 3. Verification

- [ ] 3.1 Run the dynamic avoidance D.VA validation commands in headless mode.
- [ ] 3.2 Run `openspec validate replace-avoidance-training-validation-with-dva --strict`.
