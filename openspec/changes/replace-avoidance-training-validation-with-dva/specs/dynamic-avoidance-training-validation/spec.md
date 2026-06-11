## ADDED Requirements

### Requirement: Dynamic Avoidance Training Uses D.VA Validation
Dynamic avoidance full training acceptance SHALL use a D.VA-based headless validation path instead of relying on legacy notebook execution.

#### Scenario: Headless D.VA validation command succeeds
- **WHEN** dynamic avoidance training acceptance is evaluated on a remote headless server
- **THEN** the validation SHALL run from script or test commands in the `flightning` conda environment.
- **AND** it SHALL report finite D.VA actor and critic metrics.

#### Scenario: Notebook execution is not required
- **WHEN** dynamic avoidance training acceptance is evaluated
- **THEN** notebook execution SHALL NOT be required as the primary acceptance path.

#### Scenario: Migration reuses generic D.VA capability
- **WHEN** the migration is implemented
- **THEN** it SHALL reuse the generic D.VA algorithm and observation adapter contract from `add-dva-algorithm` rather than creating a dynamic-avoidance-only algorithm fork.
