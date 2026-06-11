## ADDED Requirements

### Requirement: D.VA Headless Script Examples
D.VA SHALL include script-first examples that run in a headless server environment and do not require notebook execution for acceptance.

#### Scenario: state-only 脚本可运行
- **WHEN** the state-only D.VA example script is executed in the project test environment
- **THEN** it SHALL run a tiny rollout and report finite D.VA metrics without requiring GUI, notebook kernel state, or Rerun visualization.

#### Scenario: feature/vision 脚本可运行
- **WHEN** the feature or vision-style D.VA example script is executed in the project test environment
- **THEN** it SHALL exercise a non-default observation adapter and report finite actor and critic metrics.

### Requirement: Notebook Optionality
D.VA notebooks MAY exist as explanatory or exploratory artifacts, but they SHALL NOT be the required validation path for this change.

#### Scenario: notebook 不作为硬验收
- **WHEN** D.VA acceptance is evaluated on a remote headless server
- **THEN** script and automated test execution SHALL be sufficient to validate the change.

### Requirement: Dynamic Avoidance Validation Deferred
This change SHALL NOT migrate dynamic avoidance full training acceptance to D.VA.

#### Scenario: 动态避障完整训练验收需要 D.VA
- **WHEN** dynamic avoidance training acceptance needs to move away from the BPTT rerun notebook
- **THEN** that migration SHALL be captured by a separate OpenSpec change, such as `replace-avoidance-training-validation-with-dva`.

#### Scenario: 本 change 只提供通用算法能力
- **WHEN** `add-dva-algorithm` is implemented
- **THEN** it SHALL provide generic D.VA algorithm, adapter, tests, and headless examples that the later dynamic avoidance validation change can reuse.
