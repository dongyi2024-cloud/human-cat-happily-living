## ADDED Requirements

### Requirement: Source-backed human profile mapping
The system SHALL define human profile IDs through a mapping file that records display name, source dataset, source tables, raw source files, statistical population or occupation categories, profile status, and paper-readiness status.

#### Scenario: Data-supported profile has source mapping
- **WHEN** code requests a data-supported profile such as `remote_worker`, `service_worker`, or `default_china`
- **THEN** the profile mapping contains dataset, tables, raw files, category fields, and `profile_status` of `data_supported`

#### Scenario: Model-assumption profile is marked
- **WHEN** a profile lacks sufficient authoritative source data
- **THEN** the profile mapping marks it as `model_assumption` and excludes it from paper-ready default comparisons

### Requirement: Local authoritative source data inputs
The system SHALL read activity time budgets from local versioned CSV files rather than hardcoding final simulation parameters.

#### Scenario: Source data is loaded
- **WHEN** `TimeUseParameterBuilder` builds a data-supported profile
- **THEN** it reads the configured local raw files and extracts the raw fields needed for activity budget generation

#### Scenario: Required source file is missing
- **WHEN** a configured raw source file cannot be found
- **THEN** profile generation fails with an error naming the missing file and profile ID

### Requirement: Activity budget derivation
The system SHALL convert source activity hours into activity minutes and ticks using explicit formulas recorded in metadata.

#### Scenario: Work location split is derived
- **WHEN** raw data includes `work_hours` and `home_work_share`
- **THEN** `home_work_ticks` is derived from `work_hours * home_work_share` and `outside_ticks` includes `work_hours * (1 - home_work_share)` plus travel time when available

#### Scenario: Tick scale is applied
- **WHEN** profile generation uses `tick_minutes`
- **THEN** each activity tick count equals its activity minutes divided by `tick_minutes`, after any required normalization

### Requirement: Budget normalization
The system SHALL ensure generated activity budgets sum to the simulation day length and SHALL record whether normalization was applied.

#### Scenario: Source fields do not sum to one day
- **WHEN** raw activity minutes do not sum to `total_ticks * tick_minutes`
- **THEN** the builder scales activity minutes to the requested day length and records `normalization_applied = true` plus the scale factor

#### Scenario: Source fields already represent a full day
- **WHEN** raw activity minutes already sum to the requested day length within tolerance
- **THEN** the builder records `normalization_applied = false`

### Requirement: Activity-to-zone assumptions
The system SHALL map activities to residential zones through a versioned model-assumption mapping separate from source-derived activity durations.

#### Scenario: Zone budget is generated
- **WHEN** a human profile is built
- **THEN** the builder creates `zone_budget` from `derived_activity_budget` and the configured activity-to-zone mapping

#### Scenario: Mapping is exported separately
- **WHEN** simulation outputs are exported
- **THEN** `activity_to_zone_mapping_used.json` contains the mapping used and identifies it as a model assumption

### Requirement: Human profile output
The system SHALL output the normalized generated human profile used in a run.

#### Scenario: Human profile JSON is written
- **WHEN** simulation outputs are exported after a run
- **THEN** `human_profile_used.json` contains profile ID, display name, source, derived activity budget, zone budget, activity schedule, and assumptions

### Requirement: Source metadata output
The system SHALL output source metadata sufficient to trace the generated profile back to source files, source tables, raw fields, formulas, and model assumptions.

#### Scenario: Source metadata JSON is written
- **WHEN** simulation outputs are exported after a run
- **THEN** `source_metadata.json` contains dataset, source tables, raw files, category fields, raw values, formulas, normalization status, and model assumptions

### Requirement: Supported initial profiles
The system SHALL provide at least three data-supported profiles for initial comparison: `remote_worker`, `service_worker`, and `default_china`.

#### Scenario: Initial supported profiles are available
- **WHEN** code lists data-supported human profiles
- **THEN** it includes `remote_worker`, `service_worker`, and `default_china`
