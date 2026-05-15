## Why

`HumanAgent` currently uses fixed morning/daytime/evening zone preferences, so all human trajectories are essentially heuristic and weakly differentiated. The new `2.0human_pro/PRD.md` requires human activity budgets to be driven by authoritative time-use data so simulation parameters can be traced, reproduced, and defended in academic writing.

## What Changes

- Add data-driven human profile support for residential-friendly profile IDs such as `remote_worker`, `service_worker`, and `default_china`.
- Add source-backed profile mapping and time-use CSV inputs so raw activity budgets are read from versioned data files instead of hardcoded final parameters.
- Add a `TimeUseParameterBuilder` that converts source activity hours into activity ticks, zone budgets, activity schedules, source metadata, and explicit model assumptions.
- Update `HumanAgent` to consume a generated `human_profile`, follow an activity schedule, emit expanded Chinese behavior labels, and support a logical `outside` state.
- Expand tick records with human zone, activity, state, and profile fields while preserving existing analyzer-required human position and behavior fields.
- Add outputs for `human_behavior_summary.json`, `human_profile_used.json`, `source_metadata.json`, and `activity_to_zone_mapping_used.json`.
- Update the simulation report to show human profile, data source, tick scale, activity budget, and the distinction between data-driven activity durations and model-defined zone mapping.
- Keep the cat agent behavior model unchanged except for compatibility with expanded tick records and outputs.

## Capabilities

### New Capabilities

- `human-time-use-profiles`: Defines authoritative-data-backed human profile inputs, profile mapping, source metadata, activity budget derivation, tick scaling, and reproducibility rules.
- `human-activity-simulation`: Defines activity-schedule-driven `HumanAgent` behavior, outside state semantics, human behavior labels, zone mapping assumptions, and summary outputs.

### Modified Capabilities

- None.

## Impact

- Affected code: `simulation_v9.py`, plus new modules and config/data files for time-use parameter generation.
- Affected outputs: `tick_records.csv`, `simulation_result.png`, `human_behavior_summary.json`, `human_profile_used.json`, `source_metadata.json`, and `activity_to_zone_mapping_used.json`.
- Affected workflows: default simulation run, batch profile comparison, downstream trajectory analysis, dashboard generation, and paper-method reproducibility checks.
- Dependencies: no new third-party dependency is expected; implementation should use Python standard library CSV/JSON parsing plus existing NumPy/Pandas stack where appropriate.
