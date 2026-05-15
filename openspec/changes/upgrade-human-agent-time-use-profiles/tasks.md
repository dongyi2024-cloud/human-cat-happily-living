## 1. Source Data and Mapping Setup

- [x] 1.1 Add local `data/` CSV fixtures for the initial supported profiles with raw fields needed by the builder.
- [x] 1.2 Add `config/human_profile_mapping.json` for `remote_worker`, `service_worker`, and `default_china`.
- [x] 1.3 Add a versioned `ACTIVITY_TO_ZONE_MAP` definition and mark it as model assumption metadata.
- [x] 1.4 Add validation helpers for required source files, required raw fields, profile status, and paper-readiness flags.

## 2. TimeUseParameterBuilder

- [x] 2.1 Create `time_use_parameter_builder.py` with `TimeUseParameterBuilder`.
- [x] 2.2 Implement source table loading from local CSV files.
- [x] 2.3 Implement raw activity budget extraction for US ATUS-backed profiles.
- [x] 2.4 Implement raw activity budget extraction for China 2018 time-use baseline.
- [x] 2.5 Implement hour-to-minute and minute-to-tick conversion with `tick_minutes`.
- [x] 2.6 Implement activity budget normalization and record scale factor plus `normalization_applied`.
- [x] 2.7 Implement activity-to-zone budget mapping from the versioned mapping assumption.
- [x] 2.8 Implement activity schedule generation with model-defined order metadata.
- [x] 2.9 Implement source metadata generation with datasets, tables, files, raw values, formulas, and assumptions.

## 3. HumanAgent Schedule Model

- [x] 3.1 Update `HumanAgent.__init__` to accept and store a generated `human_profile`.
- [x] 3.2 Replace fixed phase-based activity selection with activity schedule progression.
- [x] 3.3 Add current activity, current zone, human state, and remaining activity duration fields.
- [x] 3.4 Implement activity-to-zone target selection for indoor activities.
- [x] 3.5 Implement logical `outside` state with no indoor movement or heatmap increment.
- [x] 3.6 Add Chinese behavior labels for sleep, home_work, outside, household, care, leisure, education, move, and wander.
- [x] 3.7 Track human zone stay ticks, activity ticks, behavior counts, outside ticks, and final state.

## 4. Simulation Integration and Outputs

- [x] 4.1 Update `Simulation.__init__` to accept `human_profile_id`, `country`, `day_type`, `tick_minutes`, `data_dir`, and mapping path without breaking default construction.
- [x] 4.2 Build the generated human profile at simulation boundary and pass it to `HumanAgent`.
- [x] 4.3 Expand tick records with `human_zone`, `human_activity`, `human_state`, and `human_profile_id`.
- [x] 4.4 Add export support for `human_behavior_summary.json`.
- [x] 4.5 Add export support for `human_profile_used.json`.
- [x] 4.6 Add export support for `source_metadata.json`.
- [x] 4.7 Add export support for `activity_to_zone_mapping_used.json`.
- [x] 4.8 Update visualization report text to show human profile, source dataset, source tables, tick scale, activity budget, and mapping assumption note.
- [x] 4.9 Ensure cat profile behavior and cat outputs remain compatible.

## 5. Downstream Analysis Compatibility

- [x] 5.1 Update `TrajectoryAnalyzer` to skip outside human ticks for indoor grids and co-occurrence calculations.
- [x] 5.2 Add behavior weights for new Chinese human behavior labels.
- [x] 5.3 Verify existing cat and human analyzer-required fields remain present.
- [x] 5.4 Ensure dashboard generation works with expanded human records and outside ticks.

## 6. Verification

- [x] 6.1 Run default simulation and verify `simulation_result.png` plus all cat and human JSON/CSV outputs are generated.
- [x] 6.2 Run `remote_worker` and verify home work time and work-zone heatmap presence are data-derived.
- [x] 6.3 Run `service_worker` and verify outside time is higher than `remote_worker`.
- [x] 6.4 Run `default_china` and verify budget fields are derived from China 2018 source rows.
- [x] 6.5 Verify `human_activity_ticks` sums to total ticks for each tested profile.
- [x] 6.6 Verify `source_metadata.json` contains source files, tables, raw values, formulas, normalization status, and model assumptions.
- [x] 6.7 Verify `activity_to_zone_mapping_used.json` records the model-defined mapping separately from source data.
- [x] 6.8 Run trajectory/metrics/node/dashboard analysis using the generated `tick_records.csv`.
