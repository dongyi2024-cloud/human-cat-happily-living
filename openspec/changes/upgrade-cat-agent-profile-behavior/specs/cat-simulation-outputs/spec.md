## ADDED Requirements

### Requirement: Expanded tick records
The simulation SHALL record tick-level cat fields for zone, behavior, energy, stress, hunger, and boredom in addition to existing cat and human positions and human behavior.

#### Scenario: Tick records include cat state
- **WHEN** a simulation run completes
- **THEN** every tick record contains `cat_zone`, `cat_behavior`, `cat_energy`, `cat_stress`, `cat_hunger`, and `cat_boredom`

#### Scenario: Tick records remain analyzer-compatible
- **WHEN** downstream trajectory analysis reads the tick records
- **THEN** existing fields `tick`, `cat_x`, `cat_y`, `cat_behavior`, `human_x`, `human_y`, and `human_behavior` remain present

### Requirement: Cat behavior summary output
The simulation SHALL output a `cat_behavior_summary.json` file containing zone stay ticks, behavior counts, behavior durations, final dynamic state, and summary totals for the completed run.

#### Scenario: Summary JSON is written
- **WHEN** the simulation exports outputs after a run
- **THEN** `cat_behavior_summary.json` exists and contains `zone_stay_ticks`, `behavior_counts`, `behavior_durations`, and `final_state`

#### Scenario: Summary counts match run length
- **WHEN** a simulation runs for `N` ticks
- **THEN** the sum of `zone_stay_ticks` values equals `N`

### Requirement: Cat profile output
The simulation SHALL output a `cat_profile_used.json` file containing the normalized profile used for the run.

#### Scenario: Profile JSON is written
- **WHEN** the simulation exports outputs after a run
- **THEN** `cat_profile_used.json` exists and includes objective fields, personality fields, and the profile name

### Requirement: Tick CSV output
The simulation SHALL support exporting `tick_records.csv` using the expanded tick record fields.

#### Scenario: Tick CSV is written
- **WHEN** tick records are exported after a run
- **THEN** `tick_records.csv` contains one row per tick and includes the expanded cat state fields

### Requirement: Visualization report includes cat profile context
The simulation report panel SHALL include key cat profile and final state values without removing the existing trajectory, cat heatmap, and human heatmap panels.

#### Scenario: Report includes profile summary
- **WHEN** the simulation visualization is generated
- **THEN** the report includes the cat profile name or age stage and key final state values

### Requirement: Reproducible comparison runs
The simulation SHALL accept an optional random seed and use it to make profile comparison runs reproducible.

#### Scenario: Same seed reproduces records
- **WHEN** two simulations use the same floor plan, same cat profile, same total ticks, and same random seed
- **THEN** they produce the same tick-level cat behavior sequence
