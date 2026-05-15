## ADDED Requirements

### Requirement: HumanAgent accepts generated human profile
The simulation SHALL create `HumanAgent` from a generated `human_profile` while preserving default construction compatibility.

#### Scenario: Generated profile is used
- **WHEN** simulation is created with `human_profile_id`, `country`, and `day_type`
- **THEN** `HumanAgent` receives a generated profile containing activity budget, zone budget, activity schedule, source, and assumptions

#### Scenario: Default construction still works
- **WHEN** simulation is created without explicit human profile arguments
- **THEN** a valid default generated human profile is used

### Requirement: Activity schedule drives human behavior
The human agent SHALL select activities from a generated activity schedule rather than from fixed morning/daytime/evening zone lists.

#### Scenario: Activity advances by duration
- **WHEN** the current activity has consumed its configured duration
- **THEN** the human agent advances to the next activity in the schedule

#### Scenario: Zone is selected from current activity
- **WHEN** the current activity is an indoor activity such as `home_work`, `household`, `care`, `leisure`, or `education`
- **THEN** the target residential zone is selected from the activity-to-zone mapping for that activity

### Requirement: Outside state
The human agent SHALL support a logical `outside` state for activity time spent outside the residence.

#### Scenario: Human is outside
- **WHEN** the current activity is `outside`
- **THEN** `human_state` is `outside`, `human_zone` is `outside`, `human_behavior` is `外出`, and indoor human heatmap counts do not increase

#### Scenario: Outside tick coordinates
- **WHEN** a tick record is written while the human is outside
- **THEN** `human_x` and `human_y` are empty/null-compatible and no synthetic indoor coordinate is invented

### Requirement: Human behavior labels
The human agent SHALL output analyzer-compatible Chinese behavior labels for generated human activities.

#### Scenario: Activity labels are emitted
- **WHEN** tick records are written
- **THEN** human activities map to Chinese labels including `睡眠`, `居家工作`, `外出`, `家务`, `照护`, `休闲`, `学习`, `移动`, and `闲逛`

#### Scenario: Behavior weights support new labels
- **WHEN** downstream intensity metrics process human behavior labels
- **THEN** new human labels have explicit behavior weights or documented default handling

### Requirement: Expanded tick records
The simulation SHALL record human zone, activity, state, and profile ID in addition to existing tick-level cat and human fields.

#### Scenario: Expanded human fields are present
- **WHEN** a simulation run completes
- **THEN** every tick record contains `human_zone`, `human_activity`, `human_behavior`, `human_state`, and `human_profile_id`

#### Scenario: Existing analyzer fields remain present
- **WHEN** downstream trajectory analysis reads tick records
- **THEN** existing fields `tick`, `cat_x`, `cat_y`, `cat_behavior`, `human_x`, `human_y`, and `human_behavior` remain present

### Requirement: Analyzer handles outside ticks
The trajectory analyzer SHALL handle outside human ticks without treating outside time as an indoor coordinate.

#### Scenario: Outside ticks are skipped for indoor grids
- **WHEN** a tick has `human_state = outside` or empty `human_x` and `human_y`
- **THEN** the analyzer excludes that human tick from indoor human behavior grids and indoor co-occurrence calculations

### Requirement: Human behavior summary output
The simulation SHALL output a human behavior summary containing zone stay ticks, activity ticks, behavior counts, outside ticks, final state, and profile summary.

#### Scenario: Human summary JSON is written
- **WHEN** simulation outputs are exported after a run
- **THEN** `human_behavior_summary.json` contains `human_zone_stay_ticks`, `human_activity_ticks`, `behavior_counts`, `outside_ticks`, `final_state`, and `human_profile_summary`

#### Scenario: Summary totals match run length
- **WHEN** a simulation runs for `N` ticks
- **THEN** the sum of `human_activity_ticks` equals `N`

### Requirement: Visualization report includes human profile context
The simulation report panel SHALL include human profile, source dataset, source tables, tick scale, activity budget, and a note that residential zone mapping is model-defined.

#### Scenario: Report includes human source context
- **WHEN** simulation visualization is generated
- **THEN** the report includes the generated human profile display name or ID and the source dataset used

### Requirement: Cat behavior remains unchanged
The human profile upgrade SHALL NOT change cat profile behavior selection, cat dynamic state updates, or cat output semantics except where tick record compatibility requires shared export changes.

#### Scenario: Cat profile path still works
- **WHEN** simulation is run with an existing cat profile and default human profile
- **THEN** cat profile fields, cat behavior summary, and cat tick labels are still generated
