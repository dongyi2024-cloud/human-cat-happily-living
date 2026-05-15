## Why

The current cat agent in `simulation_v9.py` is driven mainly by energy and random goal selection, so different cats produce weakly differentiated trajectories and heatmaps. The new `2.0cat_pro/PRD.md` requires the simulation to support comparable cat personalities, objective profiles, dynamic states, and behavior statistics for human-cat co-living layout analysis.

## What Changes

- Add a `cat_profile` input model with objective attributes such as age stage, sex, neuter status, mobility, sensory levels, body condition, and disease history.
- Add five personality dimensions: neuroticism, extraversion, dominance, impulsiveness, and agreeableness.
- Replace the cat agent's fixed energy-based goal list with weighted behavior selection.
- Add dynamic cat state fields for hunger, stress, boredom, security, social need, energy, and satisfaction.
- Modulate movement speed, run probability, behavior switching, and spatial preference using personality plus objective profile modifiers.
- Extend cat behavior labels while preserving compatibility with downstream tick-level trajectory analysis.
- Record cat zone stay ticks, behavior counts, behavior durations, profile summary, and final state summary.
- Output `cat_behavior_summary.json`, `cat_profile_used.json`, and an expanded `tick_records.csv`.
- Support deterministic comparison runs through an optional random seed.
- Keep human agent behavior unchanged except for data needed by cat output records.

## Capabilities

### New Capabilities

- `cat-profile-behavior`: Defines configurable cat profile inputs, personality-driven behavior selection, dynamic state updates, movement/run modulation, and preset comparison profiles.
- `cat-simulation-outputs`: Defines expanded tick records and JSON outputs for profile, behavior, zone stay, and state summaries.

### Modified Capabilities

- None.

## Impact

- Affected code: `simulation_v9.py` primarily, with possible compatibility updates to `trajectory_analyzer.py` if behavior labels or tick CSV columns change.
- Affected outputs: `simulation_result.png`, `tick_records.csv`, `cat_behavior_summary.json`, and `cat_profile_used.json`.
- Affected workflows: single-cat single-floor-plan simulation, local comparison experiments across multiple profiles, downstream trajectory and metric analysis.
- Dependencies: no new third-party dependency is expected.
