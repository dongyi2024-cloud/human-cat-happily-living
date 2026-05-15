## 1. Profile Model Setup

- [x] 1.1 Add default cat profile data with objective fields and five personality dimensions.
- [x] 1.2 Add four named preset profiles: sensitive hiding, curious active, friendly companion, and senior arthritis.
- [x] 1.3 Add profile normalization helpers so missing custom profile fields fall back to defaults.
- [x] 1.4 Add clamp and weighted random choice helpers.

## 2. CatAgent Behavior Model

- [x] 2.1 Update `CatAgent.__init__` to accept and store a normalized `cat_profile`.
- [x] 2.2 Replace standalone cat state fields with a dynamic `self.state` dictionary while preserving existing energy and satisfaction access for reports.
- [x] 2.3 Add age and disease modifier calculation helpers.
- [x] 2.4 Implement `calculate_behavior_weights()` using personality, objective modifiers, and dynamic state.
- [x] 2.5 Implement behavior-to-zone mapping for rest, feed, explore, watch_window, hide, run, wander, claim_spot, and approach_human.
- [x] 2.6 Replace fixed goal selection with behavior-weighted goal selection.
- [x] 2.7 Implement dynamic state updates for hunger, stress, boredom, security, social_need, energy, and satisfaction.
- [x] 2.8 Implement speed, run probability, and goal change limit modulation with bounds.
- [x] 2.9 Keep tick-level cat behavior output analyzer-compatible through Chinese behavior labels.

## 3. Statistics and Outputs

- [x] 3.1 Track `zone_stay_ticks`, `behavior_counts`, and `behavior_durations` during each cat tick.
- [x] 3.2 Expand simulation tick records with `cat_zone`, `cat_energy`, `cat_stress`, `cat_hunger`, and `cat_boredom`.
- [x] 3.3 Add export support for `tick_records.csv`.
- [x] 3.4 Add export support for `cat_behavior_summary.json`.
- [x] 3.5 Add export support for `cat_profile_used.json`.
- [x] 3.6 Update visualization report text to include cat profile context and final dynamic state.

## 4. Simulation Integration

- [x] 4.1 Update `Simulation.__init__` to accept `cat_profile`, `random_seed`, and output configuration without breaking default construction.
- [x] 4.2 Seed both `random` and `numpy.random` when `random_seed` is provided.
- [x] 4.3 Ensure the human agent behavior remains unchanged.
- [x] 4.4 Ensure generated outputs default to a predictable location and do not break existing `simulation_result.png` behavior.

## 5. Verification

- [x] 5.1 Run the default simulation and verify trajectory, cat heatmap, human heatmap, and report are still generated.
- [x] 5.2 Run the four preset cat profiles with the same seed and verify behavior summaries differ in expected directions.
- [x] 5.3 Verify `cat_behavior_summary.json` zone stay totals equal total ticks.
- [x] 5.4 Verify `cat_profile_used.json` contains the normalized profile used for the run.
- [x] 5.5 Verify `tick_records.csv` contains existing analyzer-required fields plus expanded cat state fields.
- [x] 5.6 Run downstream trajectory/metrics analysis or update behavior weights if new output labels require it.
