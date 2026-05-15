## Context

`simulation_v9.py` currently keeps the cat model inside a single `CatAgent` class. The agent stores `energy` and `satisfaction`, picks a goal from fixed zone lists, moves toward that goal, sometimes runs with a fixed probability, and emits Chinese behavior labels consumed by tick-level trajectory analysis.

The PRD in `2.0cat_pro/PRD.md` changes the model from a generic cat into a configurable research subject. The implementation must remain easy to run as a standalone script, avoid new dependencies, preserve tick-level data as the source of truth, and avoid breaking the existing human agent and downstream analysis modules.

## Goals / Non-Goals

**Goals:**

- Add a default cat profile plus named preset profiles for comparison experiments.
- Let objective profile and five personality dimensions affect behavior weights, movement speed, run probability, target switching, and zone preference.
- Track dynamic state fields in one `self.state` dictionary while keeping existing report fields compatible.
- Preserve Chinese behavior labels in tick records unless downstream analyzers are updated in the same change.
- Add summary and profile JSON outputs for repeatable comparison.
- Keep the first implementation inside `simulation_v9.py` unless extraction becomes necessary during implementation.

**Non-Goals:**

- No pet-owner questionnaire mapping.
- No multi-cat simulation.
- No complex real-time human-cat interaction.
- No medical diagnosis or health advice.
- No new CAD/DXF, sensor, event, litter-box, high-place, or object-level spatial systems.
- No new third-party dependencies.

## Decisions

### Keep internal behavior IDs separate from output labels

Use English behavior IDs internally, such as `rest`, `feed`, `explore`, `watch_window`, `hide`, `run`, `wander`, `claim_spot`, and `approach_human`. Convert them to Chinese labels when writing tick records and reports.

Alternative considered: emit English labels everywhere. This would require broader changes to `trajectory_analyzer.py` behavior weights and existing reports. Keeping Chinese output minimizes downstream breakage.

### Store all cat state in `self.state`

`CatAgent` will keep `energy`, `satisfaction`, `hunger`, `stress`, `boredom`, `security`, and `social_need` in `self.state`. Compatibility properties or synchronized assignments can preserve existing `self.energy` and `self.satisfaction` references until visualization/report code is updated.

Alternative considered: add each state as a standalone attribute. A dictionary better matches the PRD and makes JSON summary output straightforward.

### Use weighted behavior choice as the decision center

`choose_new_goal()` will become behavior-driven: calculate weights from state, personality, age modifiers, disease modifiers, and current zone; pick a behavior; then map that behavior to candidate zones.

Alternative considered: retain the current fixed target lists and only tweak probabilities. That would not satisfy the core requirement that personality and objective conditions drive visible behavioral differences.

### Treat `approach_human` as a zone-level behavior in this phase

The first implementation maps `approach_human` to shared and human-used zones (`shared`, `human_work`, `human_sleep`) instead of chasing the human agent's exact current position.

Alternative considered: pass human position into every cat step. That is useful later but introduces human-cat interaction complexity that the PRD explicitly excludes from this stage.

### Use objective modifiers as bounded multipliers

Age and disease history will produce small, explainable multipliers for speed, run, rest, explore, hide, and satisfaction where applicable. Values are clamped to prevent immobile or hyperactive edge cases.

Alternative considered: hard rules such as "senior cats never run." That would be less realistic and more brittle.

### Add deterministic seeds at simulation boundary

`Simulation(..., random_seed=None)` will seed both `random` and `numpy.random` when provided. This supports comparison experiments without changing existing default behavior.

Alternative considered: global fixed seed. That would make all default runs identical and reduce exploratory use.

## Risks / Trade-offs

- Behavior labels drift from analyzer weights → keep Chinese tick labels and update `BEHAVIOR_WEIGHTS` only if new labels enter downstream analysis.
- Too many parameters make behavior hard to explain → keep formulas simple, bounded, and localized in named helper methods.
- Profile effects may be too subtle or too strong → add four preset profiles and compare behavior counts/heatmaps as implementation checks.
- JSON output paths could clutter the repo root → default to an `outputs/` directory while preserving `simulation_result.png` compatibility if the current entry point still writes to root.
- `approach_human` may not visibly follow the human → document it as zone-level affinity in this stage and defer exact proximity behavior.

## Migration Plan

1. Add profile defaults and helper functions without changing the command-line entry point.
2. Update `CatAgent` to accept `cat_profile` and produce the same basic trajectory/heatmap outputs.
3. Add expanded tick fields and JSON exports.
4. Run default simulation to verify backward-compatible execution.
5. Run the four preset profiles with a fixed seed to confirm distinct behavior summaries.

Rollback is simple because this is a local script change: revert `simulation_v9.py` and remove generated output files.

## Open Questions

- Should the main script run one default profile or generate all four preset profile comparisons by default?
- Should `tick_records.csv` be written automatically on every `Simulation.run()` or only through an explicit export method?
- Should summary outputs go to `outputs/` only, or should root-level legacy outputs remain the default?
