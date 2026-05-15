## ADDED Requirements

### Requirement: Cat profile input
The simulation SHALL allow a cat to be created from a `cat_profile` containing objective attributes and five personality dimensions, and SHALL use default values when the caller does not provide a profile.

#### Scenario: Default profile is used
- **WHEN** a simulation is created without a `cat_profile`
- **THEN** the cat agent uses a valid adult default profile with all required objective and personality fields

#### Scenario: Custom profile is used
- **WHEN** a simulation is created with a custom `cat_profile`
- **THEN** the cat agent uses the provided objective and personality values for behavior decisions

### Requirement: Personality-driven behavior weights
The cat agent SHALL choose its next behavior using weighted random selection derived from dynamic state, zone context, and personality dimensions: neuroticism, extraversion, dominance, impulsiveness, and agreeableness.

#### Scenario: Sensitive profile increases hiding
- **WHEN** the cat profile has high neuroticism and low agreeableness
- **THEN** `hide` and `rest` receive higher selection weights than they would for the default profile under equivalent state

#### Scenario: Curious active profile increases exploration
- **WHEN** the cat profile has high extraversion and high impulsiveness
- **THEN** `explore`, `run`, and `watch_window` receive higher selection weights than they would for the default profile under equivalent state

#### Scenario: Friendly profile increases human-zone affinity
- **WHEN** the cat profile has high agreeableness
- **THEN** `approach_human` receives a higher selection weight and maps to shared or human-used zones

### Requirement: Objective profile modifiers
The cat agent SHALL use age stage, mobility level, body condition, neuter status, and disease history as bounded modifiers on speed, run probability, rest weight, exploration weight, stress, and satisfaction where applicable.

#### Scenario: Senior arthritis profile reduces activity
- **WHEN** the cat profile has `age_stage` of `senior`, `mobility_level` below `1.0`, and disease history containing `arthritis`
- **THEN** movement speed and run probability are lower than the default adult profile, while rest weight is higher

#### Scenario: Unneutered territorial profile increases claiming
- **WHEN** the cat profile is not neutered and has high dominance
- **THEN** `claim_spot` receives a higher selection weight than it would for a neutered default-dominance cat

### Requirement: Dynamic cat state
The cat agent SHALL maintain dynamic state fields for energy, satisfaction, hunger, stress, boredom, security, and social need, and SHALL update them during each simulation tick.

#### Scenario: Hunger increases over time
- **WHEN** the cat advances ticks without feeding
- **THEN** hunger increases within the configured bounds

#### Scenario: Feeding reduces hunger
- **WHEN** the cat reaches a feeding behavior goal
- **THEN** hunger decreases and energy increases within the configured bounds

#### Scenario: Resting improves recovery
- **WHEN** the cat reaches a rest or hide-compatible safe zone
- **THEN** energy and security increase and stress decreases within the configured bounds

### Requirement: Movement and behavior switching modulation
The cat agent SHALL calculate step length, run probability, and goal change limit from base config values plus personality and objective modifiers, with lower and upper bounds.

#### Scenario: Impulsive cat switches more frequently
- **WHEN** two cats have equivalent state but one has higher impulsiveness
- **THEN** the higher-impulsiveness cat has a lower goal change limit

#### Scenario: Mobility impairment limits movement
- **WHEN** a cat profile has reduced mobility or mobility-limiting disease history
- **THEN** calculated step length and run probability are reduced but remain above minimum bounds

### Requirement: Preset comparison profiles
The simulation SHALL provide at least four preset cat profiles for comparison: sensitive hiding, curious active, friendly companion, and senior arthritis.

#### Scenario: Presets are available
- **WHEN** code requests the preset cat profiles
- **THEN** it receives four named profiles with objective and personality values matching their intended behavior archetypes
