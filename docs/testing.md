# Testing

The suite has 125 tests in 14 files. Stage 1 had 66 tests in 10 files; the evaluation, pairing and fix work of 2026-10-15 added 31, the `--policies` and `--seed-start` options of 2026-10-16 added 13, and milestone M1 (measurements) added 15 (section 7). It runs in about 145 seconds on the Windows machine in section 6, most of it the prefix-equivalence test (it was 16 to 27 seconds for the 66 tests on the WSL2 development machine), including one scripted table build, one short training run and the episode replays of the pairing tests. Run it from the repository root:

```bash
.venv/bin/python -m pytest              # all 125 tests (Windows: .\.venv\Scripts\python.exe -m pytest)
.venv/bin/python -m pytest -q tests/test_physics.py   # one file
.venv/bin/python -m pytest -k weak      # tests whose names contain "weak"
```

Three session-scoped fixtures in `tests/conftest.py` are shared across the suite:

| Fixture | What it builds | Cost |
|---|---|---|
| `arm` | The arm6 platform, the tiers, the damage model and the scripted table | Built once, about 4 s |
| `fast_settings` | Training settings with 16 candidates, 2 instances, 12 generations, 40 gate runs, 10 table rollouts | Instant |
| `shoulder_store` | Shoulder policies for T1 to T3, trained with `fast_settings` into a temporary folder | About 2 s |

## 1. What each file guards

| File | Tests | What it guards |
|---|---|---|
| `test_tiers.py` | 8 | Pass and fail at and around each threshold, for both directions (min and max). NaN readings always fail. Tier passes only if every check does. Noisy pass probability is 0.5 at the threshold and 0 or 1 without noise. Vector and scalar versions agree. Tier order is T0 to T3. |
| `test_physics.py` | 5 | A batch of one matches a batch of three, element by element. A locked joint in a batch does not change its neighbours. An aborted joint stops updating. A locked joint stalls at full torque in the direction of the demand, and at zero torque with no demand (regression for the sign defect). The controller is called with an error array and a speed array, each the size of the batch, and nothing else. |
| `test_probes.py` | 11 | The scripted reference keeps its known answers: healthy joints reach the top tier; locked fails T0; missing sensor is never moved; the weak shoulder overloads on its hold; the weak motor on the base passes everything; the broken link fails T2 on every joint; healthy hold sag is g/kp; the table is deterministic; moves spend energy; the prior sums to 1. |
| `test_policies.py` | 4 | Zero error and zero speed give zero torque (no hidden gravity term). Gain multipliers scale the nominal gains exactly. A stored policy loads back equal. A missing policy fails with a message naming the training command. |
| `test_training.py` | 7 | The Wilson lower bound matches its formula. Centred ranks are symmetric and order-preserving. Training is deterministic for a seed. Learned gains stay inside their clamps. The stiffness multiplier never exceeds nominal. The gate's trust decision uses the Wilson bound. The real settings load, with three pairwise-distinct seed bases. The training and gate seed ranges (base + slot index) never overlap, and the table seed is in neither. |
| `test_assessors.py` | 6 | Scripted and learned assessors return the same prediction shape. A trained healthy shoulder passes the move with probability above 0.8. A missing sensor is never run by a learned policy, and produces zero energy. A weak shoulder hold is predicted to overload with probability above 0.5. An execution reports the violations it hit. A locked base fails T0 on the scripted reference. |
| `test_belief.py` | 7 | Consistent evidence moves belief towards the hypothesis that predicts it. The probability floor stops a hypothesis being ruled out for good. Expected information is zero when all hypotheses predict the same outcome, and one bit when the outcome reveals the level exactly. Information is bounded by the outcome's entropy. Binary entropy values. Level probability sums correctly. |
| `test_gating.py` | 8 | A failed proximal joint blocks the joints below it at the same tier. The other chain is unaffected. A tier waits for its own lower tier. An undecided proximal joint makes the one below wait. A belief alone does not pass a tier: a probe is needed. The repeat limit gives `unknown`, which spreads down the chain. Whole-robot gating blocks joints in other chains; chain gating does not. An untrusted skill is `unknown` and its dependants inherit that. |
| `test_planner.py` | 7 | No candidates means stop. Passive never probes. The safety veto blocks a probe whose violation probability exceeds the cap, for the planner and for random. The planner still runs a zero-information probe if it is the only one left. It prefers the probe that separates hypotheses. A high risk weight makes it avoid a risky probe. The fixed battery runs in tier order, then joint order. |
| `test_episode.py` | 10 | A healthy arm reaches level 3 on every joint. A locked base blocks every joint above it at T0, and nothing above it is probed. Every method stays within the budget and produces a serialisable report. Passive runs no probes. The same seed and method give identical reports. Pairing: the r-th run of a probe gives the same outcome under every method, whatever the order. The same probes with the same outcomes give the same final belief, including in a different order. Probe streams differ by probe and repeat, and never reuse the model table's numbers. `shared` mode uses one generator in probe order and is deterministic. An unknown stream mode is rejected. |
| `test_evaluate.py` | 16 | Brier and log-loss match hand calculations. Log-loss is finite at probabilities 0 and 1. Level forecasts are cumulative. The MAP set keeps every tied state, and a tie gives half credit. Multi-class Brier matches a hand calculation. Passive scores equal the prior-only forecast. Probing a healthy arm beats the prior on every proper score. A failed locked base is a scored tie, while the old argmax metric is unchanged. The existing score keys keep their meaning. The paired bootstrap is exactly zero for identical methods, exact for a constant shift, pairs by seed, is reproducible, and rejects mismatched input. |
| `test_compare_methods.py` | 21 | The comparison script on 2 seeds: the original columns come first and in order; one trial row per method, budget and seed, with the same damage per seed; paired rows compare every other method with fixed, and each interval contains its mean; passive runs no probes; the ceiling equals fixed at a budget that cannot bind. Seeds: without `--seed-start` they are 0 to n - 1; with it, every method runs on the same shifted seeds, and the seed value (not its position) sets the damage. Invalid `--seeds`, `--seed-start`, `--bootstrap`, and `--policies` with the scripted assessment are rejected with a message. `--policies` is used as given; a missing folder is rejected, not replaced; the default is `results/policies/<platform>`; an incomplete folder stops the run before any episode and names the missing files. `--curve-out`: one row per method, seed and k = 0 .. the largest budget; the summary and trial schemas are unchanged; curve rows at each budget equal the separately run trials. |
| `test_measurements.py` | 12 | Prefix evaluation: for every method and 11 seeds (including a real abort), the replayed state after k probes gives the same report and scores, field by field, as a separate run with budget k (k = 0, 1, 6, 7, 12, 16); prefixes past the end of the log are the whole episode; replay does not change the episode. Probing time is the sum of the executed durations; a non-aborted probe lasts its full skill, an aborted one less, a missing sensor 0 s; the executed duration is the physics' alive steps times DT. The violation types are the ones the physics reports, and their counts match the probe log. Report confidence equals P(level >= k), lies in [0, 1] and does not increase across tiers. |
| `test_assess_policies.py` | 3 | No flags give the stage-1 paths. `--policies` and `--out` override them. `--policies` is the folder actually read, and nothing is written when it holds no policies. |

## 2. Test categories

- **Unit tests of pure functions** (`tiers`, `belief`, `policies` control law, `training` helpers): exact values, edge cases, invariants.
- **Physics checks** (`physics`, `probes`): behaviour that follows from the equations, with the numbers taken from the parameters.
- **Interface checks** (`assessors`, `gating`, `planner`): the shapes and rules that the layers rely on.
- **End-to-end checks** (`episode`): a full run from table to report, on the real platform.

## 3. What is not tested

These are known gaps. Each is either listed in the stage-1 plan as acceptance, or covered by an experiment instead of a test.

| Gap | Why it is not a unit test | Where it is checked instead |
|---|---|---|
| The learned table on all 18 skills | Training all 18 takes 19 s, which is too slow for every test run. | `experiments/assess_policies.py` (signature and agreement). |
| Evolution strategies matches a grid search on the two-gain problem | Planned in `stage1-plan.md` section 3.11 item 5. **Not implemented.** | Not checked. The determinism test (`test_training_is_deterministic_for_a_seed`) is in place. |
| The gate's trust decisions on the real settings | Results depend on 18 full training runs. | `results/policies/arm6/*.json` (all 18 trusted). |
| The whole-robot gating comparison with learned policies | Requires a full learned table and a long run. | Only the scripted comparison was run (`results/comparison_robot.csv`). |
| Numerical accuracy of the physics against a reference integrator | No independent reference is available in the repository. | Checked by hand for the healthy shoulder sag (g/kp). |
| Hardware behaviour | Stage 1 is simulation only. | See [robotics-limitations.md](robotics-limitations.md). |

## 4. When a test fails

- **A physics test fails** after a change to `physics.py` or `tasks.py`. The scripted numbers in `test_probes.py` are the first thing to check, because they encode the design. If the numbers change on purpose, update the test and the table in [model.md](model.md) together.
- **A training determinism test fails**. The usual cause is a new random draw that is not seeded, or a change in draw order. The seeds are in `configs/training.yaml`.
- **A gating test fails**. Check the status rules in [architecture.md](architecture.md), section 5, step 7, and the gating module's docstring. The chain and robot modes differ only in `_prerequisites`.
- **A planner test fails** after a change to the score. The utility is `information − risk_weight · risk − energy_weight · energy / energy_scale`, and the risk is the weighted violation probability.

## 5. Static checks

The code is checked with `pyflakes` for unused names and imports. It is not run in the test suite. To check:

```bash
python3 -m pip install pyflakes
python3 -m pyflakes src experiments tests
```

The stage-1 code passed with no output. This was not re-run for the 2026-10-15 changes, because `pyflakes` is not a declared dependency.

## 6. Tested environment

The suite and the 2026-10-15 rerun (`docs/assessment.md`, section 8) were run in this environment. It is the one tested, not a claim that these versions are the only ones that work.

| Item | Value |
|---|---|
| OS | Windows 10 Home, build 19045, x86-64 |
| CPU / RAM | Intel i5-10300H / 8 GB. No GPU is used. |
| Python | CPython 3.11.9, in `.venv` (invoked as `.\.venv\Scripts\python.exe`, without activating the venv) |
| Pinned (`requirements.txt`) | numpy 2.4.6, PyYAML 6.0.3, pytest 9.1.1 |
| Transitive | colorama 0.4.6, iniconfig 2.3.1, packaging 26.3, pluggy 1.7.0, Pygments 2.21.0 |
| Checks | Each package's `Requires-Python` allows 3.11; numpy and PyYAML are `cp311-win_amd64` wheels; `pip check` reports no conflicts. numpy 2.4 needs Python 3.11 or newer. |

The stage-1 numbers in `docs/learning.md` were produced on WSL2 with Python 3.11 and unrecorded package versions. Retraining here gave the same 18 policies, except two values that differ in the last digit: elbow T1 `kp_mult` 0.511 (documented 0.512) and wrist_1 T1 `kd_mult` 1.897 (documented 1.898). This is consistent with floating-point differences between numpy versions, which is why the versions are now pinned.

## 7. Changes on 2026-10-15

| Change | Tests |
|---|---|
| Pinned dependencies; `.pytest_cache/` ignored | — (checked with `pip install -r requirements.txt --dry-run` and `git check-ignore`) |
| Probabilistic and tie-aware scores in `src/evaluate.py`; paired bootstrap | `test_evaluate.py` |
| Per-trial, paired and ceiling outputs in `compare_methods` | `test_compare_methods.py` |
| `--policies` and `--seed-start` for `compare_methods` (2026-10-16) | `test_compare_methods.py` (13 new) |
| M1, measurements (2026-10-16): prefix evaluation, probing time, safety events by type, per-tier confidence, `--curve-out` | `test_measurements.py` (12), `test_compare_methods.py` (3 new). Also checked outside the suite: on seeds 100-101 the default run reproduces all 48 stored rows of `full_scripted_2026-10-16` and of `full_2026-10-16`, and the curve rows at k = 4 .. 24 equal them (48/48 each). |
| Per-probe noise streams in `src/episode.py`, with `shared` for the stage-1 behaviour | `test_episode.py` (5 new). Also checked outside the suite: `shared` reproduces the pre-change `episode.py` exactly in 45 of 45 episodes. |
| `--policies` and `--out` for `assess_policies` | `test_assess_policies.py`. With the default paths its output is byte-identical to the pre-change script's. |
| Locked-joint torque sign | `test_physics.py` (new test; it fails on the old code). All 240 scripted and learned table predictions are byte-identical before and after, because every reading uses the absolute torque. |
| Chained comparison in `test_the_real_settings_load` | `test_training.py` (fixed assertion and a new seed-range test) |
