# Liquid-Mind

Liquid-Mind is an operational experimental architecture combining spiking
sensory encoding, continuous-time recurrent control, latent future
prediction, and CEM-based hierarchical subgoals, evaluated on a
differential-drive navigation task (PyBullet) under irregular sensor timing.

## Honest status of the results

Use this language; the stronger claims have not been earned yet.

**What has been demonstrated (preliminary):**

- The complete SNN → CfC → world-model → CEM hierarchy executes end to end.
- The reactive liquid policy learned meaningful navigation behavior.
- Bounded mean-matched action-hold jitter did not substantially break either
  PPO or the liquid policy.
- The world model predicted held-out goal-distance change better than a
  persistence baseline over the evaluated horizon (latent error ratio 0.88;
  0.44 m two-second goal-distance error vs 0.61 m persistence).
- A ten-episode U-trap pilot yielded one hierarchical escape and a lower
  reported mean final distance. That is a pilot result, not statistical
  evidence.

**What has NOT been demonstrated:**

- Continuous-time state evolution outperforming matched discrete-time
  recurrence.
- Separation of direct dt exposure from physical-time internal dynamics
  (the pre-2026 observations carried true dt in channel 21 unmasked).
- An exact continuous-time spiking process (approximate firing-rate
  equality in one regime does not prove it; the legacy encoder emitted at
  most one binary spike per observation).
- World-model validity on all obstacle quantities the planner uses.
- Reliable U-trap improvement (0/10 vs 1/10 establishes nothing).
- Planning value isolated from extra training, shaping reward, or "any
  changing subgoal" effects (the hierarchy was warm-started and trained
  further).
- Online reasoning or hidden-state identification (jitter robustness is not
  that).
- That the architecture rather than the optimizer explains the PPO/liquid
  performance gap (47/50 PPO vs 33/50 liquid may be a CMA-ES artifact).

The next research phase isolates physical-time propagation from explicit dt
conditioning, validates planner-relevant world-model predictions, and
introduces asynchronous sensing and hidden dynamics changes that require
genuine online belief adaptation.

## Architecture

```
obs (22 ch, dt channel maskable)
  -> SpikeEncoder (analytic event-count LIF, multi-spike, held-current ZOH)
  -> shared 54-d sensory bus [spikes|obs] (absent modalities zero-filled)
  -> identical linear adapter (32 wide) -> CfC liquid policy (dt-aware)
        ^ subgoal latent, replanned every planner.period_seconds of
          PHYSICAL time: obs -> JEPA world model -> CEM planner
```

Key semantics (all recorded in checkpoint metadata, all validated on load):

| Config flag | Meaning |
|---|---|
| `agent.snn_semantics` | `event_count` (analytic multi-spike LIF; exact subthreshold propagation, exact crossing times under held current, multiplicity preserved) / `sampled_binary` (legacy one-spike-per-observation ablation) / `membrane`, `rate` (non-spiking controls) |
| `agent.timing_convention` | `causal` (SNN integrates the PREVIOUS measurement across the elapsed interval; first assimilation at elapsed time zero) / `irnn` (legacy irregular-RNN convention, clearly named, not called an exact physical model) |
| `agent.mask_direct_dt` | replace the raw dt observation channel with its nominal value before any network sees it (no timing side-channel) |
| `agent.snn_time_aware` / `cfc_time_aware` | physical elapsed time vs nominal fixed step per module (the timing factorial axes) |
| `agent.use_input_adapter` | shared 54-d sensory bus + identical adapter shapes before the CfC (fixed downstream architecture; report active parameter counts — not total-capacity matching) |

The environment integrates collision contact at every physics substep
(cost = lambda * measured contact duration) and clamps the final control
interval so every schedule ends at exactly `episode_seconds` — deterministic
tests prove fixed and irregular schedules end at identical simulated times.

## Provenance rules

- Checkpoints are `{"state", "meta"}` bundles: git commit + dirty flag,
  resolved config, seeds, parameter count, budget, timing distribution,
  package versions, canonical state checksums (verified on load), and
  parent/warm-start / world-model references. Loading validates a
  compatibility block (semantics version, input mode, timing flags, dt
  masking, SNN semantics, sensory bus, …) and fails with an actionable
  error on mismatch or checksum failure.
- Replay buffers are fingerprinted (`experience_sv{N}_{sha8}.npz`) with
  embedded metadata; incompatible or pre-provenance `experience.npz` files
  are refused unless `--allow-legacy-buffer` is set and recorded.
- Nothing overwrites an existing checkpoint or result without `--force`;
  result JSONs are timestamped and carry the exact checkpoint checksums
  evaluated.
- Pre-provenance artifacts were imported via `python main.py import-legacy`
  into `models/legacy/*_legacy.pt`; they load only with `--allow-legacy`
  and can never be mistaken for new experiments. Coverage does not yet
  extend to Stable-Baselines PPO zip internals.

## Workflow

```
.\env\python.exe -m pytest tests        # gate 0: all tests must pass first
python main.py check                    # env smoke test
python main.py import-legacy            # once: wrap pre-provenance artifacts
python main.py train-baseline           # PPO baseline (--force to retrain)
python main.py train-policy             # CMA-ES reactive policy
python main.py eval-dt                  # timing-disturbance ladder
python main.py train-wm                 # world model + planning gate
python main.py train-hier               # hierarchical (refuses ungated WM)
python main.py eval-hier                # 100-seed attribution evaluation
python scripts/run_timing_factorial.py  # dry-run of the timing factorial
python scripts/run_timing_factorial.py --smoke --run
python scripts/eval_dt_robustness.py --all-factorial
```

CMA-ES selection uses common random numbers within each generation and
saves the **validation-best** candidate (held-out seed bank), not the
noisiest training best.

## Go / no-go gates

- **LIF gate** — equal continuous input trajectories partitioned at
  15/30/60/120 Hz and irregularly must produce consistent event counts and
  terminal states, including rates above the lowest sampling frequency
  (`tests/test_lif_semantics.py`; tolerances declared in the file).
- **World-model gate** — status `passed` / `failed` / `incomplete`. Must
  beat persistence with minimum effect sizes and episode-clustered
  uncertainty on goal/bearing/quads; beat or meaningfully match the
  exact differential-drive kinematic baseline; keep left/right confusion
  below a limit on enough decisive cases; and never treat undefined
  false-safe rates (too few dangerous windows) as a pass. Verdict is
  stored in the checkpoint; `train-hier` / `eval-hier` refuse
  failed/incomplete/ungated models without `--override-wm-gate`.
  Duration-averaged actions remain the transition baseline; an
  order-sensitive action-duration encoder is next-stage work.
- **Hierarchy gate** — status `passed` / `failed` / `incomplete`. Claim
  planner value only if active planning beats zero/random/shuffled/
  heuristic controls **and** original reactive **and** equal-extra
  reactive training (`reactive-extra-budget`). Incomplete until the
  equal-extra checkpoint exists. Primary endpoint: paired success
  improvement; secondary: final distance; safety: collision duration.
- **Continuous-time gate** — claim continuous-time value only if
  physical-time variants beat matched nominal-time variants with direct dt
  masked, shared downstream adapter shapes, budgets equal, multiple
  training seeds, and disturbances that genuinely require temporal
  inference. Report active evolvable parameter counts honestly.
- **Real-time reasoning gate** — claim online adaptation only after
  hidden mid-episode dynamics changes with recovery/identification metrics
  (Stage 4; not yet implemented).

## Statistics policy

Per-episode records are saved verbatim in every result JSON. Aggregates use
bootstrap CIs (mean/median), Wilson intervals (success), paired bootstrap
and exact McNemar tests on identical seed lists. When `--cem-repeats > 1`,
repeats are aggregated within each environment seed before pairing; within-
environment planner variance is reported separately. Ten-episode runs are
pilots and are labeled as such.

## Roadmap (deferred to later iterations)

- Stage 4: independent per-sensor clocks with capture/delivery timestamps,
  delays, dropouts, staleness; hidden plant changes (motor gain, slip,
  latency, sensor bias) with a system-identification head and
  recovery-time metrics. The replay schema already carries capture and
  delivery timestamps in preparation.
- Priority 8: order-sensitive transition model (GRU-D / CfC / kinematic +
  learned residual) versus the **retained** duration-averaged-action
  baseline; compare on planner-relevant grounded errors, false-safe rate,
  CEM-selected on-policy prediction, and hierarchy performance before
  replacing the averaged predictor.
- Priority 11: full optimizer-matched architecture comparison (MLP/GRU/CfC
  x PPO/CMA-ES).
- Stage 5: uncertainty ensembles and uncertainty-penalized CEM.

## Setup

PyBullet has no prebuilt Windows wheel for modern Pythons; this project uses
a local conda-forge environment in `./env` (Python 3.11 + pybullet + numpy
via micromamba), with the pip packages from `requirements.txt` installed
into it:

```
.\env\python.exe -m pip install -r requirements.txt
```
