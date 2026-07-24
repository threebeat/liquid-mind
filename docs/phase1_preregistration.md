# Phase 1 Preregistration — Stable Specialist Representations

Experiment family: `specialists_phase1_v1` (branch `specialists_phase1_v1`,
off `main` @ `7745306` "Gate D Finished"). Gate D artifacts are frozen and
never touched. **No PASS/REVISE/RESTART verdict is rendered by the agent at
any stage**: the final deliverable maps numbers onto the decision-tree
criteria for the user to judge.

Status: DRAFT — frozen items are marked; pilot-derived fields are filled in
during Stage 2 and locked before any confirmatory data is touched.

## 1. Hypotheses under test

1. Two frozen, input-isolated echo-state specialists (lidar 128, body 64)
   with trainable 16-d message readouts support short-horizon prediction of
   their own streams better than persistence.
2. Fusing the two 16-d messages (relational predictor) beats the best
   isolated specialist (complementarity C > 0).
3. The modular frozen-reservoir system is competitive with a
   parameter-matched, fully-trainable monolithic GRU (modular advantage
   M = L_GRU − L_modular; non-inferiority of interest, not superiority).
4. Physical-time leak (alpha_t = 1 − exp(−dt/tau)) vs nominal leak
   (alpha_t = alpha_0) is a controlled ablation per stream (local-timing
   hypothesis from Gate D); no superiority is declared before confirmation.

## 2. Data and windows (frozen)

- Buffers: `training/replay_buffer.py` schema v2; per-episode ordered
  `obs (T+1, 22)`, `actions (T, 2)`, `dts (T,)`. Actions are NEVER averaged;
  `sample_chunks_by_duration` is not used anywhere in Phase 1.
- Ordered windows (`phase1/dataset.py`): context L = 15 decisions; horizon
  endpoints 0.25 / 0.5 / 1.0 / 2.0 s → 8 / 15 / 30 / 60 decisions at the
  nominal 30 Hz cadence. Windows never cross episode boundaries; anchors
  need ≥ 5 real context steps (left-padded with a valid mask) and the full
  60-decision future. Anchor stride: train 1 for smoke; pilot/confirmatory
  strides are declared in the stage configs inside the runners and are the
  same for every architecture.
- Splits are by COMPLETE episode, with a saved split manifest (episode
  indices + buffer SHA-256) reused identically by every architecture.
  Deterministic given the data seed.
- Env-seed ranges (disjoint, reserved): smoke = legacy buffer
  (seeds 10000–10079); pilot train/val/test = 20000–20299 / 21000–21074 /
  22000–22099 (300/75/100); confirmatory = 30000–30499 / 31000–31099 /
  32000–32199 (500/100/200). Pilot-test episodes are development data after
  the pilot and are BARRED from confirmatory use. Test episodes are never
  used for hyperparameter selection.

## 3. Architectures (frozen)

- `ReservoirSpecialist` (`phase1/esn.py`): frozen sparse W_in, W_rec
  (buffers; zero trainable reservoir parameters), deterministic per seed,
  W_rec rescaled to the configured spectral radius; update
  r_{t+1} = (1 − alpha_t) r_t + alpha_t tanh(W_in x_t + W_rec r_t + b);
  trainable message readout with input skip m_t = W_msg [r_t; x_t] + b_msg,
  m_t ∈ R^16.
- Lidar specialist: 128 units; input = 16 rays + normalized dt (+ declared
  wheel-action input only under an explicit config flag; default off).
  No goal, body state, pose, or labels. Body specialist: 64 units; input =
  previous wheel action, forward speed, yaw rate, normalized dt. No lidar.
- Predictors (`phase1/predictors.py`): local D_L(m_L, a_{t:t+tau}) → future
  lidar and D_B(m_B, a_{t:t+tau}) → future body, heads
  Linear(16+32, 64)–ReLU–Linear(64, target); relational
  R(m_L, m_B, ordered actions) → future lidar+body. Ordered future actions
  are consumed by a small GRU action encoder (hidden 32), endpoint hidden
  state per horizon. Adam trains readouts/encoder/heads only; ridge is
  available for linear probes; reservoirs stay frozen.
- Comparators: persistence; linear autoregressive ridge (flattened ordered
  context + ordered future actions); differential-drive kinematic body
  model (steady-state wheel map, rays persist); raw-history MLP over the
  same flattened context with the same action encoder and head structure;
  parameter-matched monolithic GRU (same inputs/targets/splits/optimizer,
  hidden size chosen to match the modular TRAINABLE count; trainable and
  total parameters plus runtime are reported for both); isolated-ESN
  (fusion off); nominal vs physical-time leak variants.

## 4. Endpoints (frozen)

Losses are endpoint MSE in normalized observation units; the combined loss
at a horizon is 0.5·MSE_lidar + 0.5·MSE_body. The modular system's output
for S is the RELATIONAL (fused) prediction.

- **Primary**: S = L_model / L_persistence on the combined loss at 0.5 s
  (lower is better), on the stage's test split.
- **Secondary**: the same ratio at 0.25 / 1.0 / 2.0 s; per-stream lidar MSE
  and body MSE; front-clearance MAE (min over front rays {14, 15, 0, 1});
  temporal-partition consistency of the physical-time leak (state difference
  when every context interval is split in two); message linear-probe R²
  (m_L → current and 0.5 s front clearance; m_B → 0.5 s body); training
  wall-clock; inference cost per window; across-seed variance of S.
- **Complementarity** C = L_best_isolated − L_combined at 0.5 s, where
  L_best_isolated is the isolated (fusion-off) run's better local loss per
  target stream combined as above, and L_combined is the fused run's
  relational loss. C > 0 means fusion helps.
- **Modular advantage** M = L_GRU − L_modular at 0.5 s (positive favors the
  modular system). The non-inferiority margin for the decision tree is
  0.05 · L_persistence (i.e. S_modular − S_GRU ≤ 0.05).

## 5. Bounded ESN configuration grid (frozen)

spectral radius ∈ {0.7, 0.9, 1.1}; leak alpha_0 ∈ {0.1, 0.3, 0.6};
recurrent sparsity ∈ {0.05, 0.1}; input scale ∈ {0.1, 0.5, 1.0}
(54 combinations; input density 0.5, bias scale 0.1, message dim 16 fixed).

Bounded pilot selection procedure (no open-ended search):
1. Screen all 54 combinations with RIDGE readouts/heads (linear, closed
   form) on pilot train → validation combined loss at 0.5 s, 2 reservoir
   seeds each; rank by mean validation loss.
2. Retrain the top 3 with the full Adam pipeline, 5 reservoir/model seeds
   each; select the ONE family with the best mean validation S; freeze it.
3. Leak-mode ablation at the frozen family: {nominal, physical} × {lidar,
   body} (4 combos), 5 seeds each, validation S; choose per-stream leak
   mode; freeze.
Tau for physical-time leak is derived from alpha_0 at the nominal interval:
tau = −dt_nom / ln(1 − alpha_0).

## 6. Reservoir technical-failure exclusion rules (frozen)

Evaluated on training-set state trajectories (`reservoir_health`):
NaN/Inf anywhere; saturation (fraction of |state| > 0.99 exceeds 0.90);
constant collapse (mean per-unit std over time < 1e-5); effective-rank
collapse (participation ratio of the state covariance spectrum < 2);
state explosion (max |state| > 1e3). A failing reservoir is regenerated
with seed + 1000 and the exclusion + regeneration is LOGGED in the stage
results JSON. Training runs with NaN/divergent loss are recorded with
status `nan_loss`, excluded, and regenerated the same way — never silently
tweaked.

## 7. Sample-size rule (frozen; value filled from pilot)

n = ceil((2.8 · sigma_d / delta)^2), delta = 0.05 on S, where sigma_d is
the pilot standard deviation of the PAIRED per-seed difference
S_ESN − S_GRU (seeds paired by index). Minimum 10 seeds per architecture.

- Pilot sigma_d: **TBD (filled from pilot)**
- Computed n: **TBD (filled from pilot)**
- Confirmatory n: **TBD (max(computed n, 10)) seeds per architecture**

## 8. Frozen pilot choices (filled during Stage 2, locked before Stage 3)

- ESN configuration family: **TBD (bounded selection of §5)**
- Leak mode per stream: **TBD (2×2 leak ablation, validation S)**
- Confirmatory seeds per architecture: **TBD**

## 9. Confirmatory analysis plan (frozen)

Identical splits for every architecture; ONE evaluation on the 200
untouched confirmatory test episodes after all training completes.
Message-ablation tests on the fused system: per message (m_L, m_B) —
zeroing, cross-episode shuffle, and random-vector replacement (matched
moments), each reported as the change in relational test loss.
Efficiency metrics: training wall-clock, inference cost per window,
trainable/total parameters. Uncertainty: hierarchical bootstrap —
episodes resampled within seed, seeds resampled as replicates (2000
draws); NO window-level pseudo-replication. Paired ESN−GRU differences use
seed pairing by index. Leakage checks: split-manifest disjointness and
buffer checksums are re-verified at load time.

## 10. Out of scope (frozen)

Global workspace, language, goal formation, sensor-fault labels, online
plasticity, planner/policy changes, SNN reservoirs, Gate D retraining,
averaging action sequences, using test episodes for selection, and any
agent-issued gate verdict.
