# Gate D Report — Timing Factorial (fixed training, 5 seeds, 100 episodes)

Analysis of the preregistered confirmatory run
(`docs/gate_d_preregistration.md`). Written after — and only after — the
full evaluation completed. Order of sections follows the preregistration
exactly; no endpoint or condition was switched post hoc.

## Provenance

- Preregistration + run-path safeguard committed as `3feb586` before
  launch (parent `6057837` "Ready for Gate D"). A follow-up commit
  `c280350` added log files only; no training/evaluation semantics changed
  between launch and analysis.
- Training: `--run --seeds 5 --stamp gate_d_v3_fixedtrain_5seed --workers 6`,
  30/30 runs `completed` on the first pass (no failures, no resumes),
  exit 0, ~22.5 h wall clock.
  Manifest: `models/factorial_manifest_gate_d_v3_fixedtrain_5seed.json`.
- Evaluation: `eval_dt_robustness.py --manifest <exact manifest>
  --episodes 100` (preregistered budget), exit 0, ~4.9 h, semantics v3,
  all 30 checkpoints integrity-verified, `excluded_runs: []`.
  Results: `results/dt_robustness_factorial_20260724_020219.json`.
- Sign convention: positive = physical/first variant more robust
  (cost metrics negated). All contrasts are paired diff-in-differences on
  shared environment seeds, per matched training seed.

## 1. Primary contrast (preregistered)

**Gaps reward robustness, `physsnn-physcfc-masked` vs
`nomsnn-nomcfc-masked`:**

| training seed | DiD (reward) | 95% CI |
|---|---|---|
| 0 | +13.62 | [−4.61, +32.87] |
| 1 | +8.00 | [−4.21, +19.28] |
| 2 | +8.50 | [−3.14, +20.41] |
| 3 | −4.46 | [−17.88, +9.18] |
| 4 | −16.57 | [−34.00, +1.06] |

**Across-seed mean +1.82 (median +8.00), positive direction in 3/5
seeds.** No per-seed CI excludes zero.

Against the preregistered thresholds:

1. Positive mean across seeds — **pass, barely** (+1.82).
2. Same direction in >= 4/5 seeds — **FAIL** (3/5).
3. Practically meaningful size — **FAIL**: the mean (+1.8) is small
   relative to the across-seed spread (per-seed effects span −16.6 to
   +13.6, sd ≈ 12).
4. No substantial collision penalty — pass (see guardrails).
5. Acceptable absolute performance — **marginal FAIL**: physphys seed 1
   essentially failed to train (fixed reward 0.47, success 23/100).
6. Coherent main effects — **FAIL** (see §4: CfC main effect ≈ 0,
   interaction negative).

**The preregistered primary claim is NOT supported.**

## 2. Quality guardrails

Fixed quality, retained quality under gaps, and collision duration per run
(reward mean / success k of 100 / collision s):

`physsnn-physcfc-masked`: fixed reward 0.5–27.0 (success 23–99/100);
gaps reward −1.3–21.6 (success 20–66/100); fixed collision 0.4–1.1 s,
gaps collision 0.3–1.5 s.

`nomsnn-nomcfc-masked`: fixed reward 2.1–20.2 (success 33–94/100);
gaps reward −5.8–12.9 (success 28–84/100); fixed collision 0.4–1.9 s,
gaps collision 0.4–2.3 s.

- Collision-duration robustness DiD at gaps: mean +0.16 s in favor of
  physphys, 3/5 seeds positive, all CIs cross zero — no collision penalty,
  but no benefit either.
- The dominant feature is **within-cell training-seed variance**: both
  cells contain one near-failed policy (physphys seed 1; nomnom seed 4 has
  fixed reward 2.1) and fixed-condition success ranges over ~60 points
  within each cell. No "robust because bad everywhere" artifact — but
  quality itself is unstable across seeds.

## 3. Secondary endpoints and conditions (never promoted)

- Gaps success-rate robustness DiD: mean −0.03, 2/5 positive
  (seed 1 +0.23*, seed 4 −0.31*; * = CI excludes zero, opposite signs).
- Gaps final-goal-distance robustness DiD: mean −0.03, 2/5 positive
  (same two seeds significant in opposite directions).
- Reward robustness DiD at other conditions: wide +1.27 (3/5),
  strong +2.55 (3/5), mild +5.88 (4/5).

The only 4/5-consistent pattern (mild) is a secondary condition with the
smallest disturbance, and per the preregistration it cannot be promoted to
a claim.

## 4. Masked 2x2 (gaps, reward robustness)

| effect | across-seed mean | direction |
|---|---|---|
| SNN physical-time main effect | +2.07 | 3/5 positive |
| CfC physical-time main effect | −0.25 | 2/5 positive |
| interaction | −9.37 | 2/5 positive |

No per-seed CI excludes zero for any effect. The interaction trends
negative (combining both physical mechanisms is, if anything, worse than
additive), which contradicts a coherent "physical time helps" story.

Exploratory note (hypothesis-generating only, not preregistered as
primary): the simple effect of physical SNN timing *at nominal CfC*
(`physsnn-nomcfc-masked` vs `nomsnn-nomcfc-masked`) was positive in
**5/5 seeds** (mean +6.75, every CI crossing zero). If any signal
survives here, it is "physical SNN timing helps when the CfC is nominal"
— worth a targeted follow-up, not a claim.

## 5. Visible-dt controls (gaps, reward robustness)

- masked vs visible at phys/phys: mean −1.46, 3/5 positive.
- masked vs visible at nom/nom: mean −0.84, 2/5 positive.

Explicit dt input neither helps nor hurts detectably; it does not
"recover" any benefit because there is no reliable benefit to recover.

## 6. Outcome mapping (predeclared tree)

**Outcome F — training-seed variance dominates.** With a strong shade of
E (honest null): the across-seed spread of the primary effect (sd ≈ 12) is
roughly 6x its mean (+1.8), individual seeds within a single cell range
from failed policies to 99% success, and two secondary endpoints produced
statistically significant effects in *opposite directions* on different
seeds. No architecture claim is warranted from this run — neither
"physical time helps" (A/B/C), nor "visible dt suffices" (D), nor a clean
null (E), because the instrument (CMA-ES training at this budget) is
noisier than the effect being measured.

Honest calls:

- The preregistered primary hypothesis is not supported at 5 seeds.
- No evidence of harm from physical-time propagation either.
- The single consistent direction (physical SNN at nominal CfC, 5/5) is
  exploratory and needs its own preregistered test before being believed.
- The binding constraint is training stability, not evaluation power:
  100 episodes gave tight per-seed CIs; the across-seed variance is the
  problem. Before any Gate E, either (a) more seeds per cell, (b) a
  training protocol with lower variance (longer budgets, restarts, or
  seed-selection rules declared in advance), or (c) an effect size large
  enough to clear this noise floor.
