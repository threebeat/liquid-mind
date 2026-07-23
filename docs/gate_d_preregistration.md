# Gate D Preregistration — Timing Factorial (fixed-training, 5 seeds)

Preregistered 2026-07-22, BEFORE any full-budget factorial result was seen.
This document is frozen: endpoints, conditions, thresholds, and evaluation
budget below may not be changed after launch. Any deviation must be reported
as a deviation.

- **Stamp:** `gate_d_v3_fixedtrain_5seed`
- **Manifest:** `models/factorial_manifest_gate_d_v3_fixedtrain_5seed.json`
- **Code state:** committed together with this file (parent commit
  `6057837` "Ready for Gate D"); no training/evaluation semantics changes
  between launch and analysis. Semantics version 3.
- **Preregistered evaluation budget:** **100 episodes per condition per
  run** (chosen now; not 50-then-inspect-then-100).

## Experimental units

- Six architecture cells:
  `physsnn-physcfc-masked`, `physsnn-nomcfc-masked`,
  `nomsnn-physcfc-masked`, `nomsnn-nomcfc-masked`,
  `physsnn-physcfc-visible`, `nomsnn-nomcfc-visible`.
- Five independent training seeds per cell (0-4) -> 30 trained policies.
- Fixed nominal timing during training (no `--train-jitter`).
- Identical CMA seed indices, generation seed streams, and validation banks
  across cells (matched blocked replicates); deterministic initialization —
  the same training seed yields the same initial parameter vector in every
  cell.
- Full budgets: 60 generations, population 24, 2 screening episodes per
  candidate, 12 validation episodes; 7,944 evolvable parameters per cell.
- Five evaluation timing conditions: fixed, mild, strong, wide, gaps.

## Primary research question

Does physical-time internal state propagation retain more control
performance than nominal-step propagation under irregular observation
timing, when direct dt exposure is masked?

## Primary comparison and endpoint

- **Cells:** `physsnn-physcfc-masked` vs `nomsnn-nomcfc-masked`.
- **Condition:** **gaps** (rare 250-1000 ms observation dropouts). Chosen
  because fixed/mild/strong jitter already produced a null result in pilot
  data, and gaps create a real divergence between physical elapsed time and
  one nominal update.
- **Endpoint:** reward robustness = `reward_gaps - reward_fixed`, computed
  per environment seed within each training run.
- **Effect:** `robustness_physphys - robustness_nomnom` per matched
  training seed, summarized across the five seeds. Positive = the
  physical/physical cell retains more reward.

## Secondary endpoints (reported, never promoted post hoc)

Success-frequency robustness; final-goal-distance robustness;
collision-duration robustness (cost metrics sign-flipped so positive =
more robust); absolute retained reward under gaps; absolute success under
gaps; path efficiency. Secondary disturbance conditions: wide, strong,
mild. Whichever condition "looks best" afterward is NOT elevated into the
primary claim.

## Mechanism questions

- Masked 2x2 (four masked cells): does physical SNN evolution help; does
  physical CfC evolution help; do they interact?
  Formulas (D_ij = robustness with SNN factor i, CfC factor j, 1=physical):
  SNN = ((D10-D00)+(D11-D01))/2; CfC = ((D01-D00)+(D11-D10))/2;
  interaction = D11-D10-D01+D00.
- Visible-dt controls: can explicit timing-conditioned action selection
  recover the benefit without physical-time state propagation?
  (`masked_vs_visible` contrasts at phys/phys and nom/nom.)

## Quality guardrail

A policy is not called robust merely because it performs badly everywhere.
All three preregistered quantities are reported per run: fixed quality,
robustness (degradation), and retained disturbed quality. A positive
robustness claim additionally requires acceptable fixed reward, fixed
success, and disturbed absolute performance for the physical cell.

## Interpretation thresholds (5 seeds; no single p-value decides)

A credible positive physical-time effect requires ALL of:

1. positive mean primary effect across training seeds;
2. consistent direction in at least 4 of 5 matched seeds;
3. a practically meaningful reward difference (not merely nonzero);
4. no substantial collision-duration penalty;
5. acceptable absolute performance (guardrail above);
6. coherent SNN/CfC main effects (the 2x2 pattern does not contradict the
   headline contrast).

Five training seeds are a small architecture sample. This is the first
controlled factorial result, not the final word on continuous-time
learning.

## Analysis order (fixed)

1. Primary contrast (gaps reward robustness, physphys vs nomnom).
2. Guardrails (fixed quality, retained quality, collision duration).
3. Secondary endpoints and conditions.
4. Masked 2x2 main effects and interaction.
5. Visible-dt controls.
6. Map the pattern to the predeclared outcome tree (A: phys/phys clearly
   wins; B: CfC-time only; C: SNN-time only; D: visible dt suffices;
   E: nothing matters — honest null; F: seed variance dominates — no
   architecture claim until training variance is controlled).

## Commands (exact)

```
python scripts/run_timing_factorial.py --preflight --seeds 5
python scripts/run_timing_factorial.py --run --seeds 5 --stamp gate_d_v3_fixedtrain_5seed --workers 6
python scripts/eval_dt_robustness.py --manifest models/factorial_manifest_gate_d_v3_fixedtrain_5seed.json --episodes 100
```

No `--train-jitter`. No `--force` (resume = same command, same stamp;
verified checkpoints become `skipped_valid`, failures re-attempt with
recorded history). Evaluation uses the explicit manifest path, never
`--all-factorial`.
