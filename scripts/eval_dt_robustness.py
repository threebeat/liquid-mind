"""Timing-robustness evaluation — with the confounds controlled.

Agents are evaluated under a progressive timing-disturbance ladder
(Priority 4, level A). Design notes (deliberate, to keep the comparison
fair):
  - jitter substep ranges are centered on the nominal 8 substeps, so the MEAN
    control rate is identical across jitter conditions — only the variance
    changes;
  - episodes truncate at exactly the same simulated horizon and time costs
    are charged per unit of simulated time, so all conditions face the same
    physical problem;
  - the "gaps" condition additionally injects rare 250-1000 ms observation
    gaps (env.gap_prob) on top of mild jitter.

Supports:
  - default liquid_policy.pt + PPO baseline comparison;
  - --checkpoint / --config for a single factorial cell;
  - --manifest / --all-factorial to evaluate every run of a factorial
    manifest. Manifest results are keyed cells -> training_runs -> run_id
    so multiple training seeds never overwrite each other; per-run
    degradation, paired difference-in-differences contrasts at every ladder
    level, and 2x2 factorial main effects/interaction are computed with the
    locked sign convention (positive = physical variant more robust).

Statistical structure (two-level, kept strictly separate):
  - WITHIN a training run: episodes are paired by environment seed across
    ladder levels; per-environment degradation is bootstrapped over env
    seeds (checkpoint-level evaluation uncertainty).
  - ACROSS training runs: each run is reduced to ONE effect estimate per
    metric/level; training seeds are blocked replicates, matched across
    cells for paired contrasts. Episode records are never pooled across
    training seeds as independent observations, and per-checkpoint bootstrap
    draws are never fed into the architecture-level summary. Matched seeds
    share CMA seed / generation streams / eval banks but the trained
    policies still differ by everything downstream of the timing mechanism.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import MODELS_DIR, ROOT, ensure_dirs, load_config
from provenance import checkpoint_ref, gather_provenance, write_results
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv
from scripts.eval_common import (SIGN_CONVENTION, compare, factorial_effects,
                                 paired_did, robustness_by_env, run_episode,
                                 summarize)
from scripts.factorial_io import (discover_latest_manifest,
                                  load_agent_for_checkpoint, load_manifest,
                                  verify_manifest_cell)

# (label, substeps_min, substeps_max, gap_prob) — jitter means equal the
# nominal 8 substeps; "gaps" adds rare 250-1000 ms observation dropouts
LADDER = [("fixed", None, None, 0.0),
          ("mild", 6, 10, 0.0),
          ("strong", 4, 12, 0.0),
          ("wide", 2, 14, 0.0),
          ("gaps", 6, 10, 0.02)]

SEED_BASE = 555_000

# Metrics that get degradation / DiD treatment (see eval_common sign rules)
DEG_METRICS = ("reward", "success_rate", "final_goal_dist",
               "collision_duration")

# Preregistered cross-cell contrasts (a, b, label); positive = a more robust
PREREGISTERED_CONTRASTS = [
    ("physsnn-physcfc-masked", "nomsnn-nomcfc-masked",
     "physphys_vs_nomnom"),
    ("physsnn-physcfc-masked", "nomsnn-physcfc-masked",
     "snn_phys_vs_nom_at_physcfc"),
    ("physsnn-nomcfc-masked", "nomsnn-nomcfc-masked",
     "snn_phys_vs_nom_at_nomcfc"),
    ("physsnn-physcfc-masked", "physsnn-nomcfc-masked",
     "cfc_phys_vs_nom_at_physsnn"),
    ("nomsnn-physcfc-masked", "nomsnn-nomcfc-masked",
     "cfc_phys_vs_nom_at_nomcfc"),
    ("physsnn-physcfc-masked", "physsnn-physcfc-visible",
     "masked_vs_visible_at_physphys"),
    ("nomsnn-nomcfc-masked", "nomsnn-nomcfc-visible",
     "masked_vs_visible_at_nomnom"),
]

# 2x2 factorial over the four masked cells: (snn_physical, cfc_physical)
MASKED_2X2 = {(0, 0): "nomsnn-nomcfc-masked",
              (1, 0): "physsnn-nomcfc-masked",
              (0, 1): "nomsnn-physcfc-masked",
              (1, 1): "physsnn-physcfc-masked"}

DISTURBED_LEVELS = [lvl for lvl, *_ in LADDER if lvl != "fixed"]


def _env_for(cfg, smin, smax, gap_prob):
    if smin is None:
        return NavEnv(cfg, irregular_dt=False)
    c = copy.deepcopy(cfg)
    c["env"]["substeps_min"] = smin
    c["env"]["substeps_max"] = smax
    c["env"]["gap_prob"] = gap_prob
    return NavEnv(c, irregular_dt=True)


def _liquid_actor(cfg, allow_legacy, checkpoint=None):
    path = checkpoint or os.path.join(MODELS_DIR, "liquid_policy.pt")
    agent = HybridAgent(cfg, mode="reactive")
    agent.load(path, allow_legacy=allow_legacy)
    return agent.act, agent.reset, agent


def _baseline_actor(cfg, allow_legacy):
    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(MODELS_DIR, "ppo_baseline.zip"))
    return (lambda obs, dt: model.predict(obs, deterministic=True)[0],
            lambda: None, None)


def _eval_actor(name, act, reset, cfg, episodes, allow_legacy=False,
                checkpoint_path=None):
    """Run the disturbance ladder for one actor; return per-level records."""
    results = {}
    per_level_records = {}
    for label, smin, smax, gap in LADDER:
        env = _env_for(cfg, smin, smax, gap)
        recs = [run_episode(env, act, reset, SEED_BASE + ep)
                for ep in range(episodes)]
        env.close()
        per_level_records[label] = recs
        s = summarize(recs)
        results[label] = {"summary": s, "episodes": recs}
        print(f"{name:34s} {label:6s}: "
              f"R={s['reward_mean']['point']:7.2f} "
              f"[{s['reward_mean']['lo']:.2f},{s['reward_mean']['hi']:.2f}]"
              f"  (median {s['reward_median']['point']:.2f})  "
              f"success={s['success']['k']}/{s['success']['n']}",
              flush=True)
    for label in per_level_records:
        if label == "fixed":
            continue
        c = compare(per_level_records[label], per_level_records["fixed"],
                    label, "fixed")
        results[label]["vs_fixed"] = c
        d = c["reward_diff"]
        print(f"{name:34s} {label:6s} vs fixed: "
              f"dR={d['mean_diff']:+7.2f} [{d['lo']:.2f},{d['hi']:.2f}] "
              f"{'(excludes 0)' if d['excludes_zero'] else '(includes 0)'}")
    if checkpoint_path:
        results["checkpoint"] = checkpoint_ref(checkpoint_path)
    return results, per_level_records


# ------------------------------------------------ factorial (multi-run) path


def _bootstrap_mean(values, seed=0):
    from scripts.eval_common import bootstrap_ci
    return bootstrap_ci(list(values), np.mean, seed=seed)


def _run_robustness(per_level):
    """Per-level, per-metric robustness for ONE training run.

    Returns {level: {metric: {"per_env": {env_seed: R}, "mean": ci,
                              "missing_env_seeds": ...}}} where higher R =
    more robust (cost metrics already negated in robustness_by_env)."""
    rob = {}
    for label in DISTURBED_LEVELS:
        if label not in per_level:
            continue
        rob[label] = {}
        for metric in DEG_METRICS:
            per_env, missing = robustness_by_env(
                per_level[label], per_level["fixed"], metric)
            rob[label][metric] = {
                "per_env": {str(s): v for s, v in per_env.items()},
                "_per_env_raw": per_env,          # int keys, in-memory only
                "n_env": len(per_env),
                "missing_env_seeds": missing,
                "mean": _bootstrap_mean(per_env.values()),
            }
    return rob


def _across_seeds(values):
    """Spread of per-run point estimates across training seeds. Each run
    contributes exactly ONE value; no episode pooling, no per-run bootstrap
    draws enter here."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"n_runs": 0, "values": [], "mean": None, "median": None,
                "std": None, "min": None, "max": None}
    arr = np.asarray(v, dtype=np.float64)
    return {"n_runs": int(len(arr)), "values": v,
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def _strip_private(obj):
    """Drop in-memory-only keys (leading underscore) before JSON output."""
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _eval_factorial_cells(cells, episodes, allow_legacy, base_cfg):
    """Evaluate every manifest RUN; nest results by cell -> training run.

    Only entries with status completed/skipped_valid are evaluated; every
    entry is integrity-verified against its artifact and planned spec before
    loading, and verification failures exclude the run loudly.
    """
    results = {
        "sign_convention": SIGN_CONVENTION,
        "cells": {},
        "excluded_runs": [],
        "contrasts": {},
        "factorial_effects": {},
        "descriptive_fixed_condition": {},
    }
    runs_by_cell: dict[str, list[dict]] = {}

    for cell in cells:
        cell_id = cell.get("cell_id") or cell.get("name")
        run_id = cell.get("run_id") or f"{cell_id}__s{cell.get('training_seed', 0)}"
        status = cell.get("status", "completed")
        if status not in ("completed", "skipped_valid"):
            results["excluded_runs"].append(
                {"run_id": run_id, "cell_id": cell_id,
                 "reason": f"status={status!r} (not completed/skipped_valid)"})
            print(f"[eval-dt] EXCLUDE {run_id}: status={status}")
            continue
        try:
            verification = verify_manifest_cell(cell)
        except (ValueError, FileNotFoundError) as e:
            results["excluded_runs"].append(
                {"run_id": run_id, "cell_id": cell_id,
                 "reason": f"integrity verification failed: {e}"})
            print(f"[eval-dt] EXCLUDE {run_id}: verification FAILED\n  {e}")
            continue

        ckpt = cell["checkpoint"]
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        agent, cfg, config_source = load_agent_for_checkpoint(
            path, allow_legacy=allow_legacy, base_cfg=base_cfg, cell=cell)
        print(f"\n[eval-dt] === {run_id} ({path}) ===", flush=True)
        ladder_res, per_level = _eval_actor(
            run_id, agent.act, agent.reset, cfg, episodes,
            checkpoint_path=path)

        rob = _run_robustness(per_level)
        run_record = {
            "run_id": run_id,
            "cell_id": cell_id,
            "training_seed": (int(cell["training_seed"])
                              if cell.get("training_seed") is not None
                              else None),
            "attempt": cell.get("attempt"),
            "status": status,
            "config_source": config_source,
            "verification": verification,
            "checkpoint": checkpoint_ref(path),
            "ladder": ladder_res,
            # quality = fixed-condition performance; retained_quality =
            # absolute disturbed performance; robustness = disturbed - fixed
            # (sign-adjusted). All three reported so "degrades less because
            # it is bad everywhere" stays visible.
            "quality_fixed": ladder_res["fixed"]["summary"],
            "retained_quality": {lvl: ladder_res[lvl]["summary"]
                                 for lvl in DISTURBED_LEVELS},
            "robustness": rob,
        }
        results["cells"].setdefault(cell_id, {"training_runs": {}})
        if run_id in results["cells"][cell_id]["training_runs"]:
            results["excluded_runs"].append(
                {"run_id": run_id, "cell_id": cell_id,
                 "reason": "duplicate run_id in manifest"})
            print(f"[eval-dt] EXCLUDE duplicate run_id {run_id}")
            continue
        results["cells"][cell_id]["training_runs"][run_id] = run_record
        runs_by_cell.setdefault(cell_id, []).append(run_record)

    # ---- across-training-seed summaries per cell (one estimate per run) ----
    for cell_id, runs in runs_by_cell.items():
        estimates = []
        for r in runs:
            est = {lvl: {m: r["robustness"][lvl][m]["mean"]["point"]
                         for m in DEG_METRICS}
                   for lvl in r["robustness"]}
            estimates.append({"run_id": r["run_id"],
                              "training_seed": r["training_seed"],
                              "robustness_point": est})
        uncertainty = {}
        for lvl in DISTURBED_LEVELS:
            uncertainty[lvl] = {}
            for m in DEG_METRICS:
                pts = [e["robustness_point"].get(lvl, {}).get(m)
                       for e in estimates]
                uncertainty[lvl][m] = _across_seeds(pts)
        results["cells"][cell_id]["training_seed_estimates"] = estimates
        results["cells"][cell_id]["training_seed_uncertainty"] = uncertainty

    # ---- paired difference-in-differences contrasts (matched seeds) ----
    def _by_seed(cell_id):
        return {r["training_seed"]: r for r in runs_by_cell.get(cell_id, [])
                if r["training_seed"] is not None}

    for a_cell, b_cell, label in PREREGISTERED_CONTRASTS:
        a_runs, b_runs = _by_seed(a_cell), _by_seed(b_cell)
        if not a_runs or not b_runs:
            results["contrasts"][label] = {
                "a": a_cell, "b": b_cell,
                "skipped": f"missing cell(s): "
                           f"{[c for c, r in [(a_cell, a_runs), (b_cell, b_runs)] if not r]}"}
            continue
        matched = sorted(set(a_runs) & set(b_runs))
        unmatched = {"a_only": sorted(set(a_runs) - set(b_runs)),
                     "b_only": sorted(set(b_runs) - set(a_runs))}
        per_replicate = {}
        for s in matched:
            per_replicate[str(s)] = {
                lvl: {m: paired_did(
                    a_runs[s]["robustness"][lvl][m]["_per_env_raw"],
                    b_runs[s]["robustness"][lvl][m]["_per_env_raw"])
                    for m in DEG_METRICS}
                for lvl in DISTURBED_LEVELS}
        across = {lvl: {m: _across_seeds(
            [per_replicate[str(s)][lvl][m]["mean_diff"] for s in matched])
            for m in DEG_METRICS} for lvl in DISTURBED_LEVELS}
        results["contrasts"][label] = {
            "a": a_cell, "b": b_cell,
            "sign": "positive = a more robust than b",
            "matched_training_seeds": matched,
            "n_matched_replicates": len(matched),
            "unmatched_training_seeds": unmatched,
            "per_replicate": per_replicate,
            "across_training_seeds": across,
        }
        for lvl in DISTURBED_LEVELS:
            d = across[lvl]["reward"]
            if d["n_runs"]:
                print(f"[DiD] {label:34s} {lvl:6s} reward: "
                      f"mean={d['mean']:+7.3f} over {d['n_runs']} matched "
                      f"replicate(s)")

        # descriptive fixed-condition comparison per matched replicate
        desc = {}
        for s in matched:
            desc[str(s)] = compare(
                a_runs[s]["ladder"]["fixed"]["episodes"],
                b_runs[s]["ladder"]["fixed"]["episodes"], a_cell, b_cell)
        results["descriptive_fixed_condition"][label] = {
            "note": "descriptive_fixed_condition: absolute fixed-condition "
                    "comparison, not a robustness contrast",
            "per_replicate": desc}

    # ---- 2x2 factorial effects over the four masked cells ----
    cell_runs = {ij: _by_seed(cid) for ij, cid in MASKED_2X2.items()}
    if all(cell_runs.values()):
        matched = sorted(set.intersection(*[set(r) for r in
                                            cell_runs.values()]))
        per_replicate = {}
        for s in matched:
            per_replicate[str(s)] = {}
            for lvl in DISTURBED_LEVELS:
                per_replicate[str(s)][lvl] = {}
                for m in DEG_METRICS:
                    per_replicate[str(s)][lvl][m] = factorial_effects(
                        cell_runs[(0, 0)][s]["robustness"][lvl][m]["_per_env_raw"],
                        cell_runs[(1, 0)][s]["robustness"][lvl][m]["_per_env_raw"],
                        cell_runs[(0, 1)][s]["robustness"][lvl][m]["_per_env_raw"],
                        cell_runs[(1, 1)][s]["robustness"][lvl][m]["_per_env_raw"])
        across = {}
        for lvl in DISTURBED_LEVELS:
            across[lvl] = {}
            for m in DEG_METRICS:
                across[lvl][m] = {
                    eff: _across_seeds(
                        [(per_replicate[str(s)][lvl][m][eff] or {}).get("point")
                         for s in matched])
                    for eff in ("snn_main_effect", "cfc_main_effect",
                                "interaction")}
        results["factorial_effects"] = {
            "cells": {f"snn{i}_cfc{j}": cid
                      for (i, j), cid in MASKED_2X2.items()},
            "formulas": {
                "snn_main_effect": "((D10-D00)+(D11-D01))/2",
                "cfc_main_effect": "((D01-D00)+(D11-D10))/2",
                "interaction": "D11-D10-D01+D00"},
            "matched_training_seeds": matched,
            "per_replicate": per_replicate,
            "across_training_seeds": across,
        }
        for lvl in DISTURBED_LEVELS:
            fx = across[lvl]["reward"]
            if fx["snn_main_effect"]["n_runs"]:
                print(f"[2x2] {lvl:6s} reward: "
                      f"SNN={fx['snn_main_effect']['mean']:+7.3f} "
                      f"CfC={fx['cfc_main_effect']['mean']:+7.3f} "
                      f"IXN={fx['interaction']['mean']:+7.3f}")
    else:
        missing = [cid for ij, cid in MASKED_2X2.items()
                   if not cell_runs[ij]]
        results["factorial_effects"] = {"skipped": f"missing cells: {missing}"}

    return _strip_private(results)


def main(episodes: int = 50, allow_legacy: bool = False,
         checkpoint: str | None = None, config_path: str | None = None,
         manifest: str | None = None, all_factorial: bool = False):
    ensure_dirs()
    base_cfg = load_config(config_path)

    # Factorial / multi-checkpoint path
    if manifest or all_factorial:
        man_path = manifest
        if all_factorial and not man_path:
            man_path = discover_latest_manifest(smoke=None)
            if man_path is None:
                raise FileNotFoundError(
                    "no factorial_manifest_*.json found under models/; "
                    "train with scripts/run_timing_factorial.py --run first")
        print(f"[eval-dt] loading factorial manifest {man_path}")
        man = load_manifest(man_path)
        results = _eval_factorial_cells(
            man["cells"], episodes, allow_legacy, base_cfg)
        results["meta"] = gather_provenance(
            base_cfg, experiment_name="eval_dt_factorial",
            seeds={"seed_base": SEED_BASE, "episodes": episodes},
            extra={"ladder": [{"label": l, "substeps_min": a,
                               "substeps_max": b, "gap_prob": g}
                              for l, a, b, g in LADDER],
                   "manifest": os.path.relpath(man_path, ROOT)
                   if os.path.isabs(man_path) else man_path,
                   "smoke": man.get("smoke")})
        out = write_results("dt_robustness_factorial", results)
        print(f"saved {out}")
        return results

    # Single checkpoint or default liquid + PPO
    results = {"agents": {}}
    if checkpoint:
        path = checkpoint if os.path.isabs(checkpoint) else (
            checkpoint if os.path.exists(checkpoint)
            else os.path.join(ROOT, checkpoint))
        agent, cfg, _src = load_agent_for_checkpoint(
            path, allow_legacy=allow_legacy, base_cfg=base_cfg)
        name = os.path.splitext(os.path.basename(path))[0]
        cell_res, _ = _eval_actor(name, agent.act, agent.reset, cfg, episodes,
                                  checkpoint_path=path)
        results["agents"][name] = cell_res
        ckpt_refs = {name: checkpoint_ref(path)}
    else:
        cfg = base_cfg
        for name, make_actor in [("mlp_baseline", _baseline_actor),
                                 ("liquid", _liquid_actor)]:
            act, reset, _ = make_actor(cfg, allow_legacy)
            cell_res, _ = _eval_actor(name, act, reset, cfg, episodes)
            results["agents"][name] = cell_res
        ckpt_refs = {
            "liquid": checkpoint_ref(os.path.join(MODELS_DIR, "liquid_policy.pt")),
            "mlp_baseline": checkpoint_ref(
                os.path.join(MODELS_DIR, "ppo_baseline.zip"))}

    results["meta"] = gather_provenance(
        base_cfg, experiment_name="eval_dt_robustness",
        seeds={"seed_base": SEED_BASE, "episodes": episodes},
        extra={"ladder": [{"label": l, "substeps_min": a, "substeps_max": b,
                           "gap_prob": g} for l, a, b, g in LADDER],
               "checkpoints": ckpt_refs})
    out = write_results("dt_robustness", results)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--allow-legacy", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="evaluate a single policy checkpoint")
    ap.add_argument("--config", type=str, default=None,
                    help="optional base YAML (checkpoint meta still preferred)")
    ap.add_argument("--manifest", type=str, default=None,
                    help="factorial manifest JSON listing runs to evaluate")
    ap.add_argument("--all-factorial", action="store_true",
                    help="evaluate the newest models/factorial_manifest_*.json")
    a = ap.parse_args()
    main(a.episodes, a.allow_legacy, a.checkpoint, a.config,
         a.manifest, a.all_factorial)
