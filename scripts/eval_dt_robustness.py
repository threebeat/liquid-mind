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
  - --manifest / --all-factorial to evaluate every cell in a factorial
    manifest with matching reconstructed configurations, identical seed
    banks, within-policy degradation, and paired cross-cell effects.

Every episode record is retained; aggregates use bootstrap CIs and Wilson
intervals, and each jitter level is compared to the fixed condition with a
paired bootstrap on identical seeds (Priority 5).
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
from scripts.eval_common import compare, run_episode, summarize
from scripts.factorial_io import (discover_latest_manifest, load_agent_for_checkpoint,
                                  load_manifest)

# (label, substeps_min, substeps_max, gap_prob) — jitter means equal the
# nominal 8 substeps; "gaps" adds rare 250-1000 ms observation dropouts
LADDER = [("fixed", None, None, 0.0),
          ("mild", 6, 10, 0.0),
          ("strong", 4, 12, 0.0),
          ("wide", 2, 14, 0.0),
          ("gaps", 6, 10, 0.02)]

SEED_BASE = 555_000


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


def _eval_factorial_cells(cells, episodes, allow_legacy, base_cfg):
    """Evaluate every factorial cell; return agent results + cross-cell pairs."""
    results = {"agents": {}, "factorial_comparisons": {}}
    fixed_by_cell = {}
    for cell in cells:
        ckpt = cell["checkpoint"]
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        if not os.path.exists(path):
            print(f"[eval-dt] SKIP {cell.get('name', ckpt)}: missing {path}")
            continue
        agent, cfg = load_agent_for_checkpoint(
            path, allow_legacy=allow_legacy, base_cfg=base_cfg, cell=cell)
        name = cell.get("name") or cell.get("experiment") or os.path.basename(path)
        print(f"\n[eval-dt] === {name} ({path}) ===", flush=True)
        cell_res, per_level = _eval_actor(
            name, agent.act, agent.reset, cfg, episodes,
            checkpoint_path=path)
        cell_res["variant_factors"] = cell.get("variant_factors")
        cell_res["training_seed"] = cell.get("training_seed")
        results["agents"][name] = cell_res
        fixed_by_cell[name] = per_level["fixed"]

    # Paired cross-cell comparisons on the fixed ladder level (identical seeds)
    names = sorted(fixed_by_cell)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}_vs_{b}"
            c = compare(fixed_by_cell[a], fixed_by_cell[b], a, b)
            results["factorial_comparisons"][key] = c
            d = c["reward_diff"]
            print(f"[factorial] {a} vs {b} (fixed): "
                  f"dR={d['mean_diff']:+7.2f} [{d['lo']:.2f},{d['hi']:.2f}] "
                  f"{'(excludes 0)' if d['excludes_zero'] else '(includes 0)'}")
    return results


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
        agent, cfg = load_agent_for_checkpoint(
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
                    help="factorial manifest JSON listing cells to evaluate")
    ap.add_argument("--all-factorial", action="store_true",
                    help="evaluate the newest models/factorial_manifest_*.json")
    a = ap.parse_args()
    main(a.episodes, a.allow_legacy, a.checkpoint, a.config,
         a.manifest, a.all_factorial)
