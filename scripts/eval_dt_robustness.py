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

from common import MODELS_DIR, ensure_dirs, load_config
from provenance import checkpoint_ref, gather_provenance, write_results
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv
from scripts.eval_common import compare, run_episode, summarize

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


def _liquid_actor(cfg, allow_legacy):
    agent = HybridAgent(cfg, mode="reactive")
    agent.load(os.path.join(MODELS_DIR, "liquid_policy.pt"),
               allow_legacy=allow_legacy)
    return agent.act, agent.reset


def _baseline_actor(cfg, allow_legacy):
    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(MODELS_DIR, "ppo_baseline.zip"))
    return (lambda obs, dt: model.predict(obs, deterministic=True)[0],
            lambda: None)


def main(episodes: int = 50, allow_legacy: bool = False):
    cfg = load_config()
    ensure_dirs()
    results = {"agents": {}}
    for name, make_actor in [("mlp_baseline", _baseline_actor),
                             ("liquid", _liquid_actor)]:
        act, reset = make_actor(cfg, allow_legacy)
        results["agents"][name] = {}
        per_level_records = {}
        for label, smin, smax, gap in LADDER:
            env = _env_for(cfg, smin, smax, gap)
            recs = [run_episode(env, act, reset, SEED_BASE + ep)
                    for ep in range(episodes)]
            env.close()
            per_level_records[label] = recs
            s = summarize(recs)
            results["agents"][name][label] = {"summary": s, "episodes": recs}
            print(f"{name:14s} {label:6s}: "
                  f"R={s['reward_mean']['point']:7.2f} "
                  f"[{s['reward_mean']['lo']:.2f},{s['reward_mean']['hi']:.2f}]"
                  f"  (median {s['reward_median']['point']:.2f})  "
                  f"success={s['success']['k']}/{s['success']['n']}",
                  flush=True)
        # paired comparison of each disturbed level against fixed timing
        for label in per_level_records:
            if label == "fixed":
                continue
            c = compare(per_level_records[label], per_level_records["fixed"],
                        label, "fixed")
            results["agents"][name][label]["vs_fixed"] = c
            d = c["reward_diff"]
            print(f"{name:14s} {label:6s} vs fixed: "
                  f"dR={d['mean_diff']:+7.2f} [{d['lo']:.2f},{d['hi']:.2f}] "
                  f"{'(excludes 0)' if d['excludes_zero'] else '(includes 0)'}")

    results["meta"] = gather_provenance(
        cfg, experiment_name="eval_dt_robustness",
        seeds={"seed_base": SEED_BASE, "episodes": episodes},
        extra={"ladder": [{"label": l, "substeps_min": a, "substeps_max": b,
                           "gap_prob": g} for l, a, b, g in LADDER],
               "checkpoints": {
                   "liquid": checkpoint_ref(
                       os.path.join(MODELS_DIR, "liquid_policy.pt")),
                   "mlp_baseline": checkpoint_ref(
                       os.path.join(MODELS_DIR, "ppo_baseline.zip"))}})
    out = write_results("dt_robustness", results)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--allow-legacy", action="store_true")
    a = ap.parse_args()
    main(a.episodes, a.allow_legacy)
