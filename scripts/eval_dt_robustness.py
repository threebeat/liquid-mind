"""Phase 2c: the real-time claim, tested — with the confounds controlled.

Both agents are trained at a fixed 30 Hz control rate and evaluated under
increasing timing jitter. Design notes (deliberate, to keep the comparison
fair):
  - jitter substep ranges are centered on the nominal 8 substeps, so the MEAN
    control rate is identical across conditions — only the variance changes;
  - episodes truncate on simulated time and the step penalty is charged per
    unit of simulated time, so all conditions face the same physical horizon
    and reward rate;
  - both agents see dt in the observation; the liquid agent additionally
    integrates it into its continuous-time dynamics (CfC timespans + LIF
    exp(-dt/tau) leak). That integration is the mechanism under test.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv

# (label, substeps_min, substeps_max) — all means equal the nominal 8
JITTER_LEVELS = [("fixed", None, None), ("mild", 6, 10), ("strong", 4, 12)]


def _env_for(cfg, smin, smax):
    if smin is None:
        return NavEnv(cfg, irregular_dt=False)
    c = copy.deepcopy(cfg)
    c["env"]["substeps_min"] = smin
    c["env"]["substeps_max"] = smax
    return NavEnv(c, irregular_dt=True)


def run_liquid(cfg, smin, smax, episodes, seed0):
    env = _env_for(cfg, smin, smax)
    agent = HybridAgent(cfg, mode="reactive")
    agent.load(os.path.join(MODELS_DIR, "liquid_policy.pt"))
    rewards, successes = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent.reset()
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, env._last_dt))
            total += r
            done = term or trunc
        rewards.append(total)
        successes += int(info["is_success"])
    env.close()
    return np.asarray(rewards), successes


def run_baseline(cfg, smin, smax, episodes, seed0):
    from stable_baselines3 import PPO
    env = _env_for(cfg, smin, smax)
    model = PPO.load(os.path.join(MODELS_DIR, "ppo_baseline.zip"))
    rewards, successes = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
        successes += int(info["is_success"])
    env.close()
    return np.asarray(rewards), successes


def main(episodes: int = 50):
    cfg = load_config()
    ensure_dirs()
    results = {}
    for name, fn in [("mlp_baseline", run_baseline), ("liquid", run_liquid)]:
        results[name] = {}
        r_fixed = None
        for label, smin, smax in JITTER_LEVELS:
            rw, n_succ = fn(cfg, smin, smax, episodes, 555_000)
            mean = float(rw.mean())
            ci95 = float(1.96 * rw.std(ddof=1) / np.sqrt(len(rw)))
            if label == "fixed":
                r_fixed = mean
            drop = (r_fixed - mean) / (abs(r_fixed) + 1e-8) * 100
            results[name][label] = {
                "reward_mean": mean, "reward_ci95": ci95,
                "reward_median": float(np.median(rw)),
                "successes": n_succ, "episodes": episodes,
                "drop_vs_fixed_pct": drop}
            print(f"{name:14s} {label:6s}: R={mean:7.2f} +/-{ci95:5.2f} "
                  f"(median {np.median(rw):6.2f})  "
                  f"success={n_succ}/{episodes}  drop={drop:.1f}%", flush=True)
    out = os.path.join(RESULTS_DIR, "dt_robustness.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    main()
