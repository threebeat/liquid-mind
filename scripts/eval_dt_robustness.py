"""Phase 2c: the real-time claim, tested.

Both agents were trained at a fixed 30 Hz control rate. Here we evaluate them
while the sensor stream's timing is randomized (each step simulates a random
number of physics substeps, 15-60 Hz). The MLP baseline sees dt only as one
more input feature; the liquid CfC policy integrates it into its continuous-
time dynamics. Prediction: the liquid agent degrades gracefully, the MLP
degrades more.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv


def run_liquid(cfg, irregular, episodes, seed0):
    env = NavEnv(cfg, irregular_dt=irregular)
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
    return float(np.mean(rewards)), successes / episodes


def run_baseline(cfg, irregular, episodes, seed0):
    from stable_baselines3 import PPO
    env = NavEnv(cfg, irregular_dt=irregular)
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
    return float(np.mean(rewards)), successes / episodes


def main(episodes: int = 20):
    cfg = load_config()
    ensure_dirs()
    results = {}
    for name, fn in [("mlp_baseline", run_baseline), ("liquid", run_liquid)]:
        r_fix, s_fix = fn(cfg, False, episodes, 555_000)
        r_jit, s_jit = fn(cfg, True, episodes, 555_000)
        drop = (r_fix - r_jit) / (abs(r_fix) + 1e-8) * 100
        results[name] = {
            "fixed_dt": {"reward": r_fix, "success": s_fix},
            "irregular_dt": {"reward": r_jit, "success": s_jit},
            "reward_drop_pct": drop}
        print(f"{name:14s} fixed: R={r_fix:7.2f} success={s_fix:.0%}   "
              f"irregular: R={r_jit:7.2f} success={s_jit:.0%}   "
              f"drop={drop:.1f}%")
    out = os.path.join(RESULTS_DIR, "dt_robustness.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    main()
