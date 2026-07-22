"""Live PyBullet GUI demo of whatever brains are trained so far.

    python scripts/run_live.py                     # best available agent
    python scripts/run_live.py --agent random      # Phase 0 sanity check
    python scripts/run_live.py --agent baseline    # PPO MLP
    python scripts/run_live.py --agent liquid      # SNN + CfC reactive
    python scripts/run_live.py --agent hier        # full hierarchical stack
    python scripts/run_live.py --jitter            # irregular sensor timing
    python scripts/run_live.py --layout u_trap     # the planning test arena
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import MODELS_DIR, load_config
from environment.nav_env import OBS_DIM, NavEnv


def pick_agent(cfg, choice):
    have = {n: os.path.exists(os.path.join(MODELS_DIR, f))
            for n, f in [("hier", "hier_policy.pt"),
                         ("liquid", "liquid_policy.pt"),
                         ("baseline", "ppo_baseline.zip")]}
    if choice == "auto":
        choice = next((n for n in ("hier", "liquid", "baseline") if have[n]),
                      "random")
    print(f"[live] agent: {choice}")

    if choice == "random":
        rng = np.random.default_rng()
        return lambda obs, dt: rng.uniform(-1, 1, 2), lambda: None
    if choice == "baseline":
        from stable_baselines3 import PPO
        model = PPO.load(os.path.join(MODELS_DIR, "ppo_baseline.zip"))
        return (lambda obs, dt: model.predict(obs, deterministic=True)[0],
                lambda: None)

    from agent.hybrid_agent import HybridAgent
    from agent.world_model import WorldModel
    wm = None
    if choice == "hier":
        wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                        hidden=int(cfg["world_model"]["hidden_dim"]))
        wm.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "world_model.pt"), weights_only=True))
        wm.eval()
    agent = HybridAgent(cfg, mode="hierarchical" if choice == "hier"
                        else "reactive", world_model=wm)
    agent.load(os.path.join(
        MODELS_DIR, "hier_policy.pt" if choice == "hier" else "liquid_policy.pt"))
    return agent.act, agent.reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="auto",
                    choices=["auto", "random", "baseline", "liquid", "hier"])
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--layout", default="random", choices=["random", "u_trap"])
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    env = NavEnv(cfg, render_mode="human", irregular_dt=args.jitter,
                 layout=args.layout)
    act, reset = pick_agent(cfg, args.agent)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        reset()
        total, done, steps = 0.0, False, 0
        while not done:
            t0 = time.time()
            obs, r, term, trunc, info = env.step(act(obs, env._last_dt))
            total += r
            steps += 1
            done = term or trunc
            time.sleep(max(0.0, info["dt"] - (time.time() - t0)))  # real time
        print(f"[live] episode {ep + 1}: reward={total:.2f} steps={steps} "
              f"success={info['is_success']}")
    env.close()


if __name__ == "__main__":
    main()
