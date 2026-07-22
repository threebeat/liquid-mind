"""Phase 4 demo: reactive vs hierarchical agent on the U-trap layout.

The U-shaped wall sits between robot and goal with its opening facing the
robot: greedy goal-seeking drives straight into the pocket. The hierarchical
agent plans in latent space through the JEPA world model and should detour.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config
from agent.hybrid_agent import HybridAgent
from agent.world_model import WorldModel
from environment.nav_env import OBS_DIM, NavEnv


def evaluate(cfg, mode: str, policy_file: str, episodes: int, layout: str):
    wm = None
    if mode == "hierarchical":
        wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                        hidden=int(cfg["world_model"]["hidden_dim"]))
        wm.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "world_model.pt"), weights_only=True))
        wm.eval()
    env = NavEnv(cfg, layout=layout)
    agent = HybridAgent(cfg, mode=mode, world_model=wm)
    agent.load(os.path.join(MODELS_DIR, policy_file))
    rewards, successes, final_dists = [], 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=777_000 + ep)
        agent.reset()
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, env._last_dt))
            total += r
            done = term or trunc
        rewards.append(total)
        successes += int(info["is_success"])
        final_dists.append(info["goal_dist"])
    env.close()
    return {"reward": float(np.mean(rewards)), "success": successes / episodes,
            "final_goal_dist": float(np.mean(final_dists))}


def main(episodes: int = 10, layout: str = "u_trap"):
    cfg = load_config()
    ensure_dirs()
    results = {}
    for name, mode, pf in [("reactive", "reactive", "liquid_policy.pt"),
                           ("hierarchical", "hierarchical", "hier_policy.pt")]:
        res = evaluate(cfg, mode, pf, episodes, layout)
        results[name] = res
        print(f"{name:13s} on {layout}: reward={res['reward']:7.2f}  "
              f"success={res['success']:.0%}  "
              f"final_dist={res['final_goal_dist']:.2f} m")
    out = os.path.join(RESULTS_DIR, f"hierarchy_{layout}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    main()
