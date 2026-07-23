"""Phase 3: collect experience and train the JEPA-style world model.

Data comes from the agent's own behavior: the trained liquid policy with
exploration noise if available, otherwise random actions. The model is
verified by multi-step latent prediction error against a persistence
baseline (just assuming the world doesn't change).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import DATA_DIR, MODELS_DIR, ensure_dirs, load_config
from agent.hybrid_agent import HybridAgent
from agent.world_model import WorldModel
from environment.nav_env import OBS_DIM, NavEnv
from training.replay_buffer import ReplayBuffer


def collect(cfg: dict, episodes: int) -> ReplayBuffer:
    env = NavEnv(cfg, irregular_dt=True)   # train the model on irregular time
    agent = None
    policy_path = os.path.join(MODELS_DIR, "liquid_policy.pt")
    if os.path.exists(policy_path):
        agent = HybridAgent(cfg, mode="reactive")
        agent.load(policy_path)
        print("[collect] using liquid policy + exploration noise")
    else:
        print("[collect] no policy found, using random actions")

    buf = ReplayBuffer()
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        if agent:
            agent.reset()
        obs_list, act_list, dt_list = [obs], [], []
        done = False
        while not done:
            if agent and rng.random() > 0.3:
                action = agent.act(obs, env._last_dt)
                action = np.clip(action + rng.normal(0, 0.3, 2), -1, 1)
            else:
                action = rng.uniform(-1, 1, 2)
            obs, _, term, trunc, info = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            dt_list.append(info["dt"])
            done = term or trunc
        buf.add_episode(np.array(obs_list), np.array(act_list),
                        np.array(dt_list))
    env.close()
    print(f"[collect] {len(buf)} episodes, {buf.n_steps()} steps")
    return buf


def evaluate_multistep(wm: WorldModel, buf: ReplayBuffer, chunk: int,
                       steps: int = 4) -> dict:
    """Latent error rolling the predictor `steps` chunks ahead vs persistence."""
    rng = np.random.default_rng(1)
    errs, base = [], []
    with torch.no_grad():
        for _ in range(200):
            e = buf.episodes[rng.integers(len(buf.episodes))]
            T = len(e["actions"])
            if T <= chunk * steps:
                continue
            t = int(rng.integers(0, T - chunk * steps))
            z = wm.encode(torch.from_numpy(e["obs"][t:t + 1]))
            z0 = z.clone()
            for k in range(steps):
                lo = t + k * chunk
                a = torch.from_numpy(
                    e["actions"][lo:lo + chunk].mean(axis=0, keepdims=True))
                d = torch.tensor([[e["dts"][lo:lo + chunk].sum()]])
                z = wm.predict_next(z, a.float(), d)
            z_true = wm.target_encoder(
                torch.from_numpy(e["obs"][t + chunk * steps:t + chunk * steps + 1]))
            errs.append(float(torch.norm(z - z_true)))
            base.append(float(torch.norm(z0 - z_true)))
    return {"pred_err": float(np.mean(errs)),
            "persistence_err": float(np.mean(base)),
            "ratio": float(np.mean(errs) / (np.mean(base) + 1e-8))}


def train(config: dict | None = None):
    cfg = config or load_config()
    ensure_dirs()
    wcfg = cfg["world_model"]
    chunk = int(wcfg["chunk_steps"])

    buf_path = os.path.join(DATA_DIR, "experience.npz")
    if os.path.exists(buf_path):
        buf = ReplayBuffer.load(buf_path)
        print(f"[wm] loaded buffer: {len(buf)} episodes")
    else:
        buf = collect(cfg, int(wcfg["collect_episodes"]))
        buf.save(buf_path)

    # hold out 10% of episodes: the model is validated on trajectories it
    # never fit, not on its own training data
    n_hold = max(1, len(buf) // 10)
    heldout = ReplayBuffer()
    heldout.episodes = buf.episodes[:n_hold]
    train_buf = ReplayBuffer()
    train_buf.episodes = buf.episodes[n_hold:]
    buf = train_buf
    print(f"[wm] train {len(buf)} episodes, held out {n_hold}")

    wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                    hidden=int(wcfg["hidden_dim"]),
                    ema_momentum=float(wcfg["ema_momentum"]))
    optim = torch.optim.Adam(wm.parameters(), lr=float(wcfg["lr"]))
    rng = np.random.default_rng(0)
    steps = int(wcfg["train_steps"])
    for step in range(steps):
        o, a, d, o2 = buf.sample_chunks(int(wcfg["batch_size"]), chunk, rng)
        loss, parts = wm.loss(torch.from_numpy(o), torch.from_numpy(a),
                              torch.from_numpy(d), torch.from_numpy(o2))
        optim.zero_grad()
        loss.backward()
        optim.step()
        wm.update_target()
        if (step + 1) % 500 == 0:
            print(f"[wm] step {step + 1}/{steps} "
                  + " ".join(f"{k}={v:.4f}" for k, v in parts.items()))

    path = os.path.join(MODELS_DIR, "world_model.pt")
    torch.save(wm.state_dict(), path)
    metrics = evaluate_multistep(wm, heldout, chunk)
    print(f"[wm] saved {path}")
    print(f"[wm] 4-chunk latent error: {metrics['pred_err']:.4f} "
          f"(persistence baseline {metrics['persistence_err']:.4f}, "
          f"ratio {metrics['ratio']:.2f} — below 1.0 means the model "
          f"predicts better than assuming nothing changes)")
    return metrics


if __name__ == "__main__":
    train()
