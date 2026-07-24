"""Episode collection for Phase 1 (fingerprinted buffers, disjoint seeds).

Reuses NavEnv (irregular dt) + ReplayBuffer with the same behavior policy as
training/train_world_model.collect (liquid policy + exploration noise when
available, random actions otherwise). Every stage/split gets its RESERVED
disjoint env-seed range from phase1.SEED_RANGES; buffers are saved under
fingerprinted names with full provenance metadata.

Usage:
    .\\env\\python.exe -m phase1.collect_data --stage pilot
    .\\env\\python.exe -m phase1.collect_data --stage confirmatory
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import DATA_DIR, MODELS_DIR, ensure_dirs, load_config
from environment.nav_env import NavEnv
from phase1 import EXPERIMENT, SEED_RANGES, ensure_p1_dirs
from provenance import file_checksum
from training.replay_buffer import ReplayBuffer


def collect_range(cfg: dict, seed_lo: int, seed_hi: int,
                  noise_rng_seed: int, log=print) -> ReplayBuffer:
    """One episode per env seed in [seed_lo, seed_hi], inclusive."""
    env = NavEnv(cfg, irregular_dt=True)
    agent = None
    policy_path = os.path.join(MODELS_DIR, "liquid_policy.pt")
    if os.path.exists(policy_path):
        try:
            from agent.hybrid_agent import HybridAgent
            agent = HybridAgent(cfg, mode="reactive")
            agent.load(policy_path)
            log("[p1-collect] using liquid policy + exploration noise")
        except (ValueError, RuntimeError) as e:
            agent = None
            log(f"[p1-collect] policy unusable ({e}); random actions")
    else:
        log("[p1-collect] no policy found, using random actions")

    buf = ReplayBuffer()
    rng = np.random.default_rng(noise_rng_seed)
    for ep_seed in range(seed_lo, seed_hi + 1):
        obs, info = env.reset(seed=ep_seed)
        if agent:
            agent.reset()
        obs_list, act_list, dt_list = [obs], [], []
        t_cap = [info.get("sim_time", 0.0)]
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
            t_cap.append(info["sim_time"])
            done = term or trunc
        buf.add_episode(np.array(obs_list), np.array(act_list),
                        np.array(dt_list), np.array(t_cap))
        if (len(buf) % 25) == 0:
            log(f"[p1-collect] {len(buf)}/{seed_hi - seed_lo + 1} episodes")
    env.close()
    return buf


def _policy_usable(cfg: dict) -> bool:
    """The legacy liquid_policy.pt is a bare pre-provenance state dict;
    HybridAgent.load refuses it, so collection falls back to random actions.
    Resolve that up front so the buffer meta records the TRUE behavior."""
    policy_path = os.path.join(MODELS_DIR, "liquid_policy.pt")
    if not os.path.exists(policy_path):
        return False
    try:
        from agent.hybrid_agent import HybridAgent
        agent = HybridAgent(cfg, mode="reactive")
        agent.load(policy_path)
        return True
    except (ValueError, RuntimeError):
        return False


def buffer_path_for(cfg: dict, stage: str, split: str) -> tuple[str, dict]:
    lo, hi = SEED_RANGES[stage][split]
    policy_path = os.path.join(MODELS_DIR, "liquid_policy.pt")
    usable = _policy_usable(cfg)
    meta = ReplayBuffer.build_meta(
        cfg, seed_range=(lo, hi),
        policy_checkpoint=policy_path if usable else None,
        extra={"experiment": EXPERIMENT, "phase1_stage": stage,
               "phase1_split": split,
               "behavior_policy": "liquid+noise" if usable else "random"})
    # fingerprint over STABLE fields only (git commit/dirty churns across
    # commits and would silently re-collect identical data under new names)
    fp_meta = {k: v for k, v in meta.items() if k != "git"}
    return ReplayBuffer.fingerprint_name(fp_meta, DATA_DIR), meta


def collect_stage(stage: str, cfg: dict | None = None, log=print) -> dict:
    """Collect every split of a stage (skipping buffers that already exist).
    Returns {split: {"path", "sha256", "episodes", "seconds"}}."""
    cfg = cfg or load_config()
    ensure_dirs()
    ensure_p1_dirs()
    out = {}
    for split, (lo, hi) in SEED_RANGES[stage].items():
        path, meta = buffer_path_for(cfg, stage, split)
        if os.path.exists(path):
            buf = ReplayBuffer.load(path)
            log(f"[p1-collect] {stage}/{split}: exists {path} "
                f"({len(buf)} episodes)")
            out[split] = {"path": path, "sha256": file_checksum(path),
                          "episodes": len(buf), "seconds": 0.0,
                          "seed_range": [lo, hi], "reused": True}
            continue
        t0 = time.perf_counter()
        # noise rng seeded from the range start: reproducible, disjoint
        buf = collect_range(cfg, lo, hi, noise_rng_seed=lo, log=log)
        buf.save(path, meta=meta)
        dt = time.perf_counter() - t0
        log(f"[p1-collect] {stage}/{split}: saved {path} "
            f"({len(buf)} episodes, {buf.n_steps()} steps, {dt:.0f}s)")
        out[split] = {"path": path, "sha256": file_checksum(path),
                      "episodes": len(buf), "seconds": dt,
                      "seed_range": [lo, hi], "reused": False}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pilot", "confirmatory"])
    args = ap.parse_args()
    info = collect_stage(args.stage)
    for split, d in info.items():
        print(f"{split}: {d['path']} ({d['episodes']} eps)")
