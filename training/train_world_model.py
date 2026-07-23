"""Phase 3: collect experience and train the JEPA-style world model.

Data comes from the agent's own behavior: the trained liquid policy with
exploration noise if available, otherwise random actions. Chunks are defined
by target PHYSICAL duration (world_model.chunk_seconds), so a chunk contains
a variable number of action events under irregular timing.

The model must pass a PLANNING GATE before hierarchical training/evaluation
will use it (Priority 7). The gate evaluates every grounded readout —
goal distance (m), goal bearing (rad), and all four directional obstacle
minima — at open-loop horizons of 0.5/1/2/4 s against:
  - a persistence baseline (assume nothing changes);
  - a differential-drive kinematic baseline (integrate wheel commands for
    goal distance/bearing; persistence for obstacle rays);
plus a false-safe rate (model predicts clearance, reality is dangerous) and
a left/right directional-confusion rate. The gate verdict is stored INSIDE
the checkpoint metadata; downstream consumers refuse a failed or ungated
model unless explicitly overridden.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import DATA_DIR, MODELS_DIR, ensure_dirs, load_config
from provenance import (SEMANTICS_VERSION, checkpoint_ref, gather_provenance,
                        save_checkpoint)
from agent.hybrid_agent import HybridAgent
from agent.world_model import WorldModel, _QUADRANTS
from environment.nav_env import (AXLE_TRACK, OBS_DIM, WHEEL_RADIUS, NavEnv)
from training.replay_buffer import ReplayBuffer


def wm_compat(cfg: dict) -> dict:
    """Compatibility block for world-model checkpoints."""
    w = cfg["world_model"]
    return {"semantics_version": SEMANTICS_VERSION,
            "obs_dim": OBS_DIM,
            "latent_dim": int(cfg["agent"]["latent_dim"]),
            "hidden_dim": int(w["hidden_dim"]),
            "chunk_seconds": float(w.get("chunk_seconds", 0.5))}


# ------------------------------------------------------------------ collect


def collect(cfg: dict, episodes: int) -> ReplayBuffer:
    env = NavEnv(cfg, irregular_dt=True)   # train the model on irregular time
    agent = None
    policy_path = os.path.join(MODELS_DIR, "liquid_policy.pt")
    if os.path.exists(policy_path):
        try:
            agent = HybridAgent(cfg, mode="reactive")
            agent.load(policy_path)
            print("[collect] using liquid policy + exploration noise")
        except (ValueError, RuntimeError) as e:
            agent = None
            print(f"[collect] policy checkpoint unusable ({e}); "
                  f"falling back to random actions")
    else:
        print("[collect] no policy found, using random actions")

    buf = ReplayBuffer()
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        obs, info = env.reset(seed=10_000 + ep)
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
    env.close()
    print(f"[collect] {len(buf)} episodes, {buf.n_steps()} steps")
    return buf


# ----------------------------------------------------------- grounded truth


def _grounded(obs_row: np.ndarray) -> dict:
    """Physical-unit grounded state extracted from a raw observation."""
    rays = obs_row[:16]
    quads = {name: float(rays[idx].min()) for name, idx in _QUADRANTS.items()}
    return {"goal_dist_m": float(obs_row[16]) * 10.0,
            "bearing": float(math.atan2(obs_row[17], obs_row[18])),
            "quads": quads}


def _readout_to_grounded(read: np.ndarray) -> dict:
    """Same structure from the model readout [gd, sin, cos, f, l, b, r]."""
    return {"goal_dist_m": float(read[0]) * 10.0,
            "bearing": float(math.atan2(read[1], read[2])),
            "quads": {"front": float(read[3]), "left": float(read[4]),
                      "back": float(read[5]), "right": float(read[6])}}


def _ang_err(a: float, b: float) -> float:
    return abs(float(np.angle(np.exp(1j * (a - b)))))


def _kinematic_rollout(g0: dict, actions: np.ndarray, dts: np.ndarray,
                       max_wheel_speed: float) -> dict:
    """Differential-drive constant-command baseline: integrate wheel
    commands to predict goal distance/bearing; rays use persistence."""
    d, b = g0["goal_dist_m"], g0["bearing"]
    gx, gy = d * math.cos(b), d * math.sin(b)      # goal in robot frame
    for a, dt in zip(actions, dts):
        vl = float(a[0]) * max_wheel_speed * WHEEL_RADIUS
        vr = float(a[1]) * max_wheel_speed * WHEEL_RADIUS
        v = 0.5 * (vl + vr)
        w = (vr - vl) / AXLE_TRACK
        gx -= v * dt                                # robot moves forward
        c, s = math.cos(-w * dt), math.sin(-w * dt)  # frame rotates
        gx, gy = c * gx - s * gy, s * gx + c * gy
    return {"goal_dist_m": math.hypot(gx, gy),
            "bearing": math.atan2(gy, gx),
            "quads": dict(g0["quads"])}             # persistence for rays


# ------------------------------------------------------------ planning gate

DANGER_TRUE = 0.08      # true min hit fraction below this = dangerous
SAFE_PRED = 0.12        # predicted min above this = model calls it safe
LR_MARGIN = 0.10        # only score L/R ordering when truth is decisive


def evaluate_gate(wm: WorldModel, buf: ReplayBuffer, chunk_seconds: float,
                  cfg: dict, horizons=(0.5, 1.0, 2.0, 4.0),
                  n_windows: int = 300, seed: int = 1) -> dict:
    """Open-loop rollout evaluation of every grounded readout, against
    persistence and kinematic baselines. Returns metrics + gate verdict."""
    rng = np.random.default_rng(seed)
    max_wheel = float(cfg["env"]["max_wheel_speed"])
    n_chunks_max = int(round(max(horizons) / chunk_seconds))
    h_chunks = {h: int(round(h / chunk_seconds)) for h in horizons}

    acc = {h: {"goal": {"model": [], "persist": [], "kin": []},
               "bear": {"model": [], "persist": [], "kin": []},
               "quad": {"model": [], "persist": []},
               "danger_true": [], "pred_safe": [], "persist_safe": [],
               "lr_total": 0, "lr_wrong": 0}
           for h in horizons}

    tried = 0
    done_windows = 0
    with torch.no_grad():
        while done_windows < n_windows and tried < n_windows * 20:
            tried += 1
            e = buf.episodes[rng.integers(len(buf.episodes))]
            T = len(e["actions"])
            if T < 4:
                continue
            t0 = int(rng.integers(0, T))
            # build consecutive duration-based chunks from t0
            bounds, idx = [], t0
            ok = True
            for _ in range(n_chunks_max):
                got = ReplayBuffer.steps_for_duration(e["dts"], idx,
                                                      chunk_seconds)
                if got is None:
                    ok = False
                    break
                k, actual = got
                bounds.append((idx, idx + k, actual))
                idx += k
            if not ok:
                continue
            done_windows += 1

            g0 = _grounded(e["obs"][t0])
            z = wm.encode(torch.from_numpy(e["obs"][t0:t0 + 1]))
            chunk_count = 0
            for lo, hi, actual in bounds:
                a = e["actions"][lo:hi]
                d = e["dts"][lo:hi]
                a_w = (a * d[:, None]).sum(axis=0) / d.sum()
                z = wm.predict_next(z, torch.from_numpy(a_w[None]).float(),
                                    torch.tensor([[float(actual)]]))
                chunk_count += 1
                for h, hc in h_chunks.items():
                    if hc != chunk_count:
                        continue
                    truth = _grounded(e["obs"][bounds[chunk_count - 1][1]])
                    pred = _readout_to_grounded(
                        wm.readout(z).squeeze(0).numpy())
                    kin = _kinematic_rollout(
                        g0, e["actions"][t0:hi], e["dts"][t0:hi], max_wheel)
                    a_ = acc[h]
                    a_["goal"]["model"].append(
                        abs(pred["goal_dist_m"] - truth["goal_dist_m"]))
                    a_["goal"]["persist"].append(
                        abs(g0["goal_dist_m"] - truth["goal_dist_m"]))
                    a_["goal"]["kin"].append(
                        abs(kin["goal_dist_m"] - truth["goal_dist_m"]))
                    a_["bear"]["model"].append(
                        _ang_err(pred["bearing"], truth["bearing"]))
                    a_["bear"]["persist"].append(
                        _ang_err(g0["bearing"], truth["bearing"]))
                    a_["bear"]["kin"].append(
                        _ang_err(kin["bearing"], truth["bearing"]))
                    qm = np.mean([abs(pred["quads"][q] - truth["quads"][q])
                                  for q in truth["quads"]])
                    qp = np.mean([abs(g0["quads"][q] - truth["quads"][q])
                                  for q in truth["quads"]])
                    a_["quad"]["model"].append(float(qm))
                    a_["quad"]["persist"].append(float(qp))
                    # false-safe bookkeeping
                    true_min = min(truth["quads"].values())
                    a_["danger_true"].append(true_min < DANGER_TRUE)
                    a_["pred_safe"].append(
                        min(pred["quads"].values()) >= SAFE_PRED)
                    a_["persist_safe"].append(
                        min(g0["quads"].values()) >= SAFE_PRED)
                    # left/right directional confusion
                    t_lr = truth["quads"]["left"] - truth["quads"]["right"]
                    if abs(t_lr) > LR_MARGIN:
                        p_lr = pred["quads"]["left"] - pred["quads"]["right"]
                        a_["lr_total"] += 1
                        a_["lr_wrong"] += int(np.sign(p_lr) != np.sign(t_lr))

    def _false_safe(danger, safe):
        danger = np.asarray(danger)
        safe = np.asarray(safe)
        n_danger = int(danger.sum())
        if n_danger == 0:
            return None
        return float((danger & safe).sum() / n_danger)

    metrics = {"n_windows": done_windows, "chunk_seconds": chunk_seconds,
               "horizons": {}}
    for h in horizons:
        a_ = acc[h]
        metrics["horizons"][str(h)] = {
            "goal_dist_mae_m": {k: float(np.mean(v)) if v else None
                                for k, v in a_["goal"].items()},
            "bearing_mae_rad": {k: float(np.mean(v)) if v else None
                                for k, v in a_["bear"].items()},
            "quadrant_mae": {k: float(np.mean(v)) if v else None
                             for k, v in a_["quad"].items()},
            "false_safe_rate": {
                "model": _false_safe(a_["danger_true"], a_["pred_safe"]),
                "persist": _false_safe(a_["danger_true"], a_["persist_safe"])},
            "lr_confusion_rate": (a_["lr_wrong"] / a_["lr_total"]
                                  if a_["lr_total"] else None),
            "lr_cases": a_["lr_total"],
        }

    def _m(h, group, who):
        return metrics["horizons"][str(h)][group][who]

    fs2 = metrics["horizons"]["2.0"]["false_safe_rate"]
    criteria = {
        "goal_beats_persistence_2s": _m(2.0, "goal_dist_mae_m", "model")
                                     < _m(2.0, "goal_dist_mae_m", "persist"),
        "goal_beats_persistence_4s": _m(4.0, "goal_dist_mae_m", "model")
                                     < _m(4.0, "goal_dist_mae_m", "persist"),
        "bearing_beats_persistence_2s": _m(2.0, "bearing_mae_rad", "model")
                                        < _m(2.0, "bearing_mae_rad", "persist"),
        "quad_beats_persistence_2s": _m(2.0, "quadrant_mae", "model")
                                     < _m(2.0, "quadrant_mae", "persist"),
        "false_safe_not_worse_2s": (fs2["model"] is None
                                    or fs2["persist"] is None
                                    or fs2["model"] <= fs2["persist"]),
        "open_loop_stable": (_m(4.0, "goal_dist_mae_m", "model")
                             <= 2.0 * _m(2.0, "goal_dist_mae_m", "model")
                             + 1e-9),
    }
    metrics["criteria"] = criteria
    metrics["passed"] = all(criteria.values())
    return metrics


# -------------------------------------------------------------------- train


def train(config: dict | None = None, force: bool = False,
          experiment_name: str = "world_model"):
    cfg = config or load_config()
    ensure_dirs()
    wcfg = cfg["world_model"]
    chunk_seconds = float(wcfg.get("chunk_seconds", 0.5))
    path = os.path.join(MODELS_DIR, "world_model.pt")
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{path} already exists; refusing to overwrite an existing "
            f"world model. Re-run with --force to replace it.")

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
        # chunks by target physical duration (variable event counts)
        o, a, d, o2 = buf.sample_chunks_by_duration(
            int(wcfg["batch_size"]), chunk_seconds, rng)
        loss, parts = wm.loss(torch.from_numpy(o), torch.from_numpy(a),
                              torch.from_numpy(d), torch.from_numpy(o2))
        optim.zero_grad()
        loss.backward()
        optim.step()
        wm.update_target()
        if (step + 1) % 500 == 0:
            print(f"[wm] step {step + 1}/{steps} "
                  + " ".join(f"{k}={v:.4f}" for k, v in parts.items()))

    wm.eval()
    gate = evaluate_gate(wm, heldout, chunk_seconds, cfg)
    meta = gather_provenance(
        cfg, experiment_name=experiment_name,
        extra={"gate": gate,
               "train_episodes": len(buf), "heldout_episodes": n_hold,
               "train_steps": steps,
               "buffer": checkpoint_ref(buf_path) if os.path.exists(buf_path)
               else None})
    save_checkpoint(path, wm.state_dict(), meta, wm_compat(cfg), force=force)
    print(f"[wm] saved {path}")
    _print_gate(gate)
    return gate


def _print_gate(gate: dict):
    print(f"[wm] planning gate over {gate['n_windows']} held-out windows "
          f"(chunk {gate['chunk_seconds']} s):")
    for h, m in gate["horizons"].items():
        g, b, q = m["goal_dist_mae_m"], m["bearing_mae_rad"], m["quadrant_mae"]
        fs = m["false_safe_rate"]
        print(f"[wm]  {h}s: goal {g['model']:.3f} m "
              f"(persist {g['persist']:.3f}, kin {g['kin']:.3f}) | "
              f"bearing {b['model']:.3f} rad (persist {b['persist']:.3f}) | "
              f"quad {q['model']:.3f} (persist {q['persist']:.3f}) | "
              f"false-safe {fs['model']} vs {fs['persist']} | "
              f"L/R confusion {m['lr_confusion_rate']}")
    verdict = "PASSED" if gate["passed"] else "FAILED"
    print(f"[wm] gate {verdict}: " + ", ".join(
        f"{k}={'ok' if v else 'FAIL'}" for k, v in gate["criteria"].items()))
    if not gate["passed"]:
        print("[wm] hierarchical training/evaluation will refuse this model "
              "unless --override-wm-gate is given.")


# ------------------------------------------------------------ gate checking


def load_world_model(cfg: dict, path: str | None = None,
                     override_gate: bool = False, allow_legacy: bool = False):
    """Load a world model, enforcing the planning gate (Priority 7).
    Returns (wm, meta). Refuses failed/ungated models unless overridden."""
    from provenance import load_checkpoint  # local import: avoids cycles
    path = path or os.path.join(MODELS_DIR, "world_model.pt")
    state, meta = load_checkpoint(path, expected_compat=wm_compat(cfg),
                                  allow_legacy=allow_legacy)
    gate = meta.get("gate")
    if gate is None or not gate.get("passed", False):
        status = "has no recorded planning gate" if gate is None \
            else "FAILED its planning gate"
        if not override_gate:
            raise ValueError(
                f"world model {path} {status}. Planning on an unvalidated "
                f"model is disabled; retrain the model or pass "
                f"--override-wm-gate for explicit diagnostics only.")
        print(f"[wm] WARNING: using a world model that {status} "
              f"(--override-wm-gate).")
    wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                    hidden=int(cfg["world_model"]["hidden_dim"]))
    wm.load_state_dict(state)
    wm.eval()
    return wm, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing world_model.pt")
    a = ap.parse_args()
    train(force=a.force)
