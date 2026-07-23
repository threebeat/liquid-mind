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
  - an exact differential-drive kinematic baseline (arc integration for
    goal distance/bearing; persistence for obstacle rays);
plus false-safe rate (undefined => incomplete, never passed), left/right
confusion limits, episode-clustered effect sizes, and absolute usefulness
bounds. Status is passed / failed / incomplete; downstream consumers refuse
failed or incomplete models unless explicitly overridden.
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
    """Exact differential-drive arc integration for goal distance/bearing.

    For nonzero yaw rate uses the closed-form constant-(v,ω) arc; for ω≈0
    falls back to pure translation. Obstacle rays use persistence.
    """
    d, b = g0["goal_dist_m"], g0["bearing"]
    gx, gy = d * math.cos(b), d * math.sin(b)      # goal in robot frame
    for a, dt in zip(actions, dts):
        dt = float(dt)
        if dt <= 0:
            continue
        vl = float(a[0]) * max_wheel_speed * WHEEL_RADIUS
        vr = float(a[1]) * max_wheel_speed * WHEEL_RADIUS
        v = 0.5 * (vl + vr)
        w = (vr - vl) / AXLE_TRACK
        if abs(w) < 1e-9:
            dx, dy, theta = v * dt, 0.0, 0.0
        else:
            theta = w * dt
            dx = (v / w) * math.sin(theta)
            dy = (v / w) * (1.0 - math.cos(theta))
        # Express the fixed goal in the post-motion body frame
        rx, ry = gx - dx, gy - dy
        c, s = math.cos(theta), math.sin(theta)
        gx, gy = c * rx + s * ry, -s * rx + c * ry
    return {"goal_dist_m": math.hypot(gx, gy),
            "bearing": math.atan2(gy, gx),
            "quads": dict(g0["quads"])}             # persistence for rays


# ------------------------------------------------------------ planning gate

DANGER_TRUE = 0.08      # true min hit fraction below this = dangerous
SAFE_PRED = 0.12        # predicted min above this = model calls it safe
LR_MARGIN = 0.10        # only score L/R ordering when truth is decisive
MIN_DANGER_CASES = 20   # undefined safety below this -> incomplete, not pass
MIN_LR_CASES = 20
MAX_LR_CONFUSION = 0.40
# Planner-relevant effect sizes (goal_radius default 0.4 m)
MIN_GOAL_IMPROVE_M = 0.05          # absolute MAE improvement vs persistence @2s
MIN_GOAL_IMPROVE_REL = 0.10        # or 10% relative
MAX_ABS_GOAL_MAE_2S = 0.40         # useful absolute error <= goal_radius
KIN_EPS_M = 0.02                   # model must beat kin or be within ε while
                                   # beating persistence
MIN_BEAR_IMPROVE_RAD = 0.05


def _wilson_ci(k: int, n: int, z: float = 1.96) -> dict:
    if n <= 0:
        return {"point": None, "lo": None, "hi": None, "k": 0, "n": 0}
    from scripts.eval_common import wilson_interval
    return wilson_interval(k, n, z=z)


def _clustered_mean_diff(model_errs, base_errs, episode_ids, n_boot=2000,
                         seed=0) -> dict:
    """Episode-clustered bootstrap CI on mean(base - model) improvement."""
    model_errs = np.asarray(model_errs, dtype=np.float64)
    base_errs = np.asarray(base_errs, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    if len(model_errs) == 0:
        return {"mean_diff": None, "lo": None, "hi": None,
                "excludes_zero": False, "n": 0}
    # group by episode
    groups = {}
    for i, ep in enumerate(episode_ids):
        groups.setdefault(int(ep), []).append(i)
    ep_keys = sorted(groups)
    diffs = []
    for ep in ep_keys:
        idx = groups[ep]
        diffs.append(float(np.mean(base_errs[idx] - model_errs[idx])))
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(float(sample.mean()))
    boots = np.asarray(boots)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"mean_diff": float(diffs.mean()), "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "n_episodes": len(diffs),
            "n_windows": int(len(model_errs))}


def _ci_positive(es: dict | None) -> bool:
    """Clustered-bootstrap evidence of a positive improvement: the lower
    95% bound must exceed zero."""
    return bool(es and es.get("lo") is not None and es["lo"] > 0)


def evaluate_gate(wm: WorldModel, buf: ReplayBuffer, chunk_seconds: float,
                  cfg: dict, horizons=(0.5, 1.0, 2.0, 4.0),
                  n_windows: int = 300, seed: int = 1) -> dict:
    """Open-loop rollout evaluation of every grounded readout, against
    persistence and kinematic baselines. Returns metrics + gate verdict
    with status in {passed, failed, incomplete}.

    Improvement criteria require the POINT ESTIMATE to meet the minimum
    useful effect AND the clustered (episode-level) bootstrap lower bound
    to exceed zero — i.e. evidence of positive improvement, with the point
    estimate meeting the practical-effect threshold. This is deliberately
    NOT "95% confidence that the entire minimum useful effect is achieved".
    """
    rng = np.random.default_rng(seed)
    max_wheel = float(cfg["env"]["max_wheel_speed"])
    goal_radius = float(cfg["env"].get("goal_radius", 0.4))
    max_abs_goal = min(MAX_ABS_GOAL_MAE_2S, goal_radius)
    n_chunks_max = int(round(max(horizons) / chunk_seconds))
    h_chunks = {h: int(round(h / chunk_seconds)) for h in horizons}

    acc = {h: {"goal": {"model": [], "persist": [], "kin": []},
               "bear": {"model": [], "persist": [], "kin": []},
               "quad": {"model": [], "persist": []},
               "danger_true": [], "pred_safe": [], "persist_safe": [],
               "ep_ids": [],
               "lr_total": 0, "lr_wrong": 0}
           for h in horizons}

    tried = 0
    done_windows = 0
    with torch.no_grad():
        while done_windows < n_windows and tried < n_windows * 20:
            tried += 1
            ep_i = int(rng.integers(len(buf.episodes)))
            e = buf.episodes[ep_i]
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
                    a_["ep_ids"].append(ep_i)
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

    def _false_safe_stats(danger, safe):
        danger = np.asarray(danger, dtype=bool)
        safe = np.asarray(safe, dtype=bool)
        n_danger = int(danger.sum())
        if n_danger == 0:
            return {"rate": None, "n_danger": 0, "n_false_safe": 0,
                    "wilson": _wilson_ci(0, 0)}
        n_fs = int((danger & safe).sum())
        return {"rate": float(n_fs / n_danger), "n_danger": n_danger,
                "n_false_safe": n_fs, "wilson": _wilson_ci(n_fs, n_danger)}

    metrics = {"n_windows": done_windows, "chunk_seconds": chunk_seconds,
               "horizons": {}, "effect_sizes": {}}
    for h in horizons:
        a_ = acc[h]
        fs_m = _false_safe_stats(a_["danger_true"], a_["pred_safe"])
        fs_p = _false_safe_stats(a_["danger_true"], a_["persist_safe"])
        metrics["horizons"][str(h)] = {
            "goal_dist_mae_m": {k: float(np.mean(v)) if v else None
                                for k, v in a_["goal"].items()},
            "bearing_mae_rad": {k: float(np.mean(v)) if v else None
                                for k, v in a_["bear"].items()},
            "quadrant_mae": {k: float(np.mean(v)) if v else None
                             for k, v in a_["quad"].items()},
            "false_safe": {"model": fs_m, "persist": fs_p},
            # keep legacy key for printers/tests
            "false_safe_rate": {"model": fs_m["rate"],
                                "persist": fs_p["rate"]},
            "lr_confusion_rate": (a_["lr_wrong"] / a_["lr_total"]
                                  if a_["lr_total"] else None),
            "lr_cases": a_["lr_total"],
            "n_danger": fs_m["n_danger"],
        }
        metrics["effect_sizes"][str(h)] = {
            "goal_vs_persist": _clustered_mean_diff(
                a_["goal"]["model"], a_["goal"]["persist"], a_["ep_ids"],
                seed=seed + int(h * 10)),
            "goal_vs_kin": _clustered_mean_diff(
                a_["goal"]["model"], a_["goal"]["kin"], a_["ep_ids"],
                seed=seed + 100 + int(h * 10)),
            "bear_vs_persist": _clustered_mean_diff(
                a_["bear"]["model"], a_["bear"]["persist"], a_["ep_ids"],
                seed=seed + 200 + int(h * 10)),
            "bear_vs_kin": _clustered_mean_diff(
                a_["bear"]["model"], a_["bear"]["kin"], a_["ep_ids"],
                seed=seed + 300 + int(h * 10)),
            "quad_vs_persist": _clustered_mean_diff(
                a_["quad"]["model"], a_["quad"]["persist"], a_["ep_ids"],
                seed=seed + 400 + int(h * 10)),
        }

    def _m(h, group, who):
        return metrics["horizons"][str(h)][group][who]

    h2 = metrics["horizons"]["2.0"]
    es2 = metrics["effect_sizes"]["2.0"]
    fs2 = h2["false_safe"]
    n_danger = fs2["model"]["n_danger"]
    incomplete_reasons = []

    # --- safety: never pass when undefined ---
    if n_danger < MIN_DANGER_CASES:
        incomplete_reasons.append(
            f"n_danger={n_danger} < MIN_DANGER_CASES={MIN_DANGER_CASES}")
        false_safe_ok = None
    else:
        # model false-safe not worse than persistence (point estimate) and
        # upper Wilson bound not catastrophic
        m_rate, p_rate = fs2["model"]["rate"], fs2["persist"]["rate"]
        false_safe_ok = bool(m_rate is not None and p_rate is not None
                             and m_rate <= p_rate + 1e-9
                             and fs2["model"]["wilson"]["hi"] is not None
                             and fs2["model"]["wilson"]["hi"] <= 0.5)

    lr_rate, lr_n = h2["lr_confusion_rate"], h2["lr_cases"]
    if lr_n < MIN_LR_CASES:
        incomplete_reasons.append(
            f"lr_cases={lr_n} < MIN_LR_CASES={MIN_LR_CASES}")
        lr_ok = None
    else:
        lr_ok = bool(lr_rate is not None and lr_rate <= MAX_LR_CONFUSION)

    g_model = _m(2.0, "goal_dist_mae_m", "model")
    g_persist = _m(2.0, "goal_dist_mae_m", "persist")
    g_kin = _m(2.0, "goal_dist_mae_m", "kin")
    if g_model is not None and g_persist is not None:
        improve = g_persist - g_model
    else:
        improve = None
    rel_ok = (improve is not None and g_persist > 1e-9
              and improve / g_persist >= MIN_GOAL_IMPROVE_REL)
    abs_ok = improve is not None and improve >= MIN_GOAL_IMPROVE_M
    goal_persist_ok = bool(
        g_model is not None and g_persist is not None
        and (abs_ok or rel_ok)
        and es2["goal_vs_persist"]["mean_diff"] is not None
        and es2["goal_vs_persist"]["mean_diff"] > 0
        and _ci_positive(es2["goal_vs_persist"])
        and g_model <= max_abs_goal)

    # Beat kinematic, or within ε of kin while clearly beating persistence
    kin_ok = bool(
        g_model is not None and g_kin is not None
        and (g_model < g_kin - 1e-9
             or (g_model <= g_kin + KIN_EPS_M and goal_persist_ok)))

    b_model = _m(2.0, "bearing_mae_rad", "model")
    b_persist = _m(2.0, "bearing_mae_rad", "persist")
    b_kin = _m(2.0, "bearing_mae_rad", "kin")
    bear_persist_ok = bool(
        b_model is not None and b_persist is not None
        and (b_persist - b_model) >= MIN_BEAR_IMPROVE_RAD
        and es2["bear_vs_persist"]["mean_diff"] is not None
        and es2["bear_vs_persist"]["mean_diff"] > 0
        and _ci_positive(es2["bear_vs_persist"]))
    bear_kin_ok = bool(
        b_model is not None and b_kin is not None
        and (b_model < b_kin - 1e-9 or b_model <= b_kin + 0.05))

    q_ok = bool(
        _m(2.0, "quadrant_mae", "model") is not None
        and _m(2.0, "quadrant_mae", "persist") is not None
        and _m(2.0, "quadrant_mae", "model")
        < _m(2.0, "quadrant_mae", "persist")
        and es2["quad_vs_persist"]["mean_diff"] is not None
        and es2["quad_vs_persist"]["mean_diff"] > 0
        and _ci_positive(es2["quad_vs_persist"]))

    es4 = metrics["effect_sizes"].get("4.0", {})
    criteria = {
        "goal_beats_persistence_2s": goal_persist_ok,
        "goal_beats_persistence_4s": bool(
            _m(4.0, "goal_dist_mae_m", "model") is not None
            and _m(4.0, "goal_dist_mae_m", "persist") is not None
            and _m(4.0, "goal_dist_mae_m", "model")
            < _m(4.0, "goal_dist_mae_m", "persist")
            and _ci_positive(es4.get("goal_vs_persist"))),
        "goal_beats_or_matches_kinematic_2s": kin_ok,
        "bearing_beats_persistence_2s": bear_persist_ok,
        "bearing_beats_or_matches_kinematic_2s": bear_kin_ok,
        "quad_beats_persistence_2s": q_ok,
        "false_safe_not_worse_2s": false_safe_ok,
        "lr_confusion_ok_2s": lr_ok,
        "open_loop_stable": (
            _m(4.0, "goal_dist_mae_m", "model") is not None
            and _m(2.0, "goal_dist_mae_m", "model") is not None
            and _m(4.0, "goal_dist_mae_m", "model")
            <= 2.0 * _m(2.0, "goal_dist_mae_m", "model") + 1e-9),
        "abs_goal_mae_useful_2s": bool(
            g_model is not None and g_model <= max_abs_goal),
    }
    # Any None criterion => incomplete (never passed)
    if any(v is None for v in criteria.values()) or incomplete_reasons:
        status = "incomplete"
        passed = False
    elif all(bool(v) for v in criteria.values()):
        status = "passed"
        passed = True
    else:
        status = "failed"
        passed = False

    metrics["criteria"] = criteria
    metrics["incomplete_reasons"] = incomplete_reasons
    metrics["status"] = status
    metrics["passed"] = passed
    return metrics


# -------------------------------------------------------------------- train


def train(config: dict | None = None, force: bool = False,
          experiment_name: str = "world_model",
          allow_legacy_buffer: bool = False):
    cfg = config or load_config()
    ensure_dirs()
    wcfg = cfg["world_model"]
    chunk_seconds = float(wcfg.get("chunk_seconds", 0.5))
    path = os.path.join(MODELS_DIR, "world_model.pt")
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{path} already exists; refusing to overwrite an existing "
            f"world model. Re-run with --force to replace it.")

    policy_path = os.path.join(MODELS_DIR, "liquid_policy.pt")
    policy_ref = (policy_path if os.path.exists(policy_path) else None)
    buf_meta = ReplayBuffer.build_meta(
        cfg, seed_range=(10_000, 10_000 + int(wcfg["collect_episodes"]) - 1),
        policy_checkpoint=policy_ref)
    buf_path = ReplayBuffer.fingerprint_name(buf_meta, DATA_DIR)
    legacy_path = os.path.join(DATA_DIR, "experience.npz")
    buffer_override = False

    if os.path.exists(buf_path):
        buf = ReplayBuffer.load(buf_path,
                                allow_legacy_buffer=allow_legacy_buffer)
        print(f"[wm] loaded buffer: {buf_path} ({len(buf)} episodes)")
    elif os.path.exists(legacy_path):
        # Old global name — only with explicit override
        if not allow_legacy_buffer:
            raise ValueError(
                f"Found legacy {legacy_path} without provenance metadata. "
                f"Re-collect (delete it) or pass --allow-legacy-buffer.")
        buf = ReplayBuffer.load(legacy_path, allow_legacy_buffer=True)
        buffer_override = True
        print(f"[wm] loaded LEGACY buffer {legacy_path} "
              f"(--allow-legacy-buffer); override will be recorded")
    else:
        buf = collect(cfg, int(wcfg["collect_episodes"]))
        buf.save(buf_path, meta=buf_meta)
        print(f"[wm] collected and saved {buf_path}")

    # hold out 10% of episodes: the model is validated on trajectories it
    # never fit, not on its own training data
    n_hold = max(1, len(buf) // 10)
    heldout = ReplayBuffer(meta=buf.meta)
    heldout.episodes = buf.episodes[:n_hold]
    train_buf = ReplayBuffer(meta=buf.meta)
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
    used_buf = buf_path if os.path.exists(buf_path) else legacy_path
    meta = gather_provenance(
        cfg, experiment_name=experiment_name,
        extra={"gate": gate,
               "train_episodes": len(buf), "heldout_episodes": n_hold,
               "train_steps": steps,
               "buffer": checkpoint_ref(used_buf)
               if os.path.exists(used_buf) else None,
               "allow_legacy_buffer": bool(buffer_override
                                           or allow_legacy_buffer),
               "buffer_meta": getattr(heldout, "meta", None)})
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
              f"false-safe {fs['model']} vs {fs['persist']} "
              f"(n_danger={m.get('n_danger')}) | "
              f"L/R confusion {m['lr_confusion_rate']}")
    status = gate.get("status", "passed" if gate.get("passed") else "failed")
    print(f"[wm] gate status={status.upper()} (passed={gate['passed']}): "
          + ", ".join(f"{k}={'ok' if v else ('incomplete' if v is None else 'FAIL')}"
                      for k, v in gate["criteria"].items()))
    if gate.get("incomplete_reasons"):
        print("[wm] incomplete reasons: "
              + "; ".join(gate["incomplete_reasons"]))
    if not gate["passed"]:
        print("[wm] hierarchical training/evaluation will refuse this model "
              "unless --override-wm-gate is given.")


# ------------------------------------------------------------ gate checking


def load_world_model(cfg: dict, path: str | None = None,
                     override_gate: bool = False, allow_legacy: bool = False):
    """Load a world model, enforcing the planning gate (Priority 7).
    Returns (wm, meta). Refuses failed/incomplete/ungated models unless
    overridden."""
    from provenance import load_checkpoint  # local import: avoids cycles
    path = path or os.path.join(MODELS_DIR, "world_model.pt")
    state, meta = load_checkpoint(path, expected_compat=wm_compat(cfg),
                                  allow_legacy=allow_legacy)
    gate = meta.get("gate")
    status = None
    if isinstance(gate, dict):
        status = gate.get("status")
        passed = bool(gate.get("passed", False)) and status != "incomplete"
    else:
        passed = False
    if gate is None or not passed:
        if gate is None:
            detail = "has no recorded planning gate"
        elif status == "incomplete":
            detail = "has an INCOMPLETE planning gate"
        else:
            detail = "FAILED its planning gate"
        if not override_gate:
            raise ValueError(
                f"world model {path} {detail}. Planning on an unvalidated "
                f"model is disabled; retrain the model or pass "
                f"--override-wm-gate for explicit diagnostics only.")
        print(f"[wm] WARNING: using a world model that {detail} "
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
    ap.add_argument("--allow-legacy-buffer", action="store_true",
                    help="permit pre-provenance experience.npz reuse")
    a = ap.parse_args()
    train(force=a.force, allow_legacy_buffer=a.allow_legacy_buffer)
