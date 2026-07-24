"""Training / evaluation harness for Phase 1.

All architectures see identical windows, targets, splits and optimizer
(Adam). Because the reservoirs are frozen, specialist [r; x] anchor features
are precomputed once per (data, reservoir-seed, config) and reused every
epoch -- gradients only ever flow through readouts, the action encoder and
the predictor heads.

Endpoint metrics per horizon (normalized observation units):
    lidar_mse, body_mse, combined = 0.5 * lidar_mse + 0.5 * body_mse,
    front_clearance_mae = |min(front rays pred) - min(front rays true)|.

Primary endpoint S = L_model / L_persistence on `combined` at 0.5 s.
Complementarity C = L_best_isolated - L_combined (isolated run's local
heads vs fused run's relational heads). Modular advantage
M = L_GRU - L_modular. All are computed by the stage runners from the
per-model metrics emitted here; this module never renders verdicts.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from phase1 import HORIZON_STEPS, PRIMARY_HORIZON_S
from phase1.dataset import WindowDataset

FRONT_RAYS = [14, 15, 0, 1]   # matches agent.world_model._QUADRANTS["front"]


# ------------------------------------------------------------------- data


def materialize(dataset: WindowDataset, indices=None) -> dict:
    """Stack windows once into numpy arrays ([N, ...])."""
    if indices is None:
        indices = range(len(dataset))
    return dataset.batch(list(indices))


def take(arrays: dict, idx) -> dict:
    """Torch batch dict for a set of window indices."""
    return {k: torch.from_numpy(np.ascontiguousarray(v[idx]))
            for k, v in arrays.items()}


def n_windows(arrays: dict) -> int:
    return int(arrays["lidar"].shape[0])


# ---------------------------------------------------------------- metrics


def endpoint_metrics(pred_lidar: np.ndarray, pred_body: np.ndarray,
                     true_lidar: np.ndarray, true_body: np.ndarray) -> dict:
    lm = float(np.mean((pred_lidar - true_lidar) ** 2))
    bm = float(np.mean((pred_body - true_body) ** 2))
    fc_pred = pred_lidar[:, FRONT_RAYS].min(axis=1)
    fc_true = true_lidar[:, FRONT_RAYS].min(axis=1)
    return {"lidar_mse": lm, "body_mse": bm,
            "combined": 0.5 * lm + 0.5 * bm,
            "front_clearance_mae": float(np.mean(np.abs(fc_pred - fc_true)))}


def metrics_over_horizons(preds: dict, arrays: dict, idx=None) -> dict:
    """preds: {"lidar": {h: [N,16]}, "body": {h: [N,2]}} numpy arrays."""
    out = {}
    sel = slice(None) if idx is None else idx
    for h, steps in HORIZON_STEPS.items():
        tl = arrays["future_lidar"][sel][:, steps - 1]
        tb = arrays["future_body"][sel][:, steps - 1]
        out[str(h)] = endpoint_metrics(np.asarray(preds["lidar"][h]),
                                       np.asarray(preds["body"][h]), tl, tb)
    return out


def per_episode_metrics(preds: dict, arrays: dict, episode_ids: np.ndarray,
                        horizon: float = PRIMARY_HORIZON_S) -> dict:
    """Per-episode combined loss at one horizon (for clustered bootstraps)."""
    steps = HORIZON_STEPS[horizon]
    tl = arrays["future_lidar"][:, steps - 1]
    tb = arrays["future_body"][:, steps - 1]
    pl, pb = np.asarray(preds["lidar"][horizon]), np.asarray(preds["body"][horizon])
    out = {}
    for ep in np.unique(episode_ids):
        m = episode_ids == ep
        lm = float(np.mean((pl[m] - tl[m]) ** 2))
        bm = float(np.mean((pb[m] - tb[m]) ** 2))
        out[int(ep)] = 0.5 * lm + 0.5 * bm
    return out


def episode_ids_for(dataset: WindowDataset, indices=None) -> np.ndarray:
    """GLOBAL buffer episode index per window (for clustering)."""
    if indices is None:
        indices = range(len(dataset))
    return np.array([dataset.episode_indices[dataset.anchors[int(i)][0]]
                     for i in indices])


# ----------------------------------------------------------- model runners


def _batched(n: int, batch_size: int):
    for s in range(0, n, batch_size):
        yield np.arange(s, min(s + batch_size, n))


@torch.no_grad()
def predict_torch_model(model, arrays: dict, batch_size: int = 512) -> dict:
    """Any nn.Module whose forward(batch_t) returns
    {"lidar": {h: tensor}, "body": {h: tensor}}."""
    model.eval()
    chunks = {"lidar": {h: [] for h in HORIZON_STEPS},
              "body": {h: [] for h in HORIZON_STEPS}}
    for idx in _batched(n_windows(arrays), batch_size):
        p = model(take(arrays, idx))
        for h in HORIZON_STEPS:
            chunks["lidar"][h].append(p["lidar"][h].numpy())
            chunks["body"][h].append(p["body"][h].numpy())
    return {k: {h: np.concatenate(v) for h, v in d.items()}
            for k, d in chunks.items()}


@torch.no_grad()
def predict_modular(system, feats: dict, arrays: dict, stream: str,
                    batch_size: int = 512) -> dict:
    """stream: "rel" (fused system output) or "local" (isolated heads)."""
    system.eval()
    chunks = {"lidar": {h: [] for h in HORIZON_STEPS},
              "body": {h: [] for h in HORIZON_STEPS}}
    for idx in _batched(n_windows(arrays), batch_size):
        p = system.forward_from_features(
            feats["lidar"][idx], feats["body"][idx],
            torch.from_numpy(arrays["future_actions"][idx]),
            torch.from_numpy(arrays["future_dts"][idx]))
        for h in HORIZON_STEPS:
            chunks["lidar"][h].append(p[f"{stream}_lidar"][h].numpy())
            chunks["body"][h].append(p[f"{stream}_body"][h].numpy())
    return {k: {h: np.concatenate(v) for h, v in d.items()}
            for k, d in chunks.items()}


@torch.no_grad()
def precompute_features(system, arrays: dict, batch_size: int = 256) -> dict:
    """Frozen [r; x] anchor features for both specialists."""
    fl, fb = [], []
    for idx in _batched(n_windows(arrays), batch_size):
        bt = take(arrays, idx)
        a, b = system.specialist_features(bt)
        fl.append(a)
        fb.append(b)
    return {"lidar": torch.cat(fl), "body": torch.cat(fb)}


# ----------------------------------------------------------------- training


def train_modular(system, train_arrays: dict, val_arrays: dict, epochs: int,
                  batch_size: int, lr: float, seed: int,
                  log=print) -> dict:
    """Adam over readouts + action encoder + heads, cached frozen features."""
    t0 = time.perf_counter()
    feats_tr = precompute_features(system, train_arrays)
    feats_va = precompute_features(system, val_arrays)
    feat_time = time.perf_counter() - t0

    from phase1.esn import reservoir_health
    # health check on a sample of raw state trajectories (exclusion rules)
    bt = take(train_arrays, np.arange(min(256, n_windows(train_arrays))))
    from phase1.esn import body_inputs, lidar_inputs
    xl = lidar_inputs(bt, system.nominal_dt,
                      getattr(system.lidar_esn, "action_input", False))
    xb = body_inputs(bt, system.nominal_dt)
    with torch.no_grad():
        _, st_l, _ = system.lidar_esn(xl, bt["dts"], bt["valid_mask"])
        _, st_b, _ = system.body_esn(xb, bt["dts"], bt["valid_mask"])
    health = {"lidar": reservoir_health(st_l), "body": reservoir_health(st_b)}

    opt = torch.optim.Adam(system.trainable_parameters(), lr=lr)
    fut_a = torch.from_numpy(train_arrays["future_actions"])
    fut_d = torch.from_numpy(train_arrays["future_dts"])
    tgt = {h: (torch.from_numpy(train_arrays["future_lidar"][:, s - 1]),
               torch.from_numpy(train_arrays["future_body"][:, s - 1]))
           for h, s in system.horizon_steps.items()}
    n = n_windows(train_arrays)
    history = {"train_loss": [], "val_loss": [], "status": "ok"}
    for ep in range(epochs):
        system.train()
        order = np.random.default_rng(seed * 1000 + ep).permutation(n)
        tot, nb = 0.0, 0
        for idx in _batched(n, batch_size):
            sel = order[idx]
            preds = system.forward_from_features(
                feats_tr["lidar"][sel], feats_tr["body"][sel],
                fut_a[sel], fut_d[sel])
            parts = []
            for h in system.horizon_steps:
                tl, tb = tgt[h][0][sel], tgt[h][1][sel]
                streams = ("local", "rel") if system.fusion else ("local",)
                for stream in streams:
                    parts.append(torch.mean(
                        (preds[f"{stream}_lidar"][h] - tl) ** 2))
                    parts.append(torch.mean(
                        (preds[f"{stream}_body"][h] - tb) ** 2))
            loss = torch.stack(parts).mean()
            if not torch.isfinite(loss):
                history["status"] = "nan_loss"
                log(f"[train_modular] NaN/Inf loss at epoch {ep}; aborting")
                return {"history": history, "health": health,
                        "wall_clock_s": time.perf_counter() - t0,
                        "feature_time_s": feat_time}
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        val = _val_loss_modular(system, feats_va, val_arrays)
        history["train_loss"].append(tot / max(nb, 1))
        history["val_loss"].append(val)
        log(f"[train_modular] epoch {ep + 1}/{epochs} "
            f"train={tot / max(nb, 1):.5f} val={val:.5f}")
    return {"history": history, "health": health,
            "wall_clock_s": time.perf_counter() - t0,
            "feature_time_s": feat_time,
            "features": {"train": feats_tr, "val": feats_va}}


@torch.no_grad()
def _val_loss_modular(system, feats: dict, arrays: dict,
                      batch_size: int = 1024) -> float:
    system.eval()
    parts, weights = [], []
    fut_a = torch.from_numpy(arrays["future_actions"])
    fut_d = torch.from_numpy(arrays["future_dts"])
    for idx in _batched(n_windows(arrays), batch_size):
        preds = system.forward_from_features(
            feats["lidar"][idx], feats["body"][idx], fut_a[idx], fut_d[idx])
        bt = {"future_lidar": torch.from_numpy(arrays["future_lidar"][idx]),
              "future_body": torch.from_numpy(arrays["future_body"][idx])}
        loss, _ = system.loss(preds, bt)
        parts.append(float(loss))
        weights.append(len(idx))
    return float(np.average(parts, weights=weights))


def train_torch_model(model, train_arrays: dict, val_arrays: dict,
                      epochs: int, batch_size: int, lr: float, seed: int,
                      log=print) -> dict:
    """Generic Adam loop for GRU / raw-history MLP style models with
    forward(batch_t) -> {"lidar": {h: t}, "body": {h: t}}."""
    t0 = time.perf_counter()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    n = n_windows(train_arrays)
    history = {"train_loss": [], "val_loss": [], "status": "ok"}
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed * 1000 + ep).permutation(n)
        tot, nb = 0.0, 0
        for idx in _batched(n, batch_size):
            bt = take(train_arrays, order[idx])
            preds = model(bt)
            parts = []
            for h, s in HORIZON_STEPS.items():
                parts.append(torch.mean(
                    (preds["lidar"][h] - bt["future_lidar"][:, s - 1]) ** 2))
                parts.append(torch.mean(
                    (preds["body"][h] - bt["future_body"][:, s - 1]) ** 2))
            loss = torch.stack(parts).mean()
            if not torch.isfinite(loss):
                history["status"] = "nan_loss"
                log(f"[train_torch] NaN/Inf loss at epoch {ep}; aborting")
                return {"history": history,
                        "wall_clock_s": time.perf_counter() - t0}
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        val = _val_loss_torch(model, val_arrays)
        history["train_loss"].append(tot / max(nb, 1))
        history["val_loss"].append(val)
        log(f"[train_torch] epoch {ep + 1}/{epochs} "
            f"train={tot / max(nb, 1):.5f} val={val:.5f}")
    return {"history": history, "wall_clock_s": time.perf_counter() - t0}


@torch.no_grad()
def _val_loss_torch(model, arrays: dict, batch_size: int = 1024) -> float:
    model.eval()
    parts, weights = [], []
    for idx in _batched(n_windows(arrays), batch_size):
        bt = take(arrays, idx)
        preds = model(bt)
        ps = []
        for h, s in HORIZON_STEPS.items():
            ps.append(torch.mean(
                (preds["lidar"][h] - bt["future_lidar"][:, s - 1]) ** 2))
            ps.append(torch.mean(
                (preds["body"][h] - bt["future_body"][:, s - 1]) ** 2))
        parts.append(float(torch.stack(ps).mean()))
        weights.append(len(idx))
    return float(np.average(parts, weights=weights))


# ------------------------------------------------------ derived quantities


def s_ratio(model_metrics: dict, persistence_metrics: dict,
            horizon: float = PRIMARY_HORIZON_S, key: str = "combined"):
    denom = persistence_metrics[str(horizon)][key]
    if denom <= 0:
        return None
    return model_metrics[str(horizon)][key] / denom


def inference_cost(model_fn, arrays: dict, n_repeat: int = 20) -> dict:
    """Wall-clock per single-window forward (full pipeline, batch of 1)."""
    bt_idx = np.arange(1)
    model_fn(bt_idx)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        model_fn(bt_idx)
    return {"per_window_ms": (time.perf_counter() - t0) / n_repeat * 1e3}


def partition_consistency(esn, arrays: dict, n_samples: int = 64) -> dict:
    """Physical-time leak: split every context interval in two and compare
    final states (secondary endpoint: temporal-partition consistency)."""
    idx = np.arange(min(n_samples, n_windows(arrays)))
    bt = take(arrays, idx)
    from phase1.esn import lidar_inputs
    x = lidar_inputs(bt, esn.cfg.nominal_dt,
                     getattr(esn, "action_input", False)) \
        if esn.cfg.n_inputs >= 16 else None
    if x is None:
        from phase1.esn import body_inputs
        x = body_inputs(bt, esn.cfg.nominal_dt)
    dts = bt["dts"].reshape(len(idx), -1)
    with torch.no_grad():
        _, _, r_whole = esn(x, dts, bt["valid_mask"])
        x2 = torch.repeat_interleave(x, 2, dim=1)
        d2 = torch.repeat_interleave(dts / 2.0, 2, dim=1)
        m2 = torch.repeat_interleave(bt["valid_mask"], 2, dim=1)
        _, _, r_split = esn(x2, d2, m2)
    diff = (r_whole - r_split).abs()
    return {"max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean())}
