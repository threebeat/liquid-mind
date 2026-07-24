"""Mandatory baseline suite for Phase 1 (before any interpretation).

- persistence: the last context observation persists to every horizon;
- linear autoregressive predictor: ridge regression from the flattened
  ordered context (+ ordered future actions up to the horizon) to each
  endpoint target;
- differential-drive kinematic body model: wheel commands -> steady-state
  forward speed / yaw rate (lidar has no kinematic model; body only);
- raw-history MLP: nonlinear readout of the SAME flattened context with the
  same GRU action encoder and per-horizon heads as the modular system.

Isolated-ESN (fusion off) and nominal-vs-physical-leak ESN variants are
configurations of ModularSystem, not separate classes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from environment.nav_env import AXLE_TRACK, WHEEL_RADIUS
from phase1 import HORIZON_STEPS, NOMINAL_DT
from phase1.predictors import ACT_EMB, ActionEncoder, HorizonHeads


def endpoint_targets(batch: dict, steps: int):
    return batch["future_lidar"][:, steps - 1], batch["future_body"][:, steps - 1]


def ridge_solve(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """(X^T X + lam I)^-1 X^T Y via torch LAPACK. NumPy's MKL solve hits the
    Windows 0xc06d007f fatal error in processes that also load pybullet;
    torch's linear algebra does not."""
    xt = torch.from_numpy(np.ascontiguousarray(x)).double()
    yt = torch.from_numpy(np.ascontiguousarray(y)).double()
    a = xt.T @ xt + lam * torch.eye(xt.shape[1], dtype=torch.float64)
    return torch.linalg.solve(a, xt.T @ yt).numpy()


def mm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Matrix product via torch (same 0xc06d007f workaround as ridge_solve:
    numpy MKL GEMM is fatal in pybullet-hosting processes on this machine)."""
    return (torch.from_numpy(np.ascontiguousarray(x)).double()
            @ torch.from_numpy(np.ascontiguousarray(w)).double()).numpy()


# ------------------------------------------------------------- persistence


def persistence_predictions(batch: dict, horizon_steps=None) -> dict:
    """Last context obs persists: same prediction at every horizon."""
    hs = horizon_steps or HORIZON_STEPS
    lid = batch["lidar"][:, -1]
    bod = batch["body"][:, -1]
    return {"lidar": {h: lid for h in hs}, "body": {h: bod for h in hs}}


# --------------------------------------------------------------- linear AR


class LinearARBaseline:
    """Per-horizon ridge from [flattened ordered context ; flattened ordered
    future actions up to the horizon] to the endpoint targets."""

    def __init__(self, horizon_steps=None, lam: float = 1e-2):
        self.horizon_steps = dict(horizon_steps or HORIZON_STEPS)
        self.lam = float(lam)
        self.weights: dict = {}

    @staticmethod
    def _context_features(batch: dict) -> np.ndarray:
        B = batch["lidar"].shape[0]
        return np.concatenate([
            batch["lidar"].reshape(B, -1), batch["body"].reshape(B, -1),
            batch["actions"].reshape(B, -1),
            (batch["dts"] / NOMINAL_DT).reshape(B, -1),
            batch["valid_mask"].reshape(B, -1)], axis=1)

    def _features(self, batch: dict, steps: int) -> np.ndarray:
        B = batch["lidar"].shape[0]
        fut = np.concatenate([
            batch["future_actions"][:, :steps].reshape(B, -1),
            (batch["future_dts"][:, :steps] / NOMINAL_DT).reshape(B, -1)],
            axis=1)
        x = np.concatenate([self._context_features(batch), fut], axis=1)
        return np.concatenate([x, np.ones((B, 1), np.float32)], axis=1)

    def fit(self, batch: dict):
        for h, steps in self.horizon_steps.items():
            x = self._features(batch, steps).astype(np.float64)
            tl, tb = endpoint_targets(batch, steps)
            y = np.concatenate([tl, tb], axis=1).astype(np.float64)
            self.weights[h] = ridge_solve(x, y, self.lam)
        return self

    def predict(self, batch: dict) -> dict:
        out = {"lidar": {}, "body": {}}
        for h, steps in self.horizon_steps.items():
            y = mm(self._features(batch, steps), self.weights[h])
            out["lidar"][h] = y[:, :16].astype(np.float32)
            out["body"][h] = y[:, 16:].astype(np.float32)
        return out


# ------------------------------------------------------------- ridge probe


def ridge_probe(x_train, y_train, x_val, y_val, lam: float = 1e-3) -> dict:
    """Linear probe metrics (message interpretability)."""
    x_tr = np.concatenate([x_train, np.ones((len(x_train), 1))], 1).astype(np.float64)
    x_va = np.concatenate([x_val, np.ones((len(x_val), 1))], 1).astype(np.float64)
    y_tr = np.asarray(y_train, np.float64).reshape(len(x_train), -1)
    y_va = np.asarray(y_val, np.float64).reshape(len(x_val), -1)
    w = ridge_solve(x_tr, y_tr, lam)
    pred = mm(x_va, w)
    mse = float(np.mean((pred - y_va) ** 2))
    var = float(np.mean((y_va - y_va.mean(0)) ** 2))
    return {"mse": mse, "r2": 1.0 - mse / var if var > 1e-12 else None}


# --------------------------------------------------------- kinematic (body)


def kinematic_body_predictions(batch: dict, max_wheel_speed: float = 20.0,
                               horizon_steps=None) -> dict:
    """Steady-state differential drive from the wheel command active at each
    horizon endpoint: v = R (wl + wr) / 2, w = R (wr - wl) / track,
    normalized like the observation (v / 1.2, w / 3)."""
    hs = horizon_steps or HORIZON_STEPS
    out = {}
    for h, steps in hs.items():
        a = batch["future_actions"][:, steps - 1] * max_wheel_speed
        v = WHEEL_RADIUS * (a[:, 0] + a[:, 1]) / 2.0
        w = WHEEL_RADIUS * (a[:, 1] - a[:, 0]) / AXLE_TRACK
        out[h] = np.stack([v / 1.2, w / 3.0], axis=1).astype(np.float32)
    return out


# ---------------------------------------------------------- raw-history MLP


class RawHistoryMLP(nn.Module):
    """Same flattened history as the linear AR, nonlinear trunk, same action
    encoder and per-horizon head structure as the modular system."""

    def __init__(self, context_len: int, horizon_steps=None, seed: int = 0,
                 trunk_dim: int = 64):
        super().__init__()
        hs = dict(horizon_steps or HORIZON_STEPS)
        self.horizon_steps = hs
        in_dim = context_len * (16 + 2 + 2 + 1 + 1)
        torch.manual_seed(seed + 77)
        self.trunk = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                   nn.Linear(128, trunk_dim), nn.ReLU())
        self.action_encoder = ActionEncoder(seed=seed + 78)
        self.heads = HorizonHeads(trunk_dim + ACT_EMB, 18, hs, seed + 79)

    def forward(self, batch_t: dict) -> dict:
        B = batch_t["lidar"].shape[0]
        x = torch.cat([
            batch_t["lidar"].reshape(B, -1), batch_t["body"].reshape(B, -1),
            batch_t["actions"].reshape(B, -1),
            (batch_t["dts"] / NOMINAL_DT).reshape(B, -1),
            batch_t["valid_mask"].reshape(B, -1)], dim=1)
        z = self.trunk(x)
        act = self.action_encoder.at_horizons(
            batch_t["future_actions"], batch_t["future_dts"],
            self.horizon_steps)
        rel = self.heads({h: torch.cat([z, act[h]], -1)
                          for h in self.heads.horizons})
        return {"lidar": {h: v[:, :16] for h, v in rel.items()},
                "body": {h: v[:, 16:] for h, v in rel.items()}}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
