"""Fixed echo-state specialists with trainable message readouts (Phase 1).

A ReservoirSpecialist holds frozen sparse W_in / W_rec / b (registered as
buffers: ZERO trainable reservoir parameters, deterministic per seed, W_rec
rescaled to the configured spectral radius) and a single trainable linear
message readout with input skip:

    r_{t+1} = (1 - alpha_t) r_t + alpha_t tanh(W_in x_t + W_rec r_t + b)
    m_t     = W_msg [r_t ; x_t] + b_msg          (m_t in R^16)

Leak modes:
    nominal   alpha_t = alpha_0 (constant per decision)
    physical  alpha_t = 1 - exp(-dt_t / tau), tau derived so that the leak
              matches alpha_0 at the nominal decision interval unless an
              explicit tau is configured.

Specialist input contracts (strict isolation):
    lidar  16 rays + normalized dt (+ declared wheel actions only when
           cfg.action_input is True). No goal, body state, pose, or labels.
    body   previous wheel action (2) + fwd speed + yaw rate + normalized dt.
           No lidar.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from phase1 import NOMINAL_DT

MSG_DIM = 16


@dataclass
class ReservoirConfig:
    n_units: int
    n_inputs: int
    seed: int
    spectral_radius: float = 0.9
    alpha0: float = 0.3            # nominal leak per decision
    sparsity: float = 0.1          # W_rec density
    input_scale: float = 0.5
    input_density: float = 0.5     # W_in density (fixed, not in the grid)
    bias_scale: float = 0.1
    leak_mode: str = "nominal"     # "nominal" | "physical"
    tau: float | None = None       # physical-time constant; derived if None
    msg_dim: int = MSG_DIM
    nominal_dt: float = NOMINAL_DT

    def derived_tau(self) -> float:
        if self.tau is not None:
            return float(self.tau)
        return -self.nominal_dt / math.log(1.0 - self.alpha0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tau_effective"] = self.derived_tau()
        return d


def _make_weights(cfg: ReservoirConfig):
    """Deterministic frozen weights from the config seed."""
    rng = np.random.default_rng(cfg.seed)
    w_in = rng.uniform(-cfg.input_scale, cfg.input_scale,
                       (cfg.n_units, cfg.n_inputs))
    w_in *= rng.random((cfg.n_units, cfg.n_inputs)) < cfg.input_density
    w_rec = rng.uniform(-1.0, 1.0, (cfg.n_units, cfg.n_units))
    w_rec *= rng.random((cfg.n_units, cfg.n_units)) < cfg.sparsity
    # torch LAPACK: numpy's MKL eig/solve can hit the Windows 0xc06d007f
    # fatal error in processes that also load pybullet
    eig = float(torch.linalg.eigvals(
        torch.from_numpy(w_rec).double()).abs().max())
    if eig < 1e-12:
        raise ValueError(f"degenerate reservoir (seed {cfg.seed}): "
                         f"spectral radius ~ 0; regenerate with another seed")
    w_rec *= cfg.spectral_radius / eig
    b = rng.uniform(-cfg.bias_scale, cfg.bias_scale, cfg.n_units)
    return (w_in.astype(np.float32), w_rec.astype(np.float32),
            b.astype(np.float32))


class ReservoirSpecialist(nn.Module):
    def __init__(self, cfg: ReservoirConfig):
        super().__init__()
        self.cfg = cfg
        w_in, w_rec, b = _make_weights(cfg)
        self.register_buffer("w_in", torch.from_numpy(w_in))
        self.register_buffer("w_rec", torch.from_numpy(w_rec))
        self.register_buffer("b", torch.from_numpy(b))
        self.register_buffer("r0", torch.zeros(cfg.n_units))
        self.tau = cfg.derived_tau()
        # trainable message readout with input skip (the ONLY parameters)
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        self.readout = nn.Linear(cfg.n_units + cfg.n_inputs, cfg.msg_dim)
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.readout.in_features)
            self.readout.weight.uniform_(-bound, bound, generator=gen)
            self.readout.bias.uniform_(-bound, bound, generator=gen)
        self.state: torch.Tensor | None = None

    # -------------------------------------------------------------- state

    def reset(self, batch: int = 1) -> torch.Tensor:
        self.state = self.r0.expand(batch, -1).clone()
        return self.state

    def alpha(self, dt_seconds: torch.Tensor) -> torch.Tensor:
        """Leak per update. dt_seconds: [...] tensor (ignored in nominal)."""
        if self.cfg.leak_mode == "nominal":
            return torch.full_like(dt_seconds, self.cfg.alpha0)
        return 1.0 - torch.exp(-dt_seconds / self.tau)

    def step(self, x: torch.Tensor, dt_seconds: torch.Tensor,
             r: torch.Tensor) -> torch.Tensor:
        """One update. x: [B, n_in], dt_seconds: [B] or [B,1], r: [B, N]."""
        a = self.alpha(dt_seconds.reshape(-1, 1))
        pre = x @ self.w_in.T + r @ self.w_rec.T + self.b
        return (1.0 - a) * r + a * torch.tanh(pre)

    def forward(self, x_seq: torch.Tensor, dt_seq: torch.Tensor,
                mask: torch.Tensor | None = None,
                r: torch.Tensor | None = None):
        """Batched sequences. x_seq: [B, L, n_in], dt_seq: [B, L] or
        [B, L, 1] seconds, mask: [B, L] (state held where mask = 0).
        Returns (messages [B, L, msg], states [B, L, N], final state)."""
        B, L, _ = x_seq.shape
        dt_seq = dt_seq.reshape(B, L)
        if r is None:
            r = self.r0.expand(B, -1).clone()
        states, msgs = [], []
        for t in range(L):
            r_new = self.step(x_seq[:, t], dt_seq[:, t], r)
            if mask is not None:
                m = mask[:, t].reshape(-1, 1)
                r = m * r_new + (1.0 - m) * r
            else:
                r = r_new
            states.append(r)
            msgs.append(self.readout(torch.cat([r, x_seq[:, t]], dim=-1)))
        return torch.stack(msgs, 1), torch.stack(states, 1), r

    def features(self, x_seq, dt_seq, mask=None) -> torch.Tensor:
        """Frozen [r_t ; x_t] at the LAST context step -- cacheable, since
        the reservoir never trains. messages = readout(features)."""
        with torch.no_grad():
            _, states, r_last = self.forward(x_seq, dt_seq, mask)
        return torch.cat([r_last, x_seq[:, -1]], dim=-1)

    def message_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.readout(feats)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ----------------------------------------------------- specialist factories


def lidar_input_dim(action_input: bool = False) -> int:
    return 16 + 1 + (2 if action_input else 0)


BODY_INPUT_DIM = 5  # prev action (2) + fwd speed + yaw rate + dt


def make_lidar_specialist(seed: int, spectral_radius=0.9, alpha0=0.3,
                          sparsity=0.1, input_scale=0.5,
                          leak_mode="nominal", tau=None,
                          action_input: bool = False) -> ReservoirSpecialist:
    cfg = ReservoirConfig(
        n_units=128, n_inputs=lidar_input_dim(action_input), seed=seed,
        spectral_radius=spectral_radius, alpha0=alpha0, sparsity=sparsity,
        input_scale=input_scale, leak_mode=leak_mode, tau=tau)
    sp = ReservoirSpecialist(cfg)
    sp.action_input = bool(action_input)
    return sp


def make_body_specialist(seed: int, spectral_radius=0.9, alpha0=0.3,
                         sparsity=0.1, input_scale=0.5,
                         leak_mode="nominal", tau=None) -> ReservoirSpecialist:
    cfg = ReservoirConfig(
        n_units=64, n_inputs=BODY_INPUT_DIM, seed=seed,
        spectral_radius=spectral_radius, alpha0=alpha0, sparsity=sparsity,
        input_scale=input_scale, leak_mode=leak_mode, tau=tau)
    return ReservoirSpecialist(cfg)


def lidar_inputs(batch_t: dict, nominal_dt: float = NOMINAL_DT,
                 action_input: bool = False) -> torch.Tensor:
    """[B, L, 17(+2)] from a torch batch dict (strictly lidar + dt)."""
    dt_norm = batch_t["dts"] / nominal_dt
    parts = [batch_t["lidar"], dt_norm]
    if action_input:
        parts.append(batch_t["actions"])
    return torch.cat(parts, dim=-1)


def body_inputs(batch_t: dict, nominal_dt: float = NOMINAL_DT) -> torch.Tensor:
    """[B, L, 5]: prev wheel action, fwd speed, yaw rate, dt. No lidar."""
    dt_norm = batch_t["dts"] / nominal_dt
    return torch.cat([batch_t["prev_actions"], batch_t["body"], dt_norm],
                     dim=-1)


# --------------------------------------------------- technical-failure rules


def reservoir_health(states: torch.Tensor) -> dict:
    """Preregistered technical-failure diagnostics over states [B, L, N]."""
    s = states.detach()
    flat = s.reshape(-1, s.shape[-1])
    finite = bool(torch.isfinite(s).all())
    max_abs = float(s.abs().max()) if finite else float("inf")
    sat_frac = float((s.abs() > 0.99).float().mean()) if finite else 1.0
    std_time = float(s.std(dim=1).mean()) if finite else 0.0
    eff_rank = 0.0
    if finite and flat.shape[0] > 1:
        c = torch.cov(flat.T.double())
        ev = torch.linalg.eigvalsh(c).clamp(min=0)
        tot = float(ev.sum())
        if tot > 1e-12:
            eff_rank = float(tot ** 2 / (ev ** 2).sum())  # participation ratio
    failures = []
    if not finite:
        failures.append("nan_inf")
    if sat_frac > 0.90:
        failures.append("saturation")
    if std_time < 1e-5:
        failures.append("constant_collapse")
    if finite and eff_rank < 2.0:
        failures.append("rank_collapse")
    if max_abs > 1e3:
        failures.append("state_explosion")
    return {"finite": finite, "max_abs": max_abs, "saturation_frac": sat_frac,
            "mean_std_over_time": std_time, "effective_rank": eff_rank,
            "failures": failures, "ok": not failures}
