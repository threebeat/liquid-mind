"""Local and relational predictors over specialist messages (Phase 1).

    D_L(m_L, a_{t:t+tau}) -> future lidar (16)   endpoint at each horizon
    D_B(m_B, a_{t:t+tau}) -> future body  (2)
    R(m_L, m_B, a_{t:t+tau}) -> future lidar + body (18)

Actions are consumed as an ORDERED sequence by a small GRU action encoder
(never a mean action); each horizon endpoint conditions on the encoder
hidden state after exactly that many future decisions. Heads are
Linear(msg + act_emb, 64) -> ReLU -> Linear(64, target_dim), one per
horizon. Only readouts, the action encoder and the heads train; the
reservoirs stay frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phase1 import HORIZON_STEPS, NOMINAL_DT
from phase1.esn import (MSG_DIM, ReservoirSpecialist, body_inputs,
                        lidar_inputs)

ACT_EMB = 32


def _seed_module(module: nn.Module, seed: int):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 1:
                p.data = torch.empty_like(p).uniform_(
                    -(p.shape[-1] ** -0.5 if p.dim() > 1 else 0.05),
                    (p.shape[-1] ** -0.5 if p.dim() > 1 else 0.05),
                    generator=gen)
    return module


class ActionEncoder(nn.Module):
    """GRU over ordered (action, dt) future sequences."""

    def __init__(self, hidden: int = ACT_EMB, seed: int = 0,
                 nominal_dt: float = NOMINAL_DT):
        super().__init__()
        self.nominal_dt = nominal_dt
        self.gru = nn.GRU(3, hidden, batch_first=True)
        _seed_module(self.gru, seed + 11)

    def forward(self, future_actions: torch.Tensor,
                future_dts: torch.Tensor) -> torch.Tensor:
        """[B, H, 2], [B, H, 1] seconds -> hidden at every step [B, H, emb]."""
        x = torch.cat([future_actions, future_dts / self.nominal_dt], dim=-1)
        out, _ = self.gru(x)
        return out

    def at_horizons(self, future_actions, future_dts,
                    horizon_steps: dict) -> dict:
        out = self.forward(future_actions, future_dts)
        return {h: out[:, steps - 1] for h, steps in horizon_steps.items()}


def _head(in_dim: int, target_dim: int, seed: int) -> nn.Module:
    m = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                      nn.Linear(64, target_dim))
    return _seed_module(m, seed)


class HorizonHeads(nn.Module):
    """One MLP head per prediction horizon."""

    def __init__(self, in_dim: int, target_dim: int, horizon_steps: dict,
                 seed: int = 0):
        super().__init__()
        self.horizons = sorted(horizon_steps)
        self.heads = nn.ModuleDict({
            self._key(h): _head(in_dim, target_dim, seed + 100 + i)
            for i, h in enumerate(self.horizons)})

    @staticmethod
    def _key(h) -> str:
        return str(h).replace(".", "_")

    def forward(self, feats_by_h: dict) -> dict:
        return {h: self.heads[self._key(h)](feats_by_h[h])
                for h in self.horizons}


class LocalPredictor(nn.Module):
    """D_x(message, ordered future actions) -> endpoint target per horizon."""

    def __init__(self, target_dim: int, horizon_steps: dict, seed: int = 0,
                 msg_dim: int = MSG_DIM, act_emb: int = ACT_EMB):
        super().__init__()
        self.heads = HorizonHeads(msg_dim + act_emb, target_dim,
                                  horizon_steps, seed)

    def forward(self, msg: torch.Tensor, act_by_h: dict) -> dict:
        feats = {h: torch.cat([msg, act_by_h[h]], dim=-1)
                 for h in self.heads.horizons}
        return self.heads(feats)


class RelationalPredictor(nn.Module):
    """R(m_L, m_B, ordered future actions) -> future lidar + body."""

    def __init__(self, horizon_steps: dict, seed: int = 0,
                 msg_dim: int = MSG_DIM, act_emb: int = ACT_EMB):
        super().__init__()
        self.heads = HorizonHeads(2 * msg_dim + act_emb, 18,
                                  horizon_steps, seed)

    def forward(self, m_l: torch.Tensor, m_b: torch.Tensor,
                act_by_h: dict) -> dict:
        feats = {h: torch.cat([m_l, m_b, act_by_h[h]], dim=-1)
                 for h in self.heads.horizons}
        return self.heads(feats)


class ModularSystem(nn.Module):
    """Two frozen specialists + trainable readouts, action encoder, local
    predictors, and (unless fusion is disabled) the relational predictor."""

    def __init__(self, lidar_esn: ReservoirSpecialist,
                 body_esn: ReservoirSpecialist, model_seed: int = 0,
                 horizon_steps: dict | None = None, fusion: bool = True,
                 nominal_dt: float = NOMINAL_DT):
        super().__init__()
        hs = dict(horizon_steps or HORIZON_STEPS)
        self.horizon_steps = hs
        self.nominal_dt = nominal_dt
        self.fusion = bool(fusion)
        self.lidar_esn = lidar_esn
        self.body_esn = body_esn
        self.action_encoder = ActionEncoder(seed=model_seed)
        self.d_lidar = LocalPredictor(16, hs, seed=model_seed + 1)
        self.d_body = LocalPredictor(2, hs, seed=model_seed + 2)
        self.relational = RelationalPredictor(hs, seed=model_seed + 3) \
            if fusion else None

    # ------------------------------------------------------------ features

    def specialist_features(self, batch_t: dict):
        """Frozen [r; x] features at the anchor step (cacheable)."""
        xl = lidar_inputs(batch_t, self.nominal_dt,
                          getattr(self.lidar_esn, "action_input", False))
        xb = body_inputs(batch_t, self.nominal_dt)
        mask = batch_t.get("valid_mask")
        fl = self.lidar_esn.features(xl, batch_t["dts"], mask)
        fb = self.body_esn.features(xb, batch_t["dts"], mask)
        return fl, fb

    def forward_from_features(self, feat_l, feat_b, future_actions,
                              future_dts) -> dict:
        m_l = self.lidar_esn.message_from_features(feat_l)
        m_b = self.body_esn.message_from_features(feat_b)
        act = self.action_encoder.at_horizons(future_actions, future_dts,
                                              self.horizon_steps)
        preds = {"local_lidar": self.d_lidar(m_l, act),
                 "local_body": self.d_body(m_b, act)}
        if self.fusion:
            rel = self.relational(m_l, m_b, act)
            preds["rel_lidar"] = {h: v[:, :16] for h, v in rel.items()}
            preds["rel_body"] = {h: v[:, 16:] for h, v in rel.items()}
        return preds

    def forward(self, batch_t: dict) -> dict:
        fl, fb = self.specialist_features(batch_t)
        return self.forward_from_features(fl, fb, batch_t["future_actions"],
                                          batch_t["future_dts"])

    # --------------------------------------------------------------- loss

    def loss(self, preds: dict, batch_t: dict):
        """0.5 * MSE_lidar + 0.5 * MSE_body per stream/horizon; total is the
        mean over streams and horizons (relational is the system output)."""
        parts = {}
        for h, steps in self.horizon_steps.items():
            tl = batch_t["future_lidar"][:, steps - 1]
            tb = batch_t["future_body"][:, steps - 1]
            for stream in ("local", "rel") if self.fusion else ("local",):
                pl = preds[f"{stream}_lidar"][h]
                pb = preds[f"{stream}_body"][h]
                parts[f"{stream}_lidar_{h}"] = torch.mean((pl - tl) ** 2)
                parts[f"{stream}_body_{h}"] = torch.mean((pb - tb) ** 2)
        total = torch.stack(list(parts.values())).mean()
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
