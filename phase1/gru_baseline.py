"""Parameter-matched monolithic GRU comparator (Phase 1).

Consumes the SAME per-step inputs as the two specialists combined
(lidar 16 + body 2 + prev action 2 + normalized dt = 21), the same ordered
future-action conditioning (its own GRU action encoder), the same targets,
splits and optimizer. Its hidden size is chosen so that its TRAINABLE
parameter count is as close as possible to the modular system's trainable
count (the GRU recurrence trains; the reservoirs do not -- that asymmetry is
the point of comparison M = L_GRU - L_modular).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phase1 import HORIZON_STEPS, NOMINAL_DT
from phase1.predictors import (ACT_EMB, ActionEncoder, HorizonHeads,
                               _seed_module)


def count_trainable(module_or_params) -> int:
    if isinstance(module_or_params, nn.Module):
        params = [p for p in module_or_params.parameters() if p.requires_grad]
    else:
        params = list(module_or_params)
    return int(sum(p.numel() for p in params))


def count_total(module: nn.Module) -> int:
    n = sum(p.numel() for p in module.parameters())
    n += sum(b.numel() for b in module.buffers())
    return int(n)


class MonolithicGRU(nn.Module):
    def __init__(self, hidden: int, horizon_steps=None, seed: int = 0,
                 in_dim: int = 21, nominal_dt: float = NOMINAL_DT):
        super().__init__()
        hs = dict(horizon_steps or HORIZON_STEPS)
        self.horizon_steps = hs
        self.nominal_dt = nominal_dt
        self.hidden = int(hidden)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        _seed_module(self.gru, seed + 5)
        self.action_encoder = ActionEncoder(seed=seed + 6)
        self.heads = HorizonHeads(hidden + ACT_EMB, 18, hs, seed + 7)

    def context_inputs(self, batch_t: dict) -> torch.Tensor:
        dt_norm = batch_t["dts"] / self.nominal_dt
        return torch.cat([batch_t["lidar"], batch_t["body"],
                          batch_t["prev_actions"], dt_norm], dim=-1)

    def forward(self, batch_t: dict) -> dict:
        x = self.context_inputs(batch_t)
        out, _ = self.gru(x)
        # contexts are right-aligned, so the anchor is always the final step
        z = out[:, -1]
        act = self.action_encoder.at_horizons(
            batch_t["future_actions"], batch_t["future_dts"],
            self.horizon_steps)
        rel = self.heads({h: torch.cat([z, act[h]], -1)
                          for h in self.heads.horizons})
        return {"lidar": {h: v[:, :16] for h, v in rel.items()},
                "body": {h: v[:, 16:] for h, v in rel.items()}}

    def loss(self, preds: dict, batch_t: dict):
        parts = {}
        for h, steps in self.horizon_steps.items():
            tl = batch_t["future_lidar"][:, steps - 1]
            tb = batch_t["future_body"][:, steps - 1]
            parts[f"lidar_{h}"] = torch.mean((preds["lidar"][h] - tl) ** 2)
            parts[f"body_{h}"] = torch.mean((preds["body"][h] - tb) ** 2)
        total = torch.stack(list(parts.values())).mean()
        return total, {k: float(v) for k, v in parts.items()}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _analytic_trainable(hidden: int, n_horizons: int, in_dim: int) -> int:
    gru = 3 * hidden * in_dim + 3 * hidden * hidden + 6 * hidden
    act_enc = 3 * ACT_EMB * 3 + 3 * ACT_EMB * ACT_EMB + 6 * ACT_EMB
    head = (hidden + ACT_EMB) * 64 + 64 + 64 * 18 + 18
    return gru + act_enc + n_horizons * head


def matched_hidden_size(target_trainable: int, horizon_steps=None,
                        in_dim: int = 21, lo: int = 8, hi: int = 1024) -> int:
    """Hidden size whose trainable count best matches the modular system's
    trainable count (counts computed analytically; validated in tests)."""
    n_h = len(horizon_steps or HORIZON_STEPS)
    best, best_gap = lo, None
    for h in range(lo, hi + 1):
        gap = abs(_analytic_trainable(h, n_h, in_dim) - target_trainable)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
    return best
