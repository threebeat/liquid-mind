"""Liquid (CfC) policy: continuous-time recurrent controller.

Uses the closed-form continuous-time (CfC) cell from `ncps` with a sparse
AutoNCP wiring — the same family of "liquid networks" used for drone and
lane-keeping control. The key property: `forward` accepts the *elapsed time*
of each step (timespans), so the policy natively handles an irregular sensor
stream instead of assuming a fixed tick rate.

Input at each control step: [SNN spikes | raw observation | subgoal latent].
The subgoal latent is zeros in reactive (Phase 2) mode and is produced by the
JEPA planner in hierarchical (Phase 4) mode.
"""
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class LiquidPolicy(nn.Module):
    def __init__(self, in_dim: int, units: int = 24, out_dim: int = 2):
        super().__init__()
        self.units = units
        wiring = AutoNCP(units, out_dim)
        self.rnn = CfC(in_dim, wiring, batch_first=True)

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.units)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, dt: float):
        """x: (batch, features) single step. Returns action in [-1,1], new state."""
        ts = torch.full((x.shape[0], 1), float(dt))
        out, hx = self.rnn(x.unsqueeze(1), hx, timespans=ts)
        return torch.tanh(out.squeeze(1)), hx
