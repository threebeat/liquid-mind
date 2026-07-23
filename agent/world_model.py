"""JEPA-style latent world model.

Predicts the future in *embedding space*, never in raw observation space:

    z_t = encoder(obs_t)                         (context encoder, trained)
    z'_{t+1chunk} = z_t + predictor(z_t, a, dt)  (latent dynamics, residual)
    target = target_encoder(obs_{t+chunk})       (EMA copy, no gradients)

Loss = MSE(z', target) + a small variance hinge that keeps embedding
dimensions from collapsing to a constant (the classic JEPA/BYOL failure mode;
the EMA target does most of the anti-collapse work, the hinge is insurance).

A supervised readout head grounds the latent in control-relevant geometry:
    [goal_dist, sin(bearing), cos(bearing),
     min ray front, min ray left, min ray back, min ray right]
Two scalars (distance + nearest wall) would be enough to score danger but
too ambiguous to plan around it — "obstacle on the left" and "obstacle on
the right" must map to different latents for a detour to be selectable.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ray indices per body quadrant (16 rays, index i at angle 2*pi*i/16,
# 0 = straight ahead in the robot frame)
_QUADRANTS = {"front": [14, 15, 0, 1], "left": [2, 3, 4, 5],
              "back": [6, 7, 8, 9], "right": [10, 11, 12, 13]}
READOUT_DIM = 7


def readout_targets(obs: torch.Tensor) -> torch.Tensor:
    """Ground-truth grounding signals, all extractable from the observation."""
    rays = obs[:, :16]
    cols = [obs[:, 16], obs[:, 17], obs[:, 18]]      # goal dist, sin/cos bearing
    for idx in _QUADRANTS.values():
        cols.append(rays[:, idx].min(dim=1).values)
    return torch.stack(cols, dim=1)


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 16, action_dim: int = 2,
                 hidden: int = 64, ema_momentum: float = 0.995):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.encoder = _mlp([obs_dim, hidden, latent_dim])
        self.target_encoder = copy.deepcopy(self.encoder)
        for prm in self.target_encoder.parameters():
            prm.requires_grad_(False)
        # predictor input: latent + mean action over the chunk + chunk duration
        self.predictor = _mlp([latent_dim + action_dim + 1, hidden, latent_dim])
        self.readout = _mlp([latent_dim, 32, READOUT_DIM])

    @torch.no_grad()
    def update_target(self):
        m = self.ema_momentum
        for pt, po in zip(self.target_encoder.parameters(),
                          self.encoder.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1.0 - m)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_next(self, z: torch.Tensor, mean_action: torch.Tensor,
                     chunk_dt: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, mean_action, chunk_dt], dim=-1)
        return z + self.predictor(inp)

    def loss(self, obs_t, mean_action, chunk_dt, obs_next):
        z = self.encode(obs_t)
        z_pred = self.predict_next(z, mean_action, chunk_dt)
        with torch.no_grad():
            z_tgt = self.target_encoder(obs_next)
        pred_loss = F.mse_loss(z_pred, z_tgt)

        # variance hinge: every latent dim should keep std >= 0.1 across batch
        std = z.std(dim=0)
        var_loss = F.relu(0.1 - std).mean()

        # supervised readout on ground-truth signals present in the obs.
        # No detach: the task signal shapes the encoder, grounding the latent
        # space in operationally meaningful quantities (and fighting collapse
        # more directly than the variance hinge alone).
        read_loss = F.mse_loss(self.readout(z), readout_targets(obs_t))

        return pred_loss + 0.5 * var_loss + read_loss, {
            "pred": pred_loss.item(), "var": var_loss.item(),
            "read": read_loss.item(), "z_std": std.mean().item()}
