"""Slow deliberative loop: cross-entropy-method search over imagined futures.

Every macro-step (~0.5 s) the planner rolls candidate action sequences through
the world model's latent dynamics, scores the imagined end states with the
readout head (get close to the goal, don't end up near a wall), and returns
the *latent state one chunk ahead* on the best plan. That latent is the
subgoal handed to the fast liquid policy.
"""
import numpy as np
import torch

from agent.world_model import WorldModel


class CEMPlanner:
    def __init__(self, world_model: WorldModel, horizon: int = 4,
                 population: int = 64, elites: int = 8, iterations: int = 3,
                 chunk_dt: float = 0.5, action_dim: int = 2):
        self.wm = world_model
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iterations = iterations
        self.chunk_dt = chunk_dt
        self.action_dim = action_dim

    @torch.no_grad()
    def plan(self, z0: torch.Tensor) -> torch.Tensor:
        """z0: (1, latent). Returns subgoal latent (1, latent)."""
        H, P, A = self.horizon, self.population, self.action_dim
        mean = torch.zeros(H, A)
        std = torch.ones(H, A) * 0.6
        dt = torch.full((P, 1), self.chunk_dt)

        best_first_z = z0
        for _ in range(self.iterations):
            actions = (mean.unsqueeze(0) + std.unsqueeze(0)
                       * torch.randn(P, H, A)).clamp(-1, 1)
            z = z0.expand(P, -1)
            penalty = torch.zeros(P)
            first_z = None
            for h in range(H):
                z = self.wm.predict_next(z, actions[:, h, :], dt)
                if h == 0:
                    first_z = z.clone()
                read = self.wm.readout(z)          # [goal_dist_norm, min_ray]
                penalty += torch.relu(0.08 - read[:, 1]) * 5.0   # wall proximity
            goal_dist = self.wm.readout(z)[:, 0]
            score = -goal_dist - penalty
            elite_idx = torch.topk(score, self.elites).indices
            elite = actions[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0) + 1e-3
            best_first_z = first_z[elite_idx[0]:elite_idx[0] + 1]
        return best_first_z
