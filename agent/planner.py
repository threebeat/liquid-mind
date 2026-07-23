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
    def plan(self, z0: torch.Tensor, return_info: bool = False):
        """z0: (1, latent). Returns subgoal latent (1, latent).

        With return_info=True also returns a dict with the candidate-score
        distribution of the final iteration, the best score seen, and the
        predicted readout trajectory along the chosen plan — used for
        on-policy world-model validation (predicted vs realized).
        """
        H, P, A = self.horizon, self.population, self.action_dim
        mean = torch.zeros(H, A)
        std = torch.ones(H, A) * 0.6
        dt = torch.full((P, 1), self.chunk_dt)

        best_first_z = z0
        best_score = -float("inf")
        best_traj = None
        last_scores = None
        for _ in range(self.iterations):
            actions = (mean.unsqueeze(0) + std.unsqueeze(0)
                       * torch.randn(P, H, A)).clamp(-1, 1)
            z = z0.expand(P, -1)
            penalty = torch.zeros(P)
            zs = []
            for h in range(H):
                z = self.wm.predict_next(z, actions[:, h, :], dt)
                zs.append(z)
                # readout: [goal_dist, sin_b, cos_b, min ray per quadrant]
                read = self.wm.readout(z)
                min_ray = read[:, 3:].min(dim=1).values
                penalty += torch.relu(0.08 - min_ray) * 5.0   # wall proximity
            goal_dist = self.wm.readout(z)[:, 0]
            score = -goal_dist - penalty
            last_scores = score
            elite_idx = torch.topk(score, self.elites).indices
            elite = actions[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0) + 1e-3
            # keep the best candidate seen across ALL iterations, not just
            # whichever led the final iteration
            it_best = float(score[elite_idx[0]])
            if it_best > best_score:
                best_score = it_best
                i = int(elite_idx[0])
                best_first_z = zs[0][i:i + 1]
                best_traj = torch.stack([zh[i] for zh in zs])   # (H, latent)
        if not return_info:
            return best_first_z
        pred_readouts = self.wm.readout(best_traj)              # (H, readout)
        info = {
            "best_score": best_score,
            "score_mean": float(last_scores.mean()),
            "score_std": float(last_scores.std()),
            "score_min": float(last_scores.min()),
            "score_max": float(last_scores.max()),
            "chunk_dt": self.chunk_dt,
            "predicted_readouts": pred_readouts.numpy().tolist(),
        }
        return best_first_z, info
