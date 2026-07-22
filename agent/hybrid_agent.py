"""The full two-speed agent.

Fast loop (every control step, 15-60 Hz): observation -> SNN spikes ->
liquid CfC policy (dt-aware) -> wheel commands.

Slow loop (every `chunk_steps` control steps, ~1-2 Hz): encode the current
observation into latent space, run the CEM planner through the JEPA world
model, and hand the resulting subgoal latent to the fast loop.

In "reactive" mode the subgoal is zeros and only the fast loop runs.
"""
import os

import numpy as np
import torch

from agent.lnn_policy import LiquidPolicy
from agent.planner import CEMPlanner
from agent.spike_encoder import SpikeEncoder
from agent.world_model import WorldModel
from environment.nav_env import ACT_DIM, OBS_DIM


class HybridAgent:
    def __init__(self, config: dict, mode: str = "reactive",
                 world_model: WorldModel | None = None):
        a = config["agent"]
        self.mode = mode
        self.latent_dim = int(a["latent_dim"])
        self.snn = SpikeEncoder(OBS_DIM, int(a["snn_neurons"]),
                                float(a["snn_beta"]))
        in_dim = int(a["snn_neurons"]) + OBS_DIM + self.latent_dim
        self.policy = LiquidPolicy(in_dim, int(a["lnn_units"]), ACT_DIM)
        self.snn.eval()
        self.policy.eval()

        self.wm = world_model
        self.planner = None
        if mode == "hierarchical":
            assert world_model is not None, "hierarchical mode needs a world model"
            pcfg = config["planner"]
            chunk = int(config["world_model"]["chunk_steps"])
            nominal_dt = (int(config["env"]["control_substeps"])
                          / int(config["env"]["physics_hz"]))
            self.planner = CEMPlanner(
                world_model, horizon=int(pcfg["horizon"]),
                population=int(pcfg["population"]), elites=int(pcfg["elites"]),
                iterations=int(pcfg["iterations"]), chunk_dt=chunk * nominal_dt)
        self.chunk_steps = int(config["world_model"]["chunk_steps"])
        self.goal_tau = float(config["planner"].get("goal_tau", 0.4))
        self.reset()

    # ---------------------------------------------------------------- state

    def reset(self):
        self._mem = self.snn.init_state()
        self._hx = self.policy.init_state()
        self._subgoal = torch.zeros(1, self.latent_dim)         # g(t): smoothed
        self._subgoal_target = torch.zeros(1, self.latent_dim)  # z_target: planner
        self._t = 0

    def parameters(self):
        """Evolvable parameters (SNN + policy), for CMA-ES."""
        return list(self.snn.parameters()) + list(self.policy.parameters())

    # ------------------------------------------------------------------ act

    @torch.no_grad()
    def act(self, obs: np.ndarray, dt: float) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if self.mode == "hierarchical":
            if self._t % self.chunk_steps == 0:
                # Planner proposes a new target; z_target changes instantly...
                self._subgoal_target = self.planner.plan(self.wm.encode(x))
            # ...but g(t) glides toward it: g += (dt/tau)(z_target - g).
            alpha = min(1.0, dt / self.goal_tau)
            self._subgoal = self._subgoal + alpha * (self._subgoal_target
                                                     - self._subgoal)
        spk, self._mem = self.snn(x, self._mem)
        pol_in = torch.cat([spk, x, self._subgoal], dim=1)
        action, self._hx = self.policy(pol_in, self._hx, dt)
        self._t += 1
        return action.squeeze(0).numpy()

    def latent_dist_to_subgoal(self, obs: np.ndarray) -> float:
        """Distance between current latent and active subgoal (phase-4 reward)."""
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = self.wm.encode(x)
        return float(torch.norm(z - self._subgoal))

    # ------------------------------------------------------------- save/load

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"snn": self.snn.state_dict(),
                    "policy": self.policy.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.snn.load_state_dict(ckpt["snn"])
        self.policy.load_state_dict(ckpt["policy"])
