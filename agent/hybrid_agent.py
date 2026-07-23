"""The full two-speed agent.

Fast loop (every control event, 15-60 Hz): observation -> SNN events ->
input adapter -> liquid CfC policy (dt-aware) -> wheel commands.

Slow loop (every `planner.period_seconds` of PHYSICAL time, ~2 Hz): encode
the current observation into latent space, run the CEM planner through the
JEPA world model, and hand the resulting subgoal latent to the fast loop.
The planner schedule accumulates elapsed physical time and carries the
residual across boundaries, so its frequency in seconds does not drift under
irregular sensing.

In "reactive" mode the subgoal is zeros and only the fast loop runs.

Event-timing semantics (config: agent.timing_convention)
--------------------------------------------------------
The agent exposes explicit operations:

    advance(dt)   propagate internal state across dt seconds of physical
                  time under the HELD sensory input;
    observe(obs)  assimilate a newly captured measurement (held input is
                  replaced; no time passes);
    act()         emit the next action.

"causal" (default): at each event the SNN membrane is propagated across the
elapsed interval under the PREVIOUS measurement's current (physical
zero-order hold — during [t_{k-1}, t_k] only information available at
t_{k-1} exists), then the new measurement is assimilated and the action
issued. On reset, the first measurement is assimilated at elapsed time
zero — no fabricated nominal interval.

"irnn" (legacy): the NEW measurement's current is integrated across the
already-elapsed interval. This is the standard convention for
irregularly-sampled recurrent models (GRU-D, CfC) but it is not an exact
physical ZOH; it is kept as a clearly named mode for comparison.

In BOTH modes the CfC cell itself follows the irregular-RNN update
convention: the freshly assimilated features enter together with the
elapsed timespan (that is how the closed-form CfC cell is defined). The
timing_convention flag governs the sensory ZOH — which measurement the
spiking encoder integrates across the interval.

Confound controls
-----------------
agent.mask_direct_dt (default true) replaces the raw dt observation channel
(obs[21]) with its nominal value before ANY network sees it, so nominal-time
ablation arms cannot condition on the true interval through the observation
bypass. agent.use_input_adapter projects every policy_input mode through a
shared 54-d sensory bus (spikes || obs, absent modalities zero-filled) and
an identical linear adapter before the CfC — fixed adapter shapes / shared
downstream architecture, not total-parameter capacity matching. obs_only
still excludes SNN parameters from evolution.

Physical-time planner scheduling
--------------------------------
The planner accumulates elapsed physical seconds and uses modulo to preserve
residual time after a boundary. If one observation gap crosses multiple
planning periods, the agent replans once at the first event following the
missed boundaries and discards the count of missed opportunities through
modulo — it does not retroactively replan at every elapsed boundary.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from agent.lnn_policy import LiquidPolicy
from agent.planner import CEMPlanner
from agent.spike_encoder import SpikeEncoder
from agent.world_model import WorldModel
from environment.nav_env import ACT_DIM, DT_OBS_INDEX, OBS_DIM
from provenance import (SEMANTICS_VERSION, gather_provenance, load_checkpoint,
                        save_checkpoint)

SUBGOAL_SOURCES = ("planner", "zero", "random", "shuffled", "heuristic")
# Shared sensory bus: [spikes (n_neurons) | obs (OBS_DIM)]; absent slots zero.
SENSORY_BUS_SPIKES = 32  # must match default snn_neurons; validated at init
SENSORY_BUS_DIM = SENSORY_BUS_SPIKES + OBS_DIM  # 54


class HybridAgent:
    def __init__(self, config: dict, mode: str = "reactive",
                 world_model: WorldModel | None = None):
        a = config["agent"]
        self.config = config
        self.mode = mode
        self.latent_dim = int(a["latent_dim"])
        self.snn_semantics = a.get("snn_semantics", "event_count")
        self.snn = SpikeEncoder(OBS_DIM, int(a["snn_neurons"]),
                                float(a.get("snn_tau_mem", 0.15)),
                                semantics=self.snn_semantics)
        # what the policy sees (ablation switch): spikes, raw obs, or both
        self.policy_input = a.get("policy_input", "spikes_obs")
        # discrete-time control arms: feed nominal dt instead of the true one
        self.snn_time_aware = bool(a.get("snn_time_aware", True))
        self.cfc_time_aware = bool(a.get("cfc_time_aware", True))
        # confound control: hide the raw dt observation channel
        self.mask_direct_dt = bool(a.get("mask_direct_dt", True))
        self.timing_convention = a.get("timing_convention", "causal")
        assert self.timing_convention in ("causal", "irnn")
        self.nominal_dt = (int(config["env"]["control_substeps"])
                           / int(config["env"]["physics_hz"]))
        # Shared downstream architecture: every mode fills a fixed 54-d bus
        # (spikes | obs) with zeros for absent modalities, then an identical
        # Linear(54, adapter_dim). This equalizes adapter tensor shapes, not
        # total evolvable parameter count (obs_only still drops SNN params).
        self.use_input_adapter = bool(a.get("use_input_adapter", True))
        self.adapter_dim = int(a.get("adapter_dim", 32))
        self.sensory_bus_dim = SENSORY_BUS_DIM
        n_spikes = int(a["snn_neurons"])
        if self.use_input_adapter and n_spikes != SENSORY_BUS_SPIKES:
            raise ValueError(
                f"use_input_adapter requires snn_neurons={SENSORY_BUS_SPIKES} "
                f"(got {n_spikes}) so every mode shares a {SENSORY_BUS_DIM}-d "
                f"sensory bus")
        if self.use_input_adapter:
            self.adapter = nn.Linear(SENSORY_BUS_DIM, self.adapter_dim)
            in_dim = self.adapter_dim + self.latent_dim
        else:
            self.adapter = None
            feat = {"spikes_obs": n_spikes + OBS_DIM,
                    "spikes_only": n_spikes,
                    "obs_only": OBS_DIM}[self.policy_input]
            in_dim = feat + self.latent_dim
        self.policy = LiquidPolicy(in_dim, int(a["lnn_units"]), ACT_DIM)
        self.snn.eval()
        self.policy.eval()
        if self.adapter is not None:
            self.adapter.eval()

        self.wm = world_model
        self.planner = None
        # planner period in PHYSICAL seconds (falls back to the legacy
        # decision-count schedule converted at the nominal rate)
        pcfg = config["planner"]
        chunk = int(config["world_model"]["chunk_steps"])
        self.plan_period = float(pcfg.get("period_seconds",
                                          chunk * self.nominal_dt))
        if mode == "hierarchical":
            assert world_model is not None, "hierarchical mode needs a world model"
            self.planner = CEMPlanner(
                world_model, horizon=int(pcfg["horizon"]),
                population=int(pcfg["population"]), elites=int(pcfg["elites"]),
                iterations=int(pcfg["iterations"]), chunk_dt=self.plan_period)
        self.goal_tau = float(pcfg.get("goal_tau", 0.4))

        # attribution controls (Priority 6): where subgoals come from
        self.subgoal_source = "planner"
        self.scripted_subgoals: list | None = None   # for "shuffled"
        self.random_subgoal_norm = 1.0               # for "random"
        self._subgoal_rng = np.random.default_rng(0)
        # on-policy world-model validation (Priority 7)
        self.log_planner = False
        self.planner_log: list[dict] = []
        self.reset()

    # ---------------------------------------------------------------- state

    def reset(self):
        self._mem = self.snn.init_state()
        self._hx = self.policy.init_state()
        self._subgoal = torch.zeros(1, self.latent_dim)         # g(t): smoothed
        self._subgoal_target = torch.zeros(1, self.latent_dim)  # z_target: planner
        self._held_x = None            # last assimilated (masked) observation
        self._pending_feats = torch.zeros(1, self.snn.n_neurons)
        self._time_since_plan = None   # None -> plan at the first event
        self._plan_count = 0
        self._last_pred_readouts = None
        self._sim_time = 0.0
        self._t = 0
        self.planner_log = []

    def parameters(self):
        """Evolvable parameters for CMA-ES. In obs_only mode the SNN cannot
        affect the action, so its parameters are excluded rather than letting
        the optimizer waste dimensions on them. The input adapter (when
        enabled) is part of the policy and is evolved."""
        ps = []
        if self.policy_input != "obs_only":
            ps += list(self.snn.parameters())
        if self.adapter is not None:
            ps += list(self.adapter.parameters())
        return ps + list(self.policy.parameters())

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def variant_tag(self) -> str:
        """Short tag describing non-default settings, used to keep
        checkpoints from different configurations separate. Distinguishes
        policy input mode, SNN/CfC physical- vs nominal-time propagation,
        direct-dt visibility, SNN semantics and the timing convention."""
        parts = []
        if self.policy_input != "spikes_obs":
            parts.append(self.policy_input)
        if not self.snn_time_aware:
            parts.append("nomsnn")
        if not self.cfc_time_aware:
            parts.append("nomcfc")
        if not self.mask_direct_dt:
            parts.append("dtvis")
        if self.snn_semantics != "event_count":
            parts.append(self.snn_semantics)
        if self.timing_convention != "causal":
            parts.append(self.timing_convention)
        if not self.use_input_adapter:
            parts.append("noadapter")
        return ("_" + "_".join(parts)) if parts else ""

    def compat(self) -> dict:
        """Compatibility block validated when a checkpoint is loaded."""
        return {
            "semantics_version": SEMANTICS_VERSION,
            "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "policy_input": self.policy_input,
            "snn_time_aware": self.snn_time_aware,
            "cfc_time_aware": self.cfc_time_aware,
            "snn_semantics": self.snn_semantics,
            "mask_direct_dt": self.mask_direct_dt,
            "timing_convention": self.timing_convention,
            "use_input_adapter": self.use_input_adapter,
            "adapter_dim": self.adapter_dim if self.use_input_adapter else None,
            "sensory_bus_dim": (self.sensory_bus_dim
                               if self.use_input_adapter else None),
            "snn_neurons": int(self.snn.n_neurons),
            "lnn_units": int(self.policy.units),
            "latent_dim": self.latent_dim,
        }

    # -------------------------------------------------- event-timing API

    def observe(self, obs: np.ndarray):
        """Assimilate a newly captured measurement. No time passes here:
        the held sensory input is replaced at the current instant."""
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if self.mask_direct_dt:
            x = x.clone()
            x[0, DT_OBS_INDEX] = 1.0     # nominal value: true dt hidden
        self._held_x = x

    def advance(self, dt: float):
        """Propagate internal SNN state across dt seconds of physical time
        under the currently held input current."""
        dt = float(dt)
        self._sim_time += dt
        if self._time_since_plan is not None:
            self._time_since_plan += dt
        if self._held_x is None or dt <= 0.0:
            self._pending_feats = torch.zeros(1, self.snn.n_neurons)
            return
        dt_snn = dt if self.snn_time_aware else self.nominal_dt
        feats, self._mem = self.snn(self._held_x, self._mem, dt_snn)
        self._pending_feats = feats

    @torch.no_grad()
    def act(self, obs: np.ndarray, dt: float) -> np.ndarray:
        """One control event: advance / observe / act in the order given by
        the configured timing convention. `dt` is the time elapsed since the
        previous measurement (ignored and treated as zero on the first event
        after reset in causal mode: initial assimilation at elapsed time 0)."""
        if self.timing_convention == "causal":
            first = self._held_x is None
            dt_eff = 0.0 if first else float(dt)
            self.advance(dt_eff)      # propagate under the PREVIOUS input
            self.observe(obs)         # then assimilate the new measurement
        else:                         # legacy irregular-RNN convention
            dt_eff = float(dt)
            self.observe(obs)         # new measurement is held across ...
            self.advance(dt_eff)      # ... the already-elapsed interval
        return self._act(dt_eff)

    def _sensory_bus(self, spikes: torch.Tensor, obs: torch.Tensor
                     ) -> torch.Tensor:
        """Fill the shared [spikes | obs] bus; zero absent modalities."""
        B = obs.shape[0]
        bus = torch.zeros(B, SENSORY_BUS_DIM, dtype=obs.dtype, device=obs.device)
        if self.policy_input in ("spikes_obs", "spikes_only"):
            bus[:, :spikes.shape[1]] = spikes
        if self.policy_input in ("spikes_obs", "obs_only"):
            bus[:, SENSORY_BUS_SPIKES:SENSORY_BUS_SPIKES + OBS_DIM] = obs
        return bus

    def _act(self, dt: float) -> np.ndarray:
        x = self._held_x
        if self.mode == "hierarchical":
            self._update_subgoal(x, dt)
        if self.adapter is not None:
            feats = torch.tanh(self.adapter(
                self._sensory_bus(self._pending_feats, x)))
        elif self.policy_input == "spikes_obs":
            feats = torch.cat([self._pending_feats, x], dim=1)
        elif self.policy_input == "spikes_only":
            feats = self._pending_feats
        else:
            feats = x
        pol_in = torch.cat([feats, self._subgoal], dim=1)
        dt_cfc = dt if self.cfc_time_aware else self.nominal_dt
        action, self._hx = self.policy(pol_in, self._hx, dt_cfc)
        self._t += 1
        return action.squeeze(0).numpy()

    # ------------------------------------------------------------ slow loop

    def _update_subgoal(self, x: torch.Tensor, dt: float):
        """Physical-time replanning: replan every plan_period SECONDS of
        elapsed time (residual carried over so the clock never drifts).

        If a single observation gap crosses one or more planning periods,
        replan once at the first subsequent event; missed interior boundaries
        are discarded via modulo (no retroactive multi-replan)."""
        if self._time_since_plan is None:          # first event after reset
            replan = True
            self._time_since_plan = 0.0
        else:
            replan = self._time_since_plan >= self.plan_period
            if replan:
                self._time_since_plan = self._time_since_plan % self.plan_period
        if replan:
            self._subgoal_target = self._next_subgoal(x)
            self._plan_count += 1
        # g(t) glides toward the target: g += (dt/tau)(z_target - g)
        alpha = min(1.0, dt / self.goal_tau) if dt > 0 else 0.0
        self._subgoal = self._subgoal + alpha * (self._subgoal_target
                                                 - self._subgoal)

    def _next_subgoal(self, x: torch.Tensor) -> torch.Tensor:
        src = self.subgoal_source
        if src == "zero":
            return torch.zeros(1, self.latent_dim)
        if src == "random":
            g = self._subgoal_rng.standard_normal(self.latent_dim)
            g = g / (np.linalg.norm(g) + 1e-9) * self.random_subgoal_norm
            return torch.from_numpy(g.astype(np.float32)).unsqueeze(0)
        if src == "shuffled":
            seq = self.scripted_subgoals or []
            idx = min(self._plan_count, len(seq) - 1)
            if idx < 0:
                return torch.zeros(1, self.latent_dim)
            return torch.as_tensor(seq[idx], dtype=torch.float32).reshape(
                1, self.latent_dim)
        if src == "heuristic":
            return self._heuristic_subgoal(x)
        # "planner"
        z_now = self.wm.encode(x)
        if not self.log_planner:
            return self.planner.plan(z_now)
        realized = self.wm.readout(z_now).squeeze(0).numpy().tolist()
        entry = {"t": self._t, "sim_time": self._sim_time,
                 "realized_readout": realized,
                 "planner_seed": self.planner.last_seed}
        if self._last_pred_readouts is not None:
            # prediction made one macro-step ago for "now" (first chunk)
            entry["prev_predicted_readout"] = self._last_pred_readouts[0]
        sub, info = self.planner.plan(z_now, return_info=True)
        entry.update({k: info[k] for k in ("best_score", "score_mean",
                                           "score_std", "score_min",
                                           "score_max", "planner_seed")})
        entry["chosen_subgoal"] = sub.squeeze(0).numpy().tolist()
        entry["predicted_readouts"] = info["predicted_readouts"]
        self._last_pred_readouts = info["predicted_readouts"]
        self.planner_log.append(entry)
        return sub

    def _heuristic_subgoal(self, x: torch.Tensor) -> torch.Tensor:
        """Simple geometric lidar-waypoint heuristic (attribution control):
        steer toward the ray direction with the best trade-off between
        clearance and alignment with the goal bearing, encoded as a latent
        via the world model (pseudo-observation with the waypoint bearing)."""
        obs = x.squeeze(0).numpy().copy()
        rays = obs[:16]
        bearing = float(np.arctan2(obs[17], obs[18]))
        n = len(rays)
        angles = 2.0 * np.pi * np.arange(n) / n            # robot frame
        diff = np.angle(np.exp(1j * (angles - bearing)))   # wrapped
        score = rays - 0.5 * np.abs(diff) / np.pi          # clearance - misalign
        k = int(np.argmax(score))
        wp_bearing = angles[k]
        pseudo = obs.copy()
        pseudo[16] = min(obs[16], 0.15)                    # waypoint <= 1.5 m
        pseudo[17] = np.sin(wp_bearing)
        pseudo[18] = np.cos(wp_bearing)
        with torch.no_grad():
            z = self.wm.encode(torch.from_numpy(
                pseudo.astype(np.float32)).unsqueeze(0))
        return z

    def latent_dist_to_subgoal(self, obs: np.ndarray) -> float:
        """Distance between current latent and active subgoal (phase-4 reward)."""
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if self.mask_direct_dt:
            x = x.clone()
            x[0, DT_OBS_INDEX] = 1.0
        with torch.no_grad():
            z = self.wm.encode(x)
        return float(torch.norm(z - self._subgoal))

    # ------------------------------------------------------------- save/load

    def _state(self) -> dict:
        state = {"snn": self.snn.state_dict(),
                 "policy": self.policy.state_dict()}
        if self.adapter is not None:
            state["adapter"] = self.adapter.state_dict()
        return state

    def save(self, path: str, meta: dict | None = None, force: bool = False) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if meta is None:
            meta = gather_provenance(self.config, experiment_name="adhoc",
                                     variant=self.variant_tag())
        meta = dict(meta)
        meta.setdefault("mode", self.mode)
        meta.setdefault("parameter_count", self.n_parameters())
        return save_checkpoint(path, self._state(), meta, self.compat(),
                               force=force)

    def load(self, path: str, allow_legacy: bool = False) -> dict:
        state, meta = load_checkpoint(path, expected_compat=self.compat(),
                                      allow_legacy=allow_legacy)
        if meta.get("legacy", False):
            if self.adapter is not None:
                raise ValueError(
                    f"legacy checkpoint {path} predates the input adapter; "
                    f"set agent.use_input_adapter: false (and the matching "
                    f"legacy flags: snn_semantics: sampled_binary, "
                    f"timing_convention: irnn, mask_direct_dt: false) to "
                    f"evaluate it.")
        self.snn.load_state_dict(state["snn"])
        self.policy.load_state_dict(state["policy"])
        if self.adapter is not None:
            self.adapter.load_state_dict(state["adapter"])
        return meta
