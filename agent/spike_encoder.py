"""Spiking sensory front-end: continuous observations -> event features.

A single layer of leaky integrate-and-fire (LIF) neurons driven by a learned
linear projection of the observation. The membrane state persists across
control steps, so the encoder integrates the sensor stream over physical
time and emits threshold events — an event-based code.

Membrane ODE (per neuron, learnable time constant tau):
    tau * dm/dt = -m + I
Under a current I held constant across an interval the subthreshold solution
is exact:
    m(t) = I + (m0 - I) * exp(-t / tau)

Semantic modes (config: agent.snn_semantics) — choose ONE and it is recorded
in checkpoint metadata:

  "event_count" (default) — analytic event-count LIF.
      Assumes the projected current is constant over the elapsed interval
      (the caller decides WHICH observation is held; see HybridAgent's
      timing_convention). Threshold-crossing times are computed in closed
      form, the soft reset (subtract threshold, i.e. m -> 0 at an exact
      crossing) is applied, and MULTIPLE crossings per interval are counted.
      Returns the spike COUNT per neuron, preserving multiplicity.
      Exactness statement:
        - subthreshold propagation: exact;
        - threshold-crossing times: exact under the held-current assumption;
        - multiple crossings per interval: supported, counted in closed form;
        - returned representation: per-neuron event count (multiplicity kept,
          individual timestamps not returned).
      This makes the *event process* independent of how a fixed physical
      current trajectory is partitioned into observation intervals (up to
      floating-point at partition boundaries) — a property the deterministic
      tests in tests/test_lif_semantics.py check across 15/30/60/120 Hz and
      irregular partitions. It is NOT a claim of sampling-rate independence
      for time-varying inputs: the held current itself changes only when a
      new observation arrives.

  "sampled_binary" (legacy ablation) — the pre-2026 behavior, kept verbatim:
      one ZOH membrane update per call, ONE threshold test, at most one
      binary spike per neuron per observation, soft reset. Subthreshold
      propagation is exact but the event process is observation-rate
      limited and multiplicity is lost.

  "membrane" (non-spiking control) — exact ZOH membrane update, the raw
      membrane value is exposed as the feature. No threshold, no reset.

  "rate" (non-spiking control) — exact ZOH membrane update, feature is the
      smooth activation softplus(m - threshold). No threshold events.

(The hard threshold is fine for CMA-ES, which never differentiates through
it; gradient-based training would need a surrogate gradient here.)
"""
import math

import torch
import torch.nn as nn

SEMANTIC_MODES = ("event_count", "sampled_binary", "membrane", "rate")


class SpikeEncoder(nn.Module):
    def __init__(self, in_dim: int, n_neurons: int = 32,
                 tau_mem: float = 0.15, threshold: float = 1.0,
                 semantics: str = "event_count"):
        super().__init__()
        if semantics not in SEMANTIC_MODES:
            raise ValueError(f"unknown snn_semantics {semantics!r}; "
                             f"expected one of {SEMANTIC_MODES}")
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.semantics = semantics
        self.fc = nn.Linear(in_dim, n_neurons)
        # per-neuron membrane time constant (seconds), log-parameterized
        self.log_tau = nn.Parameter(
            torch.full((n_neurons,), math.log(tau_mem)))

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.n_neurons)

    # ------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor, mem: torch.Tensor, dt: float):
        """Propagate the membrane across `dt` seconds under the current
        projected from `x`, held constant. Returns (features, new membrane).

        Feature meaning depends on self.semantics (see module docstring):
        event counts, binary spikes, membrane values, or a smooth rate.
        """
        I = self.fc(x)                                     # held current
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        if self.semantics == "event_count":
            return self._event_count(I, mem, float(dt), tau)
        beta = torch.exp(-float(dt) / tau)                 # exact ZOH leak
        mem = beta * mem + (1.0 - beta) * I
        if self.semantics == "sampled_binary":
            spk = (mem >= self.threshold).float()
            mem = mem - spk * self.threshold               # soft reset
            return spk, mem
        if self.semantics == "membrane":
            return mem, mem
        # "rate": smooth activation of the exact membrane, no events
        return torch.nn.functional.softplus(mem - self.threshold), mem

    def _event_count(self, I: torch.Tensor, m0: torch.Tensor, dt: float,
                     tau: torch.Tensor):
        """Closed-form event counting under a held current.

        For I > theta and m0 < theta the first crossing is at
            t1 = tau * ln((I - m0) / (I - theta)),
        the soft reset at an exact crossing leaves m = 0, and subsequent
        inter-spike intervals are
            T = tau * ln(I / (I - theta)).
        The residual membrane is propagated through the remaining fraction
        of the interval. If m0 >= theta at entry (possible only for injected
        initial states), the excess resets fire instantaneously at t = 0.
        """
        theta = self.threshold
        eps = 1e-12
        # instantaneous resets for injected states already above threshold
        above = (m0 >= theta).float()
        n0 = above * torch.floor(m0 / theta)
        m0 = m0 - n0 * theta

        if dt <= 0.0:
            return n0, m0

        beta = torch.exp(-dt / tau)
        m_nospike = I + (m0 - I) * beta                    # exact subthreshold

        can_spike = I > theta + 1e-9                       # crossing possible
        # first crossing time (arguments clamped so masked-out lanes stay finite)
        t1 = tau * torch.log((I - m0).clamp(min=eps) / (I - theta).clamp(min=eps))
        crossed = can_spike & (t1 <= dt)
        # inter-spike period after a reset-to-zero
        T = tau * torch.log((I / (I - theta).clamp(min=eps)).clamp(min=1.0 + eps))
        # additional full periods inside the interval (tiny relative slack so
        # a crossing landing exactly on the interval end is counted once);
        # masked lanes are zeroed BEFORE arithmetic so no inf/NaN leaks in
        extra = torch.floor(((dt - t1).clamp(min=0.0) / T) * (1.0 + 1e-9))
        extra = torch.where(crossed, extra, torch.zeros_like(extra))
        t_last = t1 + extra * T
        m_after = I * (1.0 - torch.exp(-(dt - t_last).clamp(min=0.0) / tau))

        crossed_f = crossed.to(I.dtype)
        count = n0 + crossed_f * (1.0 + extra)
        mem = torch.where(crossed, m_after, m_nospike)
        return count, mem
