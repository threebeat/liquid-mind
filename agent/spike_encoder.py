"""Spiking sensory front-end: continuous observations -> sparse spike features.

A single layer of leaky integrate-and-fire (LIF) neurons driven by a learned
linear projection of the observation. The membrane state persists across
control steps, so the encoder integrates the sensor stream over time and
emits spikes when neurons cross threshold — an event-based code.

Continuous-time membrane: exact zero-order-hold solution of the LIF ODE
    tau * dm/dt = -m + I
over the elapsed interval dt:
    m' = beta * m + (1 - beta) * I,   beta = exp(-dt / tau)
with a learnable per-neuron time constant tau.

BOTH factors matter. A fixed per-call decay (snnTorch's Leaky) makes the
leak depend on the sensor rate; injecting the full current once per call
(without the 1-beta factor) makes the equilibrium level scale like tau/dt,
so a 60 Hz stream would spike ~4x as often as a 15 Hz stream for the same
physical signal. With the ZOH form, spike rate per physical second is
sampling-rate independent.

Timing semantics (causal): dt is the time elapsed since the PREVIOUS
observation — the interval the state must be integrated across to reach
"now". The agent is never told how long its action will be held; it can't
know that in an online system. Applying the newly received observation over
the just-elapsed interval (rather than holding the previous one) is the
standard convention for irregularly-sampled recurrent models (GRU-D, CfC).

(The hard threshold is fine for CMA-ES, which never differentiates through
it; gradient-based training would need a surrogate gradient here.)
"""
import math

import torch
import torch.nn as nn


class SpikeEncoder(nn.Module):
    def __init__(self, in_dim: int, n_neurons: int = 32,
                 tau_mem: float = 0.15, threshold: float = 1.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.fc = nn.Linear(in_dim, n_neurons)
        # per-neuron membrane time constant (seconds), log-parameterized
        self.log_tau = nn.Parameter(
            torch.full((n_neurons,), math.log(tau_mem)))

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.n_neurons)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, dt: float):
        """x: (batch, in_dim) -> spikes (batch, n_neurons), new membrane."""
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        beta = torch.exp(-float(dt) / tau)            # physical-time leak
        mem = beta * mem + (1.0 - beta) * self.fc(x)  # exact ZOH integration
        spk = (mem >= self.threshold).float()
        mem = mem - spk * self.threshold              # soft reset
        return spk, mem
