"""Spiking sensory front-end: continuous observations -> sparse spike features.

A single layer of leaky integrate-and-fire (LIF) neurons driven by a learned
linear projection of the observation. The membrane state persists across
control steps, so the encoder integrates the sensor stream over time and
emits spikes when neurons cross threshold — an event-based code.

Continuous-time membrane: the leak is computed from the *physical* elapsed
time of each step, beta = exp(-dt / tau), with a learnable per-neuron time
constant tau. A fixed per-call decay (like snnTorch's Leaky) would make the
membrane dynamics depend on the sensor rate — at 60 Hz it would leak 4x
slower per second than at 15 Hz — which defeats the point of a
continuous-time agent. (Note: the hard threshold below is fine for CMA-ES,
which never differentiates through it; gradient-based training would need a
surrogate gradient here.)
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
        mem = beta * mem + self.fc(x)
        spk = (mem >= self.threshold).float()
        mem = mem - spk * self.threshold              # soft reset
        return spk, mem
