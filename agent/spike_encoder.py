"""Spiking sensory front-end: continuous observations -> sparse spike features.

A single layer of leaky integrate-and-fire (LIF) neurons (snnTorch) driven by a
learned linear projection of the observation. The membrane state persists
across control steps, so the encoder is *stateful*: it integrates the sensor
stream over time and emits spikes when its neurons cross threshold — an
event-based code, like biological skin.
"""
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


class SpikeEncoder(nn.Module):
    def __init__(self, in_dim: int, n_neurons: int = 32, beta: float = 0.9):
        super().__init__()
        self.n_neurons = n_neurons
        self.fc = nn.Linear(in_dim, n_neurons)
        self.lif = snn.Leaky(beta=beta, threshold=1.0,
                             spike_grad=surrogate.fast_sigmoid())

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.n_neurons)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        """x: (batch, in_dim) -> spikes (batch, n_neurons), new membrane state."""
        cur = self.fc(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem
