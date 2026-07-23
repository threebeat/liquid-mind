"""Deterministic LIF gate (Priority 1).

Equal physical input trajectories, partitioned into different observation
schedules, must produce consistent event counts and terminal membrane
states. Tolerances are DECLARED HERE, before the tests run:

    COUNT_TOL  — total event counts across partitions of the same physical
                 trajectory may differ by at most 1 event. A threshold
                 crossing that lands exactly on a boundary is assigned to
                 one side or the other by float rounding; the CHARGE
                 invariant below shows no event is lost or duplicated;
    CHARGE_TOL — the conserved quantity count*theta + membrane (soft reset
                 preserves it exactly) must agree across partitions within
                 1e-2 (float32 accumulation over ~240 calls);
    MEM_TOL    — where no crossing is near a boundary (subthreshold /
                 incommensurate cases), membranes agree within 1e-3;
    BRUTE_TOL  — the analytic count may differ from a 10 microsecond Euler
                 reference simulation by at most 2 events over 2 seconds.
"""
import math

import numpy as np
import pytest
import torch

from agent.spike_encoder import SpikeEncoder

COUNT_TOL = 1.0
CHARGE_TOL = 1e-2
MEM_TOL = 1e-3
BRUTE_TOL = 2.0

T_TOTAL = 2.0
FREQS = (15, 30, 60, 120)
RATES = (1, 5, 10, 14, 20, 40, 80)     # target spikes per second
TAUS = (0.05, 0.15, 0.5)


def make_encoder(tau: float, semantics: str = "event_count") -> SpikeEncoder:
    enc = SpikeEncoder(1, 1, tau_mem=tau, semantics=semantics)
    with torch.no_grad():
        enc.fc.weight.fill_(1.0)
        enc.fc.bias.fill_(0.0)
    return enc


def current_for_rate(rate: float, tau: float, theta: float = 1.0) -> float:
    """Constant current that produces `rate` spikes/s in steady state:
    period T = tau * ln(I / (I - theta))  =>  I = e^k / (e^k - 1), k=1/(r*tau)."""
    k = 1.0 / (rate * tau)
    return math.exp(k) / (math.exp(k) - 1.0) * theta


@torch.no_grad()
def run_partition(enc: SpikeEncoder, current: float, dts, m0: float = 0.0):
    mem = torch.tensor([[m0]])
    x = torch.tensor([[current]])
    total = 0.0
    per_call = []
    for dt in dts:
        out, mem = enc(x, mem, float(dt))
        total += float(out.sum())
        per_call.append(float(out.sum()))
    return total, float(mem), per_call


def partitions(total: float, seed: int = 0):
    parts = {f"{f}Hz": [1.0 / f] * int(round(total * f)) for f in FREQS}
    rng = np.random.default_rng(seed)
    irr = rng.uniform(0.2, 1.8, size=40)
    parts["irregular"] = list(irr / irr.sum() * total)
    return parts


# ------------------------------------------------- partition invariance


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("rate", RATES)
def test_partition_invariance(tau, rate):
    """The same held current over the same total time must produce the same
    event process at 15/30/60/120 Hz and irregular partitions — including
    rates ABOVE the slowest sampling frequency. Counts agree within 1 event
    (crossings landing exactly on a boundary are split by float rounding);
    the soft-reset invariant count*theta + membrane shows the event is
    neither lost nor duplicated."""
    current = current_for_rate(rate, tau)
    if current - 1.0 < 1e-6:
        pytest.skip("target rate needs a current closer to threshold than "
                    "float32 can represent (I - theta < 1e-6); the regime "
                    "is untestable at this precision, not incorrect")
    enc = make_encoder(tau)
    results = {name: run_partition(enc, current, dts)
               for name, dts in partitions(T_TOTAL).items()}
    counts = {n: r[0] for n, r in results.items()}
    charges = {n: r[0] * 1.0 + r[1] for n, r in results.items()}
    ref_c = counts["120Hz"]
    ref_q = charges["120Hz"]
    for name in counts:
        assert abs(counts[name] - ref_c) <= COUNT_TOL, \
            f"count mismatch at {name}: {counts[name]} vs {ref_c} " \
            f"(tau={tau}, rate={rate})"
        assert abs(charges[name] - ref_q) <= CHARGE_TOL, \
            f"charge invariant broken at {name}: {charges[name]} vs {ref_q}"


@pytest.mark.parametrize("tau", TAUS)
def test_subthreshold_no_events(tau):
    """Subthreshold current: zero events everywhere, exact membranes."""
    enc = make_encoder(tau)
    current = 0.5   # equilibrium 0.5 < threshold 1.0
    results = {name: run_partition(enc, current, dts)
               for name, dts in partitions(T_TOTAL).items()}
    expected_m = current * (1.0 - math.exp(-T_TOTAL / tau))
    for name, (count, mem, _) in results.items():
        assert count == 0.0, f"spurious event at {name}"
        assert abs(mem - expected_m) < 1e-4, \
            f"membrane at {name}: {mem} vs exact {expected_m}"


def test_nonzero_initial_membrane():
    """Partition invariance must hold from a nonzero initial state, and a
    state injected above threshold fires its excess resets at t=0."""
    tau = 0.15
    current = current_for_rate(10, tau)
    enc = make_encoder(tau)
    results = {name: run_partition(enc, current, dts, m0=0.6)
               for name, dts in partitions(T_TOTAL).items()}
    ref = results["120Hz"]
    for name, r in results.items():
        assert abs(r[0] - ref[0]) <= COUNT_TOL
        assert abs((r[0] + r[1]) - (ref[0] + ref[1])) <= CHARGE_TOL
    # injected m0 = 2.5 >= threshold: two instantaneous resets
    out, mem = enc(torch.tensor([[0.0]]), torch.tensor([[2.5]]), 1e-6)
    assert float(out) >= 2.0
    assert float(mem) < 1.0


def test_multiple_crossings_in_one_interval():
    """An 80 Hz event process sampled at 15 Hz must report >= 2 events in
    single intervals — multiplicity is preserved, not clipped to binary."""
    tau = 0.15
    current = current_for_rate(80, tau)
    enc = make_encoder(tau)
    _, _, per_call = run_partition(enc, current, [1.0 / 15] * 30)
    assert max(per_call) >= 2.0, \
        f"no multi-spike interval found: max per call {max(per_call)}"
    # and the total is far above the 15 Hz sampling ceiling of 30 events
    assert sum(per_call) > 100


@pytest.mark.parametrize("rate,tau", [(5, 0.15), (20, 0.15), (80, 0.05)])
def test_against_bruteforce_reference(rate, tau):
    """Independent check: a 10 microsecond Euler simulation of the LIF ODE
    with threshold detection must agree with the analytic event count."""
    current = current_for_rate(rate, tau)
    h = 1e-5
    m, count = 0.0, 0
    for _ in range(int(round(T_TOTAL / h))):
        m += h * (current - m) / tau
        if m >= 1.0:
            m -= 1.0
            count += 1
    enc = make_encoder(tau)
    analytic, _, _ = run_partition(enc, current, [1.0 / 30] * 60)
    assert abs(analytic - count) <= BRUTE_TOL, \
        f"analytic {analytic} vs brute-force {count}"


# ----------------------------------------------------- other semantics


def test_sampled_binary_is_observation_rate_limited():
    """The legacy mode emits at most one binary spike per call — kept as a
    named ablation, NOT as an exact event process."""
    tau = 0.15
    current = current_for_rate(80, tau)
    enc = make_encoder(tau, semantics="sampled_binary")
    total, _, per_call = run_partition(enc, current, [1.0 / 15] * 30)
    assert max(per_call) <= 1.0
    assert total <= 30    # capped at the sampling rate

def test_membrane_mode_matches_zoh():
    tau = 0.15
    enc = make_encoder(tau, semantics="membrane")
    out, mem = enc(torch.tensor([[0.8]]), torch.tensor([[0.2]]), 0.1)
    beta = math.exp(-0.1 / tau)
    expected = beta * 0.2 + (1 - beta) * 0.8
    assert abs(float(mem) - expected) < 1e-6
    assert abs(float(out) - expected) < 1e-6


def test_rate_mode_is_smooth_and_eventless():
    enc = make_encoder(0.15, semantics="rate")
    total, _, per_call = run_partition(enc, 2.0, [1.0 / 30] * 60)
    assert all(c >= 0 for c in per_call)
    # softplus output: no discrete jumps to compare, just sanity
    assert total > 0
