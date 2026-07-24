"""Phase 1 deterministic tests: frozen reservoir specialists."""
import numpy as np
import torch

from phase1.esn import (ReservoirConfig, ReservoirSpecialist,
                        make_body_specialist, make_lidar_specialist,
                        reservoir_health)


def _cfg(seed=0, **kw):
    base = dict(n_units=32, n_inputs=5, seed=seed)
    base.update(kw)
    return ReservoirConfig(**base)


def test_fixed_reservoir_init_per_seed():
    a = ReservoirSpecialist(_cfg(seed=7))
    b = ReservoirSpecialist(_cfg(seed=7))
    assert torch.equal(a.w_in, b.w_in)
    assert torch.equal(a.w_rec, b.w_rec)
    assert torch.equal(a.b, b.b)
    assert torch.equal(a.readout.weight, b.readout.weight)


def test_different_seeds_give_different_reservoirs():
    a = ReservoirSpecialist(_cfg(seed=1))
    b = ReservoirSpecialist(_cfg(seed=2))
    assert not torch.equal(a.w_in, b.w_in)
    assert not torch.equal(a.w_rec, b.w_rec)


def test_spectral_radius_rescaled():
    sp = ReservoirSpecialist(_cfg(seed=3, spectral_radius=0.8))
    eig = np.max(np.abs(np.linalg.eigvals(sp.w_rec.numpy())))
    assert abs(eig - 0.8) < 1e-4


def test_reset_restores_initial_state():
    sp = ReservoirSpecialist(_cfg(seed=0))
    r = sp.reset(batch=4)
    assert torch.equal(r, torch.zeros(4, 32))
    x = torch.randn(4, 8, 5)
    dt = torch.full((4, 8), 1.0 / 30.0)
    _, _, r_end = sp(x, dt, r=sp.state)
    assert not torch.allclose(r_end, torch.zeros_like(r_end))
    r2 = sp.reset(batch=4)
    assert torch.equal(r2, torch.zeros(4, 32))


def test_identical_input_identical_messages():
    sp = ReservoirSpecialist(_cfg(seed=5))
    x = torch.randn(3, 10, 5)
    dt = torch.full((3, 10), 1.0 / 30.0)
    m1, s1, _ = sp(x, dt)
    m2, s2, _ = sp(x, dt)
    assert torch.equal(m1, m2)
    assert torch.equal(s1, s2)


def test_zero_trainable_reservoir_weights():
    sp = make_lidar_specialist(seed=0)
    trainables = {n for n, p in sp.named_parameters() if p.requires_grad}
    assert trainables == {"readout.weight", "readout.bias"}
    for buf_name in ("w_in", "w_rec", "b", "r0"):
        assert not getattr(sp, buf_name).requires_grad
    body = make_body_specialist(seed=0)
    assert sum(p.numel() for p in body.parameters() if p.requires_grad) \
        == (64 + 5) * 16 + 16


def test_gradients_flow_only_through_readout():
    sp = ReservoirSpecialist(_cfg(seed=4))
    w_in0, w_rec0 = sp.w_in.clone(), sp.w_rec.clone()
    x = torch.randn(2, 6, 5)
    dt = torch.full((2, 6), 1.0 / 30.0)
    msgs, _, _ = sp(x, dt)
    loss = msgs.pow(2).mean()
    loss.backward()
    assert sp.readout.weight.grad is not None
    assert sp.readout.weight.grad.abs().sum() > 0
    assert sp.w_in.grad is None and sp.w_rec.grad is None
    opt = torch.optim.Adam(sp.trainable_parameters(), lr=1e-2)
    opt.step()
    assert torch.equal(sp.w_in, w_in0)
    assert torch.equal(sp.w_rec, w_rec0)


def test_message_shape_and_input_skip():
    sp = make_lidar_specialist(seed=0)
    x = torch.randn(2, 15, 17)
    dt = torch.full((2, 15), 1.0 / 30.0)
    msgs, states, r_end = sp(x, dt)
    assert msgs.shape == (2, 15, 16)
    assert states.shape == (2, 15, 128)
    # input skip: readout consumes [r; x]
    assert sp.readout.in_features == 128 + 17


def test_nominal_vs_physical_leak_differ_under_irregular_dt():
    nom = ReservoirSpecialist(_cfg(seed=6, leak_mode="nominal"))
    phys = ReservoirSpecialist(_cfg(seed=6, leak_mode="physical"))
    x = torch.randn(1, 8, 5)
    dt = torch.rand(1, 8) * 0.04 + 0.017      # irregular 17..57 ms
    _, s_n, _ = nom(x, dt)
    _, s_p, _ = phys(x, dt)
    assert not torch.allclose(s_n, s_p)
    # at exactly the nominal interval, physical tau reproduces alpha_0
    dt_nom = torch.full((1, 8), nom.cfg.nominal_dt)
    _, s_n2, _ = nom(x, dt_nom)
    _, s_p2, _ = phys(x, dt_nom)
    assert torch.allclose(s_n2, s_p2, atol=1e-5)


def test_physical_leak_partition_consistency():
    """Splitting one interval into two (same held input) must give the same
    state within tolerance, with the error shrinking ~quadratically in dt."""
    sp = ReservoirSpecialist(_cfg(seed=8, leak_mode="physical", tau=0.2))
    torch.manual_seed(0)
    x_step = torch.randn(1, 5)
    r0 = torch.zeros(1, 32)

    def err_for(dt):
        whole = sp.step(x_step, torch.tensor([dt]), r0)
        half = sp.step(x_step, torch.tensor([dt / 2]), r0)
        split = sp.step(x_step, torch.tensor([dt / 2]), half)
        return float((whole - split).abs().max())

    e1 = err_for(0.02)
    e2 = err_for(0.01)
    assert e1 < 5e-3            # within tolerance at a realistic interval
    assert e2 < 0.35 * e1       # ~O(dt^2): halving dt quarters the error


def test_reservoir_health_flags():
    sp = ReservoirSpecialist(_cfg(seed=9))
    x = torch.randn(4, 12, 5)
    dt = torch.full((4, 12), 1.0 / 30.0)
    _, states, _ = sp(x, dt)
    h = reservoir_health(states)
    assert h["ok"], h
    bad = reservoir_health(torch.full((2, 5, 32), float("nan")))
    assert "nan_inf" in bad["failures"]
    flat = reservoir_health(torch.zeros(2, 5, 32))
    assert "constant_collapse" in flat["failures"]
