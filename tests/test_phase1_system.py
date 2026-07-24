"""Phase 1 deterministic tests: predictors, modular system, GRU matching."""
import numpy as np
import torch

from phase1 import HORIZON_STEPS, MAX_HORIZON_STEPS, PRIMARY_HORIZON_S
from phase1.experiment import build_matched_gru, build_system
from phase1.gru_baseline import count_trainable
from phase1.predictors import ActionEncoder


def _batch(B=4, L=15, H=MAX_HORIZON_STEPS, seed=0):
    rng = np.random.default_rng(seed)
    t = lambda *shape: torch.from_numpy(
        rng.normal(size=shape).astype(np.float32))
    return {
        "lidar": t(B, L, 16), "body": t(B, L, 2), "actions": t(B, L, 2),
        "prev_actions": t(B, L, 2),
        "dts": torch.full((B, L, 1), 1.0 / 30.0),
        "valid_mask": torch.ones(B, L),
        "future_lidar": t(B, H, 16), "future_body": t(B, H, 2),
        "future_actions": t(B, H, 2),
        "future_dts": torch.full((B, H, 1), 1.0 / 30.0),
    }


def test_reversed_action_order_changes_predictions():
    system = build_system(0, 0)
    bt = _batch()
    p_fwd = system(bt)
    bt_rev = dict(bt)
    bt_rev["future_actions"] = torch.flip(bt["future_actions"], dims=[1])
    p_rev = system(bt_rev)
    h = PRIMARY_HORIZON_S
    for key in ("local_lidar", "local_body", "rel_lidar", "rel_body"):
        assert not torch.allclose(p_fwd[key][h], p_rev[key][h]), key


def test_action_encoder_is_order_sensitive():
    enc = ActionEncoder(seed=0)
    a = torch.randn(2, 10, 2)
    d = torch.full((2, 10, 1), 1.0 / 30.0)
    out_fwd = enc(a, d)
    out_rev = enc(torch.flip(a, dims=[1]), d)
    assert not torch.allclose(out_fwd[:, -1], out_rev[:, -1])


def test_gradients_only_in_readouts_and_predictors():
    system = build_system(0, 0)
    bt = _batch()
    preds = system(bt)
    loss, _ = system.loss(preds, bt)
    loss.backward()
    trainable_with_grad = [n for n, p in system.named_parameters()
                           if p.requires_grad and p.grad is not None
                           and p.grad.abs().sum() > 0]
    assert any(n.startswith("lidar_esn.readout") for n in trainable_with_grad)
    assert any(n.startswith("body_esn.readout") for n in trainable_with_grad)
    assert any(n.startswith("relational") for n in trainable_with_grad)
    assert any(n.startswith("action_encoder") for n in trainable_with_grad)
    # reservoir matrices are buffers: not parameters at all, never updated
    param_names = {n for n, _ in system.named_parameters()}
    for forbidden in ("lidar_esn.w_in", "lidar_esn.w_rec", "body_esn.w_in",
                      "body_esn.w_rec"):
        assert forbidden not in param_names
    buffer_names = {n for n, _ in system.named_buffers()}
    assert "lidar_esn.w_rec" in buffer_names


def test_system_deterministic_given_seeds():
    a = build_system(3, 7)
    b = build_system(3, 7)
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k
    c = build_system(4, 7)
    assert not torch.equal(sd_a["lidar_esn.w_rec"],
                           c.state_dict()["lidar_esn.w_rec"])


def test_isolation_of_specialist_inputs():
    """Lidar messages must not depend on body state; body messages must not
    depend on lidar."""
    system = build_system(0, 0)
    bt1 = _batch(seed=1)
    bt2 = {k: v.clone() for k, v in bt1.items()}
    bt2["body"] = bt2["body"] + 1.0             # perturb body only
    fl1, fb1 = system.specialist_features(bt1)
    fl2, fb2 = system.specialist_features(bt2)
    assert torch.equal(fl1, fl2)                # lidar unaffected
    assert not torch.equal(fb1, fb2)
    bt3 = {k: v.clone() for k, v in bt1.items()}
    bt3["lidar"] = bt3["lidar"] + 1.0           # perturb lidar only
    fl3, fb3 = system.specialist_features(bt3)
    assert not torch.equal(fl1, fl3)
    assert torch.equal(fb1, fb3)                # body unaffected


def test_gru_parameter_matching():
    system = build_system(0, 0)
    target = count_trainable(system)
    gru = build_matched_gru(target, model_seed=0)
    got = count_trainable(gru)
    assert abs(got - target) / target < 0.05, (got, target)
    # the analytic count used for matching agrees with the real module
    from phase1.gru_baseline import _analytic_trainable
    assert _analytic_trainable(gru.hidden, 4, 21) == got
    # the GRU recurrence itself must be trainable (it is the contrast)
    assert any(n.startswith("gru.") and p.requires_grad
               for n, p in gru.named_parameters())


def test_horizon_steps_match_nominal_cadence():
    assert HORIZON_STEPS == {0.25: 8, 0.5: 15, 1.0: 30, 2.0: 60}
