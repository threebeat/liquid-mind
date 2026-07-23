"""Agent event-timing semantics (Priorities 2, 3).

Causal vs legacy irregular-RNN conventions, initial assimilation at elapsed
time zero, direct-dt masking, capacity matching across policy-input modes,
and checkpoint compatibility validation.
"""
import copy

import numpy as np
import pytest
import torch

from common import load_config
from agent.hybrid_agent import HybridAgent
from environment.nav_env import DT_OBS_INDEX, OBS_DIM


def cfg_with(**agent_overrides):
    cfg = copy.deepcopy(load_config())
    cfg["agent"].update(agent_overrides)
    return cfg


def make_agent(seed=0, **agent_overrides):
    torch.manual_seed(seed)
    return HybridAgent(cfg_with(**agent_overrides), mode="reactive")


def obs_stream(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, OBS_DIM).astype(np.float32) for _ in range(n)]


def test_causal_vs_irnn_differ():
    """The two conventions integrate DIFFERENT measurements across the
    elapsed interval, so with a varying input stream they must diverge."""
    stream = obs_stream()
    a_causal = make_agent(timing_convention="causal")
    a_irnn = make_agent(timing_convention="irnn")
    acts_c = [a_causal.act(o, 0.05) for o in stream]
    acts_i = [a_irnn.act(o, 0.05) for o in stream]
    diffs = [np.abs(c - i).max() for c, i in zip(acts_c, acts_i)]
    assert max(diffs) > 1e-6, "conventions produced identical trajectories"


def test_causal_initial_assimilation_at_time_zero():
    """No fabricated nominal interval on reset: the first event advances
    zero physical time regardless of the dt argument."""
    a = make_agent(timing_convention="causal")
    a.act(obs_stream()[0], 0.7)          # bogus dt on the first event
    assert a._sim_time == 0.0
    a.act(obs_stream()[1], 0.05)
    assert a._sim_time == pytest.approx(0.05)


def test_irnn_uses_given_dt_from_the_start():
    a = make_agent(timing_convention="irnn")
    a.act(obs_stream()[0], 0.05)
    assert a._sim_time == pytest.approx(0.05)


def test_mask_direct_dt_hides_the_raw_channel():
    """With masking on, two observations differing only in the raw dt
    channel must produce identical actions; with masking off they differ."""
    o1 = obs_stream()[0].copy()
    o2 = o1.copy()
    o2[DT_OBS_INDEX] = 3.0               # wildly different raw dt value

    masked = make_agent(mask_direct_dt=True)
    x1 = masked.act(o1, 0.05)
    masked.reset()
    x2 = masked.act(o2, 0.05)
    assert np.allclose(x1, x2), "masked agent leaked the raw dt channel"

    visible = make_agent(mask_direct_dt=False)
    y1 = visible.act(o1, 0.05)
    visible.reset()
    y2 = visible.act(o2, 0.05)
    assert not np.allclose(y1, y2), "dt-visible agent ignored the channel"


def test_nominal_time_arms_use_nominal_dt():
    """snn/cfc_time_aware=False + masked dt: two different true intervals
    must produce identical behavior (no timing side-channel anywhere)."""
    o = obs_stream()
    a1 = make_agent(snn_time_aware=False, cfc_time_aware=False,
                    mask_direct_dt=True)
    r1 = [a1.act(x, dt) for x, dt in zip(o, (0.02, 0.08, 0.03, 0.05, 0.07, 0.04))]
    a2 = make_agent(snn_time_aware=False, cfc_time_aware=False,
                    mask_direct_dt=True)
    r2 = [a2.act(x, dt) for x, dt in zip(o, (0.05, 0.02, 0.09, 0.03, 0.06, 0.08))]
    for x, y in zip(r1, r2):
        assert np.allclose(x, y, atol=1e-6), \
            "nominal-time agent still conditioned on the true interval"


def test_capacity_matching_across_input_modes():
    """Every policy_input mode uses the same 54→32 adapter and identical CfC
    parameter shapes (shared downstream architecture)."""
    adapter_shapes = {}
    policy_shapes = {}
    for mode in ("spikes_obs", "spikes_only", "obs_only"):
        a = make_agent(policy_input=mode, use_input_adapter=True)
        assert a.adapter.in_features == 54
        assert a.adapter.out_features == 32
        adapter_shapes[mode] = [tuple(p.shape) for p in a.adapter.parameters()]
        policy_shapes[mode] = [tuple(p.shape) for p in a.policy.parameters()]
    assert adapter_shapes["spikes_obs"] == adapter_shapes["spikes_only"] \
        == adapter_shapes["obs_only"]
    assert policy_shapes["spikes_obs"] == policy_shapes["spikes_only"] \
        == policy_shapes["obs_only"]
    # Active evolvable counts still differ (obs_only drops SNN)
    n_obs = make_agent(policy_input="obs_only").n_parameters()
    n_both = make_agent(policy_input="spikes_obs").n_parameters()
    assert n_both > n_obs


def test_obs_only_excludes_snn_parameters():
    a = make_agent(policy_input="obs_only")
    snn_params = set(map(id, a.snn.parameters()))
    assert not snn_params & set(map(id, a.parameters()))


def test_variant_tags_distinguish_cells():
    tags = set()
    for snn_t in (True, False):
        for cfc_t in (True, False):
            for mask in (True, False):
                a = make_agent(snn_time_aware=snn_t, cfc_time_aware=cfc_t,
                               mask_direct_dt=mask)
                tags.add(a.variant_tag())
    assert len(tags) == 8, f"variant tags collide: {tags}"


def test_checkpoint_compat_enforced(tmp_path):
    path = str(tmp_path / "agent.pt")
    a = make_agent(mask_direct_dt=True)
    a.save(path)
    # identical config loads fine
    make_agent(mask_direct_dt=True).load(path)
    # incompatible config is refused with the differing keys named
    b = make_agent(mask_direct_dt=False, snn_semantics="sampled_binary")
    with pytest.raises(ValueError) as e:
        b.load(path)
    assert "mask_direct_dt" in str(e.value)
    assert "snn_semantics" in str(e.value)


def test_agent_save_never_overwrites(tmp_path):
    path = str(tmp_path / "agent.pt")
    a = make_agent()
    a.save(path)
    with pytest.raises(FileExistsError):
        a.save(path)
    a.save(path, force=True)


def test_event_counts_reach_policy():
    """In event_count semantics the SNN features can exceed 1 (multiplicity
    preserved end to end)."""
    a = make_agent(snn_semantics="event_count")
    rng = np.random.default_rng(0)
    saw_multi = False
    obs = rng.uniform(-1, 1, OBS_DIM).astype(np.float32) * 3.0
    a.act(obs, 0.05)
    for _ in range(30):
        a.act(obs, 0.2)                 # long intervals, strong input
        if float(a._pending_feats.max()) >= 2.0:
            saw_multi = True
            break
    assert saw_multi, "no multi-event feature observed"
