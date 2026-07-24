"""Phase 1 deterministic tests: ordered-window dataset and splits."""
import os

import numpy as np
import pytest

from phase1 import CONTEXT_LEN, MAX_HORIZON_STEPS
from phase1.dataset import (WindowDataset, load_split_manifest,
                            make_episode_split, save_split_manifest)
from training.replay_buffer import ReplayBuffer


def _synthetic_buffer(n_episodes=6, min_T=80, seed=0) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer()
    for i in range(n_episodes):
        T = int(min_T + rng.integers(0, 40))
        obs = rng.normal(size=(T + 1, 22)).astype(np.float32)
        # mark each obs row with its episode and step index for traceability
        obs[:, 0] = i
        obs[:, 1] = np.arange(T + 1)
        actions = rng.uniform(-1, 1, (T, 2)).astype(np.float32)
        dts = rng.uniform(0.017, 0.057, T).astype(np.float32)
        buf.add_episode(obs, actions, dts)
    return buf


def test_no_window_crosses_episode_boundary():
    buf = _synthetic_buffer()
    ds = WindowDataset(buf, range(len(buf)), stride=1)
    assert len(ds) > 0
    for li, t in ds.anchors:
        T = len(ds.episodes[li]["actions"])
        assert t + MAX_HORIZON_STEPS <= T          # future stays inside
        assert t >= ds.min_context - 1             # enough real context
    for idx in range(0, len(ds), 7):
        w = ds.window(idx)
        li, t = ds.anchors[idx]
        # every valid context row and every future row carries the SAME
        # episode marker -> no cross-boundary mixing
        valid = w["valid_mask"] > 0
        assert np.all(w["lidar"][valid][:, 0] == li)
        assert np.all(w["future_lidar"][:, 0] == li)
        # future step indices are strictly consecutive from the anchor
        assert np.array_equal(w["future_lidar"][:, 1],
                              np.arange(t + 1, t + 1 + ds.H))


def test_actions_are_ordered_and_never_averaged():
    buf = _synthetic_buffer()
    ds = WindowDataset(buf, [2], stride=3)
    e = ds.episodes[0]
    for idx in range(len(ds)):
        li, t = ds.anchors[idx]
        w = ds.window(idx)
        assert np.array_equal(w["future_actions"], e["actions"][t:t + ds.H])
        L = ds.L
        j0 = t - L + 1
        for i in range(L):
            j = j0 + i
            if j < 0:
                assert w["valid_mask"][i] == 0
            else:
                assert np.array_equal(w["actions"][i], e["actions"][j])
                if j >= 1:
                    assert np.array_equal(w["prev_actions"][i],
                                          e["actions"][j - 1])


def test_dts_align_with_observations():
    buf = _synthetic_buffer(n_episodes=1)
    ds = WindowDataset(buf, [0], stride=1)
    e = ds.episodes[0]
    w = ds.window(len(ds) - 1)
    li, t = ds.anchors[len(ds) - 1]
    # dt attached to obs j is the interval that ENDED at obs j
    for i in range(ds.L):
        j = t - ds.L + 1 + i
        if j >= 1:
            assert abs(w["dts"][i, 0] - e["dts"][j - 1]) < 1e-7
    assert np.allclose(w["future_dts"][:, 0], e["dts"][t:t + ds.H])


def test_split_disjoint_and_deterministic():
    s1 = make_episode_split(50, 30, 10, 10, seed=42)
    s2 = make_episode_split(50, 30, 10, 10, seed=42)
    s3 = make_episode_split(50, 30, 10, 10, seed=43)
    assert s1 == s2
    assert s1 != s3
    assert not (set(s1["train"]) & set(s1["val"]))
    assert not (set(s1["train"]) & set(s1["test"]))
    assert not (set(s1["val"]) & set(s1["test"]))
    assert len(s1["train"]) == 30 and len(s1["val"]) == 10 \
        and len(s1["test"]) == 10


def test_split_manifest_checksum_guard(tmp_path):
    buf = _synthetic_buffer(n_episodes=4)
    p1 = str(tmp_path / "buf_a.npz")
    p2 = str(tmp_path / "buf_b.npz")
    buf.save(p1, meta={"x": 1})
    buf.save(p2, meta={"x": 2})   # different meta -> different checksum
    split = make_episode_split(4, 2, 1, 1, seed=0)
    mpath = str(tmp_path / "manifest.json")
    save_split_manifest(mpath, split, p1)
    loaded = load_split_manifest(mpath, buffer_path=p1)
    assert loaded["train"] == split["train"]
    with pytest.raises(ValueError, match="sha256"):
        load_split_manifest(mpath, buffer_path=p2)


def test_dataset_deterministic_fingerprint_and_order():
    buf = _synthetic_buffer()
    d1 = WindowDataset(buf, [0, 2, 4], stride=2)
    d2 = WindowDataset(buf, [0, 2, 4], stride=2)
    assert d1.anchors == d2.anchors
    assert d1.fingerprint("sha") == d2.fingerprint("sha")
    assert d1.fingerprint("sha") != d2.fingerprint("other-buffer")
    d3 = WindowDataset(buf, [0, 2], stride=2)
    assert d1.fingerprint("sha") != d3.fingerprint("sha")
    assert np.array_equal(d1.epoch_order(5), d2.epoch_order(5))


def test_window_shapes():
    buf = _synthetic_buffer()
    ds = WindowDataset(buf, range(len(buf)))
    w = ds.window(0)
    assert w["lidar"].shape == (CONTEXT_LEN, 16)
    assert w["body"].shape == (CONTEXT_LEN, 2)
    assert w["actions"].shape == (CONTEXT_LEN, 2)
    assert w["dts"].shape == (CONTEXT_LEN, 1)
    assert w["valid_mask"].shape == (CONTEXT_LEN,)
    assert w["future_lidar"].shape == (MAX_HORIZON_STEPS, 16)
    assert w["future_body"].shape == (MAX_HORIZON_STEPS, 2)
    b = ds.batch([0, 1, 2])
    assert b["lidar"].shape == (3, CONTEXT_LEN, 16)
