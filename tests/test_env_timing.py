"""Deterministic environment timing gate (Priority 10).

Fixed and irregular schedules must end at IDENTICAL simulated times, every
reported dt must equal the interval actually simulated, and collision cost
must integrate measured contact duration rather than charging whole
intervals for endpoint contact.
"""
import copy

import numpy as np
import pytest

from common import load_config
from environment.nav_env import NavEnv


def cfg_with(episode_seconds=2.0, **env_overrides):
    cfg = copy.deepcopy(load_config())
    cfg["env"]["episode_seconds"] = episode_seconds
    cfg["env"].update(env_overrides)
    return cfg


def run_to_truncation(env, seed, action):
    obs, info = env.reset(seed=seed)
    dts, done = [], False
    while not done:
        obs, r, term, trunc, info = env.step(action)
        dts.append(info["dt"])
        done = term or trunc
    return info, dts, term


@pytest.mark.parametrize("irregular", [False, True])
def test_exact_horizon(irregular):
    """Every schedule ends at exactly episode_seconds of simulated time."""
    cfg = cfg_with(2.0)
    env = NavEnv(cfg, irregular_dt=irregular)
    for seed in (0, 1, 2):
        info, dts, term = run_to_truncation(env, seed, np.zeros(2))
        assert not term                       # zero action can't reach goal
        assert info["sim_time"] == pytest.approx(2.0, abs=1e-12), \
            f"overshoot: ended at {info['sim_time']}"
        assert sum(dts) == pytest.approx(2.0, abs=1e-9)
    env.close()


def test_fixed_and_irregular_end_at_same_time():
    cfg = cfg_with(2.0)
    ends = []
    for irregular in (False, True):
        env = NavEnv(cfg, irregular_dt=irregular)
        info, _, _ = run_to_truncation(env, 7, np.zeros(2))
        ends.append(info["sim_time"])
        env.close()
    assert ends[0] == ends[1]


def test_reported_dt_matches_simulated_interval():
    """info['dt'] and the obs dt channel reflect the interval ACTUALLY
    simulated (integer physics substeps), including the clamped final one."""
    cfg = cfg_with(2.0)
    env = NavEnv(cfg, irregular_dt=True)
    obs, _ = env.reset(seed=3)
    done = False
    hz = cfg["env"]["physics_hz"]
    nominal_dt = cfg["env"]["control_substeps"] / hz
    while not done:
        obs, r, term, trunc, info = env.step(np.zeros(2))
        substeps = info["dt"] * hz
        assert substeps == pytest.approx(round(substeps), abs=1e-9), \
            "dt is not an integer number of physics substeps"
        assert obs[21] == pytest.approx(info["dt"] / nominal_dt, abs=1e-6)
        done = term or trunc
    env.close()


def test_collision_cost_integrates_contact_time():
    """Contact duration is measured per substep: it is bounded by the
    interval, quantized at the physics step, and entry counting works."""
    cfg = cfg_with(10.0)
    env = NavEnv(cfg, irregular_dt=False, layout="u_trap")
    obs, _ = env.reset(seed=0)
    hz = cfg["env"]["physics_hz"]
    hit = False
    done = False
    total_entries = 0
    while not done:
        obs, r, term, trunc, info = env.step(np.array([1.0, 1.0]))
        cd = info["contact_duration"]
        assert 0.0 <= cd <= info["dt"] + 1e-9
        n = cd * hz
        assert n == pytest.approx(round(n), abs=1e-9), \
            "contact duration is not an integer number of substeps"
        assert info["collided"] == (cd > 0.0)
        total_entries += info["collision_entries"]
        hit = hit or info["collided"]
        done = term or trunc
    env.close()
    # driving straight in the u-trap arena must produce contact eventually
    assert hit, "robot never collided while driving straight for 10 s"
    assert total_entries >= 1


def test_endpoint_only_contact_not_full_interval():
    """A step whose contact begins mid-interval must be charged less than
    the full interval (regression against endpoint charging)."""
    cfg = cfg_with(10.0)
    env = NavEnv(cfg, irregular_dt=False, layout="u_trap")
    env.reset(seed=0)
    partial = False
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.array([1.0, 1.0]))
        if 0.0 < info["contact_duration"] < info["dt"] - 1e-9:
            partial = True
            break
        done = term or trunc
    env.close()
    assert partial, "never observed a partial-contact interval"


def test_gap_injection():
    """gap_prob=1 forces every interval into the configured gap range."""
    cfg = cfg_with(2.0, gap_prob=1.0)
    env = NavEnv(cfg, irregular_dt=True)
    env.reset(seed=0)
    lo = cfg["env"]["gap_substeps_min"] / cfg["env"]["physics_hz"]
    done = False
    last = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(2))
        done = term or trunc
        # every interval except the clamped final one sits in the gap range
        if not done:
            assert info["dt"] >= lo - 1e-9
    env.close()
