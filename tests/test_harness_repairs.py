"""Harness repairs: CEM generator seeding, repeat aggregation, factorial I/O."""
import copy
import os

import numpy as np
import pytest
import torch

from common import load_config
from agent.hybrid_agent import HybridAgent
from agent.planner import CEMPlanner
from agent.world_model import WorldModel
from environment.nav_env import OBS_DIM
from scripts.eval_common import (aggregate_by_env_seed, compare,
                                 within_env_planner_variance)
from scripts.factorial_io import (cell_entry, load_manifest, write_manifest)


def test_cem_generator_is_deterministic():
    cfg = load_config()
    wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                    hidden=int(cfg["world_model"]["hidden_dim"]))
    wm.eval()
    p1 = CEMPlanner(wm, horizon=2, population=8, elites=2, iterations=2)
    p2 = CEMPlanner(wm, horizon=2, population=8, elites=2, iterations=2)
    z = torch.zeros(1, int(cfg["agent"]["latent_dim"]))
    p1.seed(12345)
    a, info1 = p1.plan(z, return_info=True)
    p2.seed(12345)
    b, info2 = p2.plan(z, return_info=True)
    assert torch.allclose(a, b)
    assert info1["planner_seed"] == 12345
    assert info1["best_score"] == info2["best_score"]
    # Different seed -> different plan (with high probability)
    p2.seed(99999)
    c, _ = p2.plan(z, return_info=True)
    assert not torch.allclose(a, c)


def test_cem_does_not_use_global_rng():
    cfg = load_config()
    wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                    hidden=int(cfg["world_model"]["hidden_dim"]))
    wm.eval()
    p = CEMPlanner(wm, horizon=2, population=8, elites=2, iterations=1)
    z = torch.zeros(1, int(cfg["agent"]["latent_dim"]))
    p.seed(7)
    torch.manual_seed(0)
    a, _ = p.plan(z, return_info=True)
    p.seed(7)
    torch.manual_seed(999)
    b, _ = p.plan(z, return_info=True)
    assert torch.allclose(a, b)


def test_aggregate_repeats_before_compare():
    records_a = []
    records_b = []
    for seed in (1, 2, 3):
        for rep in range(3):
            records_a.append({
                "env_seed": seed, "cem_repeat": rep, "planner_seed": 10 + rep,
                "reward": 1.0 + seed + 0.1 * rep, "success": rep > 0,
                "final_goal_dist": 1.0, "collision_duration": 0.0,
                "path_efficiency": 0.5})
            records_b.append({
                "env_seed": seed, "cem_repeat": rep, "planner_seed": 10 + rep,
                "reward": 0.5 + seed, "success": False,
                "final_goal_dist": 2.0, "collision_duration": 0.1,
                "path_efficiency": 0.4})
    agg = aggregate_by_env_seed(records_a)
    assert len(agg) == 3
    assert agg[0]["n_repeats"] == 3
    c = compare(records_a, records_b, "a", "b")
    assert c["n_pairs"] == 3
    assert c["aggregated_repeats"] is True
    assert "success_rate_diff" in c
    var = within_env_planner_variance(records_a)
    assert var["n_env_with_repeats"] == 3


def test_factorial_manifest_roundtrip(tmp_path):
    path = str(tmp_path / "factorial_manifest_smoke.json")
    cell = {"name": "physsnn-physcfc-masked",
            "snn_time_aware": True, "cfc_time_aware": True,
            "mask_direct_dt": True, "hierarchical": False}
    cfg = load_config()
    ckpt = str(tmp_path / "fake.pt")
    # minimal provenance-shaped file for cell_entry
    torch.save({"state": {"w": torch.zeros(1)},
                "meta": {"state_checksum": "abc", "compat": {}}}, ckpt)
    entry = cell_entry(cell, ckpt, 0, "factorial_test_smoke", True, cfg)
    write_manifest(path, [entry], smoke=True)
    man = load_manifest(path)
    assert man["n_cells"] == 1
    assert man["cells"][0]["name"] == "physsnn-physcfc-masked"
    assert man["cells"][0]["variant_factors"]["mask_direct_dt"] is True


def test_hierarchy_gate_incomplete_without_equal_extra():
    """Unit-level: missing equal-extra => status incomplete, passed False."""
    # Simulate the gate decision logic used in eval_hierarchy
    missing = ["equal_extra_reactive"]
    criteria = {
        "reactive": {"available": True, "passed": True},
        "hier_zero": {"available": True, "passed": True},
        "hier_random": {"available": True, "passed": True},
        "hier_shuffled": {"available": True, "passed": True},
        "hier_heuristic": {"available": True, "passed": True},
        "equal_extra_reactive": {"available": False, "passed": False},
    }
    required = list(criteria)
    if missing:
        status, passed = "incomplete", False
    elif all(criteria[k]["passed"] for k in required):
        status, passed = "passed", True
    else:
        status, passed = "failed", False
    assert status == "incomplete" and passed is False
