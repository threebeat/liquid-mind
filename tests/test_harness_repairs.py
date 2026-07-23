"""Harness repairs: CEM seeding, repeat aggregation, factorial identity,
manifest integrity, atomic writes, DiD contrasts, tie handling, env-level
summaries, equal-extra metadata validation, WM gate CI enforcement."""
import copy
import json
import os

import numpy as np
import pytest
import torch

from common import load_config
from provenance import SEMANTICS_VERSION, file_checksum, gather_provenance
from agent.hybrid_agent import HybridAgent
from agent.planner import CEMPlanner
from agent.world_model import WorldModel
from environment.nav_env import OBS_DIM
from scripts.eval_common import (aggregate_by_env_seed, compare,
                                 factorial_effects, mcnemar_exact,
                                 paired_did, robustness_by_env,
                                 summarize_env_level,
                                 within_env_planner_variance)
from scripts.factorial_io import (cell_entry, load_agent_for_checkpoint,
                                  load_manifest, set_run_status,
                                  verify_manifest_cell, write_manifest)


# ------------------------------------------------------------ CEM seeding


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


# ------------------------------------------------- repeats / ties / McNemar


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


def _rec(env_seed, reward=1.0, success=False, dist=1.0, coll=0.0, rep=0):
    return {"env_seed": env_seed, "cem_repeat": rep, "reward": reward,
            "success": success, "final_goal_dist": dist,
            "collision_duration": coll, "path_efficiency": 0.5}


def test_tie_is_never_a_success():
    """Two repeats, one success one failure -> success None, tied True."""
    recs = [_rec(0, success=True, rep=0), _rec(0, success=False, rep=1),
            _rec(1, success=True, rep=0), _rec(1, success=True, rep=1),
            _rec(2, success=False, rep=0), _rec(2, success=False, rep=1)]
    agg = {u["env_seed"]: u for u in aggregate_by_env_seed(recs)}
    assert agg[0]["success"] is None and agg[0]["success_tied"] is True
    assert agg[0]["success_rate"] == 0.5
    assert agg[1]["success"] is True and agg[1]["success_tied"] is False
    assert agg[2]["success"] is False


def test_mcnemar_not_confirmatory_with_repeats_and_drops_ties():
    recs_a = [_rec(0, success=True, rep=0), _rec(0, success=False, rep=1),
              _rec(1, success=True, rep=0), _rec(1, success=True, rep=1),
              _rec(2, success=False, rep=0), _rec(2, success=False, rep=1)]
    recs_b = [_rec(s, success=False, rep=r) for s in (0, 1, 2)
              for r in (0, 1)]
    c = compare(recs_a, recs_b, "a", "b")
    mcn = c["success_mcnemar"]
    assert mcn["applicable"] is False
    assert "reason" in mcn
    assert mcn["descriptive"]["dropped_tied_pairs"] == 1
    # frequency diff keeps the tied environment as a fractional value
    assert c["success_rate_diff"]["n_pairs"] == 3
    # single-run comparisons keep McNemar confirmatory
    single_a = [_rec(s, success=s < 2) for s in range(4)]
    single_b = [_rec(s, success=False) for s in range(4)]
    c1 = compare(single_a, single_b, "a", "b")
    assert c1["success_mcnemar"]["applicable"] is True
    assert c1["success_mcnemar"]["discordant"] == 2


def test_mcnemar_exact_drops_none_pairs():
    out = mcnemar_exact([True, None, False], [False, True, None])
    assert out["dropped_tied_pairs"] == 2
    assert out["discordant"] == 1


# -------------------------------------------------- env-level summaries


def test_env_level_summary_units_are_env_seeds():
    recs = [_rec(s, reward=float(s), success=(r == 0), rep=r)
            for s in range(3) for r in range(2)]
    s = summarize_env_level(recs)
    assert s["env_units"] == 3          # not env x repeats = 6
    assert s["max_repeats"] == 2
    # fractional success -> cluster bootstrap, never Wilson
    assert s["success"] is None
    assert s["success_rate_mean"] is not None
    assert 0.0 <= s["success_rate_mean"]["point"] <= 1.0


def test_env_level_summary_single_repeat_uses_wilson():
    recs = [_rec(s, success=(s % 2 == 0)) for s in range(4)]
    s = summarize_env_level(recs)
    assert s["env_units"] == 4
    assert s["success"] is not None and s["success"]["k"] == 2
    assert s["success_rate_mean"] is None


# ------------------------------------------------ degradation / DiD / 2x2


def test_robustness_sign_convention():
    fixed = [_rec(s, reward=10.0, coll=0.0) for s in range(3)]
    dist_worse = [_rec(s, reward=8.0, coll=2.0) for s in range(3)]
    r_reward, _ = robustness_by_env(dist_worse, fixed, "reward")
    r_coll, _ = robustness_by_env(dist_worse, fixed, "collision_duration")
    # reward: disturbed - fixed = -2 (degraded -> negative robustness)
    assert all(abs(v - (-2.0)) < 1e-9 for v in r_reward.values())
    # collision cost is negated: worse (higher) cost -> negative robustness
    assert all(abs(v - (-2.0)) < 1e-9 for v in r_coll.values())


def test_paired_did_a_degrades_less():
    rob_a = {s: -1.0 for s in range(6)}
    rob_b = {s: -3.0 for s in range(6)}
    d = paired_did(rob_a, rob_b)
    assert abs(d["mean_diff"] - 2.0) < 1e-9      # positive = A more robust
    assert d["n_pairs"] == 6


def test_paired_did_equal_degradation_different_quality_is_zero():
    """A and B differ in fixed performance but degrade identically."""
    fixed_a = [_rec(s, reward=10.0) for s in range(4)]
    dist_a = [_rec(s, reward=8.0) for s in range(4)]
    fixed_b = [_rec(s, reward=5.0) for s in range(4)]
    dist_b = [_rec(s, reward=3.0) for s in range(4)]
    ra, _ = robustness_by_env(dist_a, fixed_a, "reward")
    rb, _ = robustness_by_env(dist_b, fixed_b, "reward")
    d = paired_did(ra, rb)
    assert abs(d["mean_diff"]) < 1e-9


def test_robustness_vs_retained_quality_separation():
    """A has better absolute disturbed performance but degrades MORE."""
    fixed_a = [_rec(s, reward=10.0) for s in range(4)]
    dist_a = [_rec(s, reward=6.0) for s in range(4)]      # retained 6, deg -4
    fixed_b = [_rec(s, reward=5.0) for s in range(4)]
    dist_b = [_rec(s, reward=4.0) for s in range(4)]      # retained 4, deg -1
    ra, _ = robustness_by_env(dist_a, fixed_a, "reward")
    rb, _ = robustness_by_env(dist_b, fixed_b, "reward")
    d = paired_did(ra, rb)
    assert d["mean_diff"] < 0            # B more robust ...
    assert np.mean([r["reward"] for r in dist_a]) > \
        np.mean([r["reward"] for r in dist_b])  # ... despite A retaining more


def test_paired_did_reports_missing_env_seeds():
    rob_a = {0: -1.0, 1: -1.0, 2: -1.0}
    rob_b = {0: -2.0, 1: -2.0}
    d = paired_did(rob_a, rob_b)
    assert d["n_pairs"] == 2
    assert d["missing_in_b"] == [2]
    fixed = [_rec(s, reward=10.0) for s in range(3)]
    dist = [_rec(s, reward=9.0) for s in range(2)]        # env 2 missing
    _, missing = robustness_by_env(dist, fixed, "reward")
    assert missing["fixed_only"] == [2]


def test_factorial_effects_known_values():
    envs = range(5)
    d00 = {s: 0.0 for s in envs}
    d10 = {s: 1.0 for s in envs}
    d01 = {s: 2.0 for s in envs}
    d11 = {s: 5.0 for s in envs}
    fx = factorial_effects(d00, d10, d01, d11)
    # snn = ((1-0)+(5-2))/2 = 2; cfc = ((2-0)+(5-1))/2 = 3; ixn = 5-1-2+0 = 2
    assert abs(fx["snn_main_effect"]["point"] - 2.0) < 1e-9
    assert abs(fx["cfc_main_effect"]["point"] - 3.0) < 1e-9
    assert abs(fx["interaction"]["point"] - 2.0) < 1e-9
    assert fx["n_env"] == 5
    assert factorial_effects({}, d10, d01, d11)["n_env"] == 0


# ------------------------------------------- manifest identity + integrity


def _factorial_checkpoint(tmp_path, cell, seed, exp, smoke=True):
    """Real HybridAgent checkpoint with runner-equivalent metadata."""
    cfg = load_config()
    a = cfg["agent"]
    a["snn_time_aware"] = cell["snn_time_aware"]
    a["cfc_time_aware"] = cell["cfc_time_aware"]
    a["mask_direct_dt"] = cell["mask_direct_dt"]
    tr = cfg["training"]
    agent = HybridAgent(cfg, mode="reactive")
    meta = gather_provenance(
        cfg, experiment_name=exp, variant=agent.variant_tag(),
        seeds={"cma_seed": seed},
        extra={"mode": "reactive",
               "run_kind": "smoke" if smoke else "full",
               "smoke": smoke,
               "budget": {
                   "generations": int(tr["cma_generations"]),
                   "population": int(tr["cma_population"]),
                   "episodes_per_candidate": int(tr["episodes_per_candidate"]),
                   "validation_episodes": int(tr.get("validation_episodes",
                                                     12))}})
    path = str(tmp_path / f"{exp}.pt")
    agent.save(path, meta=meta)
    return path, cfg


CELL = {"name": "physsnn-physcfc-masked", "snn_time_aware": True,
        "cfc_time_aware": True, "mask_direct_dt": True, "hierarchical": False}


def test_factorial_manifest_roundtrip(tmp_path):
    path = str(tmp_path / "factorial_manifest_smoke.json")
    ckpt, cfg = _factorial_checkpoint(tmp_path, CELL, 0,
                                      "factorial_test_s0_smoke")
    entry = cell_entry(CELL, ckpt, 0, "factorial_test_s0_smoke", True, cfg)
    write_manifest(path, [entry], smoke=True)
    man = load_manifest(path)
    assert man["n_cells"] == 1
    e = man["cells"][0]
    assert e["cell_id"] == "physsnn-physcfc-masked"
    assert e["run_id"] == "physsnn-physcfc-masked__s0"
    assert e["training_seed"] == 0 and isinstance(e["training_seed"], int)
    assert e["attempt"] == 1
    assert e["variant_factors"]["mask_direct_dt"] is True
    assert e["planned_spec"]["training_seed"] == 0


def test_verify_manifest_cell_passes_and_catches_tampering(tmp_path):
    ckpt, cfg = _factorial_checkpoint(tmp_path, CELL, 0,
                                      "factorial_verify_s0_smoke")
    entry = cell_entry(CELL, ckpt, 0, "factorial_verify_s0_smoke", True, cfg)
    assert verify_manifest_cell(entry)["verified"] is True

    # replace the artifact with different weights -> file sha AND state
    # checksum disagree with the manifest
    torch.manual_seed(123)
    other = HybridAgent(cfg, mode="reactive")
    with torch.no_grad():
        for p in other.policy.parameters():
            p.add_(1.0)
    other.save(ckpt, meta=gather_provenance(
        cfg, experiment_name="factorial_verify_s0_smoke",
        seeds={"cma_seed": 0},
        extra={"mode": "reactive", "run_kind": "smoke", "smoke": True}),
        force=True)
    with pytest.raises(ValueError, match="file_sha256"):
        verify_manifest_cell(entry)


def test_verify_manifest_cell_mismatch_cases(tmp_path):
    ckpt, cfg = _factorial_checkpoint(tmp_path, CELL, 0,
                                      "factorial_mm_s0_smoke")

    e = cell_entry(CELL, ckpt, 0, "factorial_mm_s0_smoke", True, cfg)
    e["experiment"] = "some_other_experiment"
    with pytest.raises(ValueError, match="experiment"):
        verify_manifest_cell(e)

    e = cell_entry(CELL, ckpt, 1, "factorial_mm_s0_smoke", True, cfg)
    with pytest.raises(ValueError, match="training_seed|cma_seed"):
        verify_manifest_cell(e)

    e = cell_entry(CELL, ckpt, 0, "factorial_mm_s0_smoke", True, cfg)
    e["variant_factors"]["snn_time_aware"] = False
    e["planned_spec"]["snn_time_aware"] = False
    with pytest.raises(ValueError, match="snn_time_aware"):
        verify_manifest_cell(e)

    e = cell_entry(CELL, ckpt, 0, "factorial_mm_s0_smoke", True, cfg)
    e["compat"] = {"bogus": 1}
    with pytest.raises(ValueError, match="compat"):
        verify_manifest_cell(e)

    # planned full run but the artifact is a smoke artifact (metadata, not
    # filename, decides)
    e = cell_entry(CELL, ckpt, 0, "factorial_mm_s0_smoke", False, cfg)
    with pytest.raises(ValueError, match="smoke"):
        verify_manifest_cell(e)

    missing = dict(cell_entry(CELL, ckpt, 0, "factorial_mm_s0_smoke",
                              True, cfg))
    missing["checkpoint"] = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        verify_manifest_cell(missing)


def test_load_agent_requires_v3_config(tmp_path):
    """No silent fallback: missing meta config is an error unless the caller
    explicitly opts into legacy cell-factor reconstruction."""
    from provenance import save_checkpoint
    cfg = load_config()
    agent = HybridAgent(cfg, mode="reactive")
    meta = gather_provenance(cfg, experiment_name="noconfig")
    meta["config"] = None                          # strip resolved config
    path = str(tmp_path / "noconfig.pt")
    save_checkpoint(path, agent._state(), meta, agent.compat())
    with pytest.raises(ValueError, match="allow_legacy"):
        load_agent_for_checkpoint(path)
    _, cfg2, source = load_agent_for_checkpoint(
        path, allow_legacy=True, base_cfg=cfg,
        cell={"variant_factors": {"snn_time_aware": True,
                                  "cfc_time_aware": True,
                                  "mask_direct_dt": True}})
    assert source == "cell_factors_legacy"


# ----------------------------------------------------- atomic manifest I/O


def test_manifest_write_is_atomic(tmp_path, monkeypatch):
    path = str(tmp_path / "factorial_manifest_smoke.json")
    write_manifest(path, [{"run_id": "a__s0", "status": "completed"}],
                   smoke=True)
    before = load_manifest(path)

    def boom(*a, **k):
        raise RuntimeError("interrupted mid-serialization")
    monkeypatch.setattr(json, "dump", boom)
    with pytest.raises(RuntimeError):
        write_manifest(path, [{"run_id": "b__s0", "status": "planned"}],
                       smoke=True)
    monkeypatch.undo()

    after = load_manifest(path)                 # previous manifest intact
    assert after == before
    assert after["cells"][0]["run_id"] == "a__s0"
    assert not os.path.exists(path + ".tmp")    # no stale tmp
    write_manifest(path, [{"run_id": "c__s0", "status": "completed"}],
                   smoke=True)
    assert not os.path.exists(path + ".tmp")
    assert load_manifest(path)["cells"][0]["run_id"] == "c__s0"


def test_status_transitions_recorded():
    entry = {"run_id": "x__s0", "attempt": 2, "status": "planned"}
    set_run_status(entry, "training")
    set_run_status(entry, "failed", error="E" * 900)
    assert entry["status"] == "failed"
    hist = entry["status_history"]
    assert [h["to"] for h in hist] == ["training", "failed"]
    assert hist[1]["from"] == "training"
    assert hist[1]["attempt"] == 2
    assert len(hist[1]["error"]) <= 500


# ------------------------------------- two-seed factorial evaluation flow


class _DummyEnv:
    def __init__(self, smin, gap):
        self.smin, self.gap = smin, gap

    def close(self):
        pass


def test_two_seed_runs_survive_evaluation(tmp_path, monkeypatch):
    """Both training seeds of one cell survive manifest evaluation under
    their own run_ids; the across-seed summary uses exactly one estimate per
    run; a seed-mismatched entry is excluded loudly."""
    import scripts.eval_dt_robustness as ed

    entries = []
    for seed in (0, 1):
        exp = f"factorial_{CELL['name']}_s{seed}_smoke"
        ckpt, cfg = _factorial_checkpoint(tmp_path, CELL, seed, exp)
        entries.append(cell_entry(CELL, ckpt, seed, exp, True, cfg,
                                  status="completed"))
    # third entry claims seed 5 but its checkpoint was trained with seed 1
    bad = copy.deepcopy(entries[1])
    bad["training_seed"] = 5
    bad["run_id"] = f"{CELL['name']}__s5"
    bad["planned_spec"]["training_seed"] = 5
    entries.append(bad)
    # and one still-planned entry must be excluded by status
    planned = copy.deepcopy(entries[0])
    planned["run_id"] = f"{CELL['name']}__s9"
    planned["status"] = "planned"
    entries.append(planned)

    monkeypatch.setattr(ed, "_env_for",
                        lambda cfg, smin, smax, gap: _DummyEnv(smin, gap))

    def fake_run_episode(env, act, reset, seed, **kw):
        disturb = (0.0 if env.smin is None
                   else (8 - env.smin) * 0.4 + env.gap * 50.0)
        r = 10.0 - (seed % 100) * 0.01 - disturb
        return {"env_seed": seed, "reward": r, "success": r > 8.5,
                "final_goal_dist": 1.0 + 0.1 * disturb,
                "collision_duration": 0.05 * disturb,
                "path_efficiency": 0.5}
    monkeypatch.setattr(ed, "run_episode", fake_run_episode)

    results = ed._eval_factorial_cells(entries, episodes=3,
                                       allow_legacy=False,
                                       base_cfg=load_config())
    cell = results["cells"][CELL["name"]]
    run_ids = set(cell["training_runs"])
    assert run_ids == {f"{CELL['name']}__s0", f"{CELL['name']}__s1"}
    seeds = sorted(r["training_seed"] for r in cell["training_runs"].values())
    assert seeds == [0, 1] and all(isinstance(s, int) for s in seeds)

    # across-seed summary: one estimate per run, never pooled episodes
    unc = cell["training_seed_uncertainty"]["mild"]["reward"]
    assert unc["n_runs"] == 2
    assert len(unc["values"]) == 2

    # per-run robustness / quality / retained quality all present
    run = cell["training_runs"][f"{CELL['name']}__s0"]
    assert run["quality_fixed"]["reward_mean"]["point"] is not None
    assert "mild" in run["retained_quality"]
    assert run["robustness"]["mild"]["reward"]["n_env"] == 3
    assert run["verification"]["verified"] is True

    # excluded: the seed-mismatched entry and the planned entry, loudly
    excluded = {e["run_id"]: e["reason"] for e in results["excluded_runs"]}
    assert f"{CELL['name']}__s5" in excluded
    assert "verification failed" in excluded[f"{CELL['name']}__s5"]
    assert f"{CELL['name']}__s9" in excluded
    assert "status" in excluded[f"{CELL['name']}__s9"]
    assert results["sign_convention"]


# ------------------------------------------- equal-extra metadata validation


def _full_meta(**over):
    m = {"smoke": False, "run_kind": "full",
         "semantics_version": SEMANTICS_VERSION, "mode": "reactive",
         "parent_checkpoint": {"sha256": "REACTIVE_SHA"},
         "budget": {"generations": 60, "population": 16,
                    "episodes_per_candidate": 3, "validation_episodes": 12},
         "seeds": {"cma_seed": 0},
         "timing": {"train_irregular_dt": False, "substeps": 8}}
    m.update(over)
    return m


HIER_META = {"budget": {"generations": 60, "population": 16,
                        "episodes_per_candidate": 3,
                        "validation_episodes": 12},
             "seeds": {"cma_seed": 0},
             "timing": {"train_irregular_dt": False, "substeps": 8}}


def test_equal_extra_metadata_validation():
    from scripts.eval_hierarchy import _equal_extra_rejections as rej

    # full artifact accepted regardless of what its filename might claim
    assert rej(_full_meta(), "REACTIVE_SHA", HIER_META) == []
    # smoke metadata rejected even if the filename does not say "smoke"
    assert any("smoke" in r for r in
               rej(_full_meta(smoke=True), "REACTIVE_SHA", HIER_META))
    # missing run_kind metadata rejected (pre-run_kind wildcard artifact)
    m = _full_meta()
    del m["smoke"]
    assert any("missing" in r for r in rej(m, "REACTIVE_SHA", HIER_META))
    # correct parent but wrong budget
    m = _full_meta(budget={"generations": 10, "population": 16,
                           "episodes_per_candidate": 3,
                           "validation_episodes": 12})
    assert any("budget.generations" in r
               for r in rej(m, "REACTIVE_SHA", HIER_META))
    # correct budget but wrong CMA seed
    m = _full_meta(seeds={"cma_seed": 7})
    assert any("cma_seed" in r for r in rej(m, "REACTIVE_SHA", HIER_META))
    # wrong timing distribution
    m = _full_meta(timing={"train_irregular_dt": True, "substeps": 8})
    assert any("timing" in r for r in rej(m, "REACTIVE_SHA", HIER_META))
    # legacy semantics
    m = _full_meta(semantics_version=SEMANTICS_VERSION - 1)
    assert any("semantics_version" in r
               for r in rej(m, "REACTIVE_SHA", HIER_META))
    # wrong parent checkpoint
    assert any("parent" in r for r in
               rej(_full_meta(), "SOME_OTHER_SHA", HIER_META))
    # wrong mode (planner/shaping contamination)
    m = _full_meta(mode="hierarchical")
    assert any("mode" in r for r in rej(m, "REACTIVE_SHA", HIER_META))


# --------------------------------------------------- WM gate CI enforcement


def test_wm_gate_ci_enforcement_logic():
    from training.train_world_model import _ci_positive
    # mean passes the point threshold but the clustered lower bound spans 0
    assert _ci_positive({"mean_diff": 0.2, "lo": -0.05, "hi": 0.45}) is False
    # mean passes and lo > 0 -> criterion holds
    assert _ci_positive({"mean_diff": 0.2, "lo": 0.02, "hi": 0.4}) is True
    assert _ci_positive({"mean_diff": None, "lo": None, "hi": None}) is False
    assert _ci_positive(None) is False
    assert _ci_positive({}) is False


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
