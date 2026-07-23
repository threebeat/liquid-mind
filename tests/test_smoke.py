"""End-to-end smoke tests: minutes-scale mini training runs and the
world-model gate machinery. These prove the pipeline executes before any
expensive run is started (execution-order Stage 1)."""
import copy

import numpy as np
import pytest
import torch

from common import load_config
from provenance import gather_provenance, save_checkpoint
from agent.hybrid_agent import HybridAgent
from agent.world_model import WorldModel
from environment.nav_env import OBS_DIM, NavEnv
from training.replay_buffer import ReplayBuffer


def small_cfg():
    cfg = copy.deepcopy(load_config())
    cfg["env"]["episode_seconds"] = 3.0
    cfg["training"].update({"cma_generations": 1, "cma_population": 4,
                            "episodes_per_candidate": 1,
                            "validation_episodes": 1, "workers": 1,
                            "baseline_n_envs": 1})
    return cfg


def synthetic_buffer(n_eps=6, T=300, seed=0):
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer()
    for _ in range(n_eps):
        obs = np.zeros((T + 1, OBS_DIM), np.float32)
        obs[0] = rng.uniform(0, 1, OBS_DIM)
        for t in range(T):
            obs[t + 1] = np.clip(obs[t] + rng.normal(0, 0.02, OBS_DIM), 0, 1)
        buf.add_episode(obs, rng.uniform(-1, 1, (T, 2)),
                        rng.uniform(0.02, 0.06, T))
    return buf


# --------------------------------------------------------------- replay


def test_duration_based_chunks():
    buf = synthetic_buffer()
    rng = np.random.default_rng(0)
    o, a, d, o2 = buf.sample_chunks_by_duration(32, 0.5, rng)
    assert o.shape == (32, OBS_DIM) and o2.shape == (32, OBS_DIM)
    assert a.shape == (32, 2)
    assert (d >= 0.5 - 1e-6).all(), "chunk shorter than target duration"
    assert (d <= 0.5 + 0.06 + 1e-6).all(), "chunk overshoot beyond one event"


def test_replay_records_event_times(tmp_path):
    buf = synthetic_buffer(n_eps=1, T=10)
    e = buf.episodes[0]
    assert np.allclose(e["t_capture"][1:], np.cumsum(e["dts"]))
    assert np.allclose(e["t_delivery"], e["t_capture"])
    p = str(tmp_path / "buf.npz")
    buf.save(p)
    buf2 = ReplayBuffer.load(p)
    assert np.allclose(buf2.episodes[0]["t_capture"], e["t_capture"])
    assert buf2.meta.get("semantics_version") is not None


def test_replay_refuses_legacy_without_override(tmp_path):
    # Pre-provenance layout: no meta_json
    p = str(tmp_path / "legacy.npz")
    np.savez_compressed(
        p, n=1,
        obs_0=np.zeros((5, OBS_DIM), np.float32),
        act_0=np.zeros((4, 2), np.float32),
        dt_0=np.ones(4, np.float32) * 0.05)
    with pytest.raises(ValueError) as e:
        ReplayBuffer.load(p)
    assert "provenance" in str(e.value).lower() or "metadata" in str(e.value)
    buf = ReplayBuffer.load(p, allow_legacy_buffer=True)
    assert len(buf) == 1
    assert buf.meta.get("legacy_buffer") is True


def test_wm_gate_and_enforcement(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["agent"]["latent_dim"] = 8
    cfg["world_model"]["hidden_dim"] = 32
    buf = synthetic_buffer()
    # Enrich with near-obstacle readings so danger / L-R cases exist
    for e in buf.episodes:
        e["obs"][:, :16] = 0.05  # dangerous rays
        e["obs"][:, 4] = 0.5     # left clear
        e["obs"][:, 12] = 0.01   # right near
    wm = WorldModel(OBS_DIM, 8, hidden=32)
    optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(30):
        o, a, d, o2 = buf.sample_chunks_by_duration(64, 0.5, rng)
        loss, _ = wm.loss(torch.from_numpy(o), torch.from_numpy(a),
                          torch.from_numpy(d), torch.from_numpy(o2))
        optim.zero_grad()
        loss.backward()
        optim.step()
        wm.update_target()
    wm.eval()

    from training.train_world_model import (evaluate_gate, load_world_model,
                                            wm_compat)
    gate = evaluate_gate(wm, buf, 0.5, cfg, n_windows=40)
    assert gate["n_windows"] > 0
    assert "status" in gate
    assert gate["status"] in ("passed", "failed", "incomplete")
    assert set(gate["criteria"]) >= {"goal_beats_persistence_2s",
                                     "false_safe_not_worse_2s",
                                     "goal_beats_or_matches_kinematic_2s",
                                     "open_loop_stable"}
    for h in ("0.5", "1.0", "2.0", "4.0"):
        m = gate["horizons"][h]
        assert m["goal_dist_mae_m"]["model"] is not None
        assert m["goal_dist_mae_m"]["persist"] is not None
        assert m["goal_dist_mae_m"]["kin"] is not None

    # Synthetic buffer with no danger -> incomplete, never passed on safety
    safe_buf = synthetic_buffer(seed=1)
    for e in safe_buf.episodes:
        e["obs"][:, :16] = 0.9
    gate_safe = evaluate_gate(wm, safe_buf, 0.5, cfg, n_windows=30)
    assert gate_safe["status"] == "incomplete"
    assert gate_safe["passed"] is False
    assert gate_safe["criteria"]["false_safe_not_worse_2s"] is None

    # gate enforcement: a FAILED model is refused unless overridden
    path = str(tmp_path / "wm.pt")
    failed = dict(gate)
    failed["passed"] = False
    failed["status"] = "failed"
    meta = gather_provenance(cfg, "smoke_wm", extra={"gate": failed})
    save_checkpoint(path, wm.state_dict(), meta, wm_compat(cfg))
    with pytest.raises(ValueError) as e:
        load_world_model(cfg, path=path)
    assert "gate" in str(e.value)
    wm2, _ = load_world_model(cfg, path=path, override_gate=True)
    assert isinstance(wm2, WorldModel)


# ------------------------------------------------ physical-time replanning


def test_planner_period_is_physical_time():
    """Replanning happens every period_seconds of ELAPSED time, with the
    residual carried across the boundary — not every N decisions."""
    cfg = copy.deepcopy(load_config())
    cfg["planner"]["period_seconds"] = 0.5
    torch.manual_seed(0)
    wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                    hidden=int(cfg["world_model"]["hidden_dim"]))
    wm.eval()
    agent = HybridAgent(cfg, mode="hierarchical", world_model=wm)
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, OBS_DIM).astype(np.float32)
    # causal: first act at elapsed 0 -> plan #1
    agent.act(obs, 0.123)
    assert agent._plan_count == 1
    for _ in range(3):                    # t = 0.2, 0.4, 0.6
        agent.act(obs, 0.2)
    assert agent._plan_count == 2         # crossed 0.5 at t=0.6, residual 0.1
    agent.act(obs, 0.2)                   # 0.3
    agent.act(obs, 0.2)                   # 0.5 -> plan #3
    assert agent._plan_count == 3


# ------------------------------------------------------------- mini training


def test_cma_smoke(tmp_path, monkeypatch):
    """One tiny CMA-ES generation end to end, with provenance metadata and
    validation-best selection."""
    import training.train_policy as tp
    monkeypatch.setattr(tp, "MODELS_DIR", str(tmp_path))
    cfg = small_cfg()
    val = tp.train(generations=1, hierarchical=False, workers=1, config=cfg,
                   experiment_name="pytest_smoke_policy", seed=0)
    assert isinstance(val, float)
    from provenance import load_checkpoint
    ck = tmp_path / "pytest_smoke_policy.pt"
    assert ck.exists()
    state, meta = load_checkpoint(str(ck), expected_compat=None)
    assert meta["experiment"] == "pytest_smoke_policy"
    assert meta["selection"] == "validation_best"
    assert meta["parameter_count"] > 0
    assert meta["budget"]["generations"] == 1
    assert "history" in meta and len(meta["history"]) == 1


def test_ppo_smoke(tmp_path, monkeypatch):
    import training.train_baseline as tb
    monkeypatch.setattr(tb, "MODELS_DIR", str(tmp_path))
    mean_r = tb.train(timesteps=64, config=small_cfg())
    assert isinstance(mean_r, float)
    assert (tmp_path / "ppo_baseline.zip").exists()
    assert (tmp_path / "ppo_baseline.zip.meta.json").exists()


def test_eval_records_smoke():
    from scripts.eval_common import compare, run_episode, summarize
    cfg = small_cfg()
    env = NavEnv(cfg, irregular_dt=True)
    rng = np.random.default_rng(0)
    recs = [run_episode(env, lambda o, dt: rng.uniform(-1, 1, 2),
                        lambda: None, seed) for seed in (0, 1, 2)]
    env.close()
    for r in recs:
        assert r["termination"] in ("success", "time_limit", "step_cap")
        assert r["sim_duration"] > 0
        assert r["path_length"] >= 0
        assert 0 <= r["path_efficiency"]
    s = summarize(recs)
    assert s["episodes"] == 3
    assert s["reward_mean"]["lo"] <= s["reward_mean"]["point"] \
           <= s["reward_mean"]["hi"]
    c = compare(recs, recs, "a", "b")
    assert c["reward_diff"]["mean_diff"] == 0.0
    assert c["success_mcnemar"]["discordant"] == 0
