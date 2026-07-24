"""Phase 1 confirmatory run (Stage 3; conditional on the pilot projection).

Consumes the FROZEN choices from the pilot result JSON (config family, leak
modes, confirmatory n) — nothing is selected here. Collects the confirmatory
episode banks (500/100/200, disjoint reserved seeds), trains n seeds per
architecture (modular fused, isolated for C, parameter-matched GRU) on
identical splits, evaluates ONCE on the 200 untouched test episodes, runs
message-ablation tests (zero / cross-episode shuffle / random-vector
replacement per message), efficiency metrics, and a hierarchical bootstrap
(episodes within seed, seeds as replicates; no window-level
pseudo-replication). Emits the complete evidence package. NO verdict.

Usage: .\\env\\python.exe -m phase1.run_confirmatory [--pilot-json PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import load_config
from phase1 import (EXPERIMENT, HORIZON_STEPS, P1_MODELS, P1_RESULTS,
                    PRIMARY_HORIZON_S, ensure_p1_dirs)
from phase1.collect_data import collect_stage
from phase1.dataset import WindowDataset
from phase1.experiment import (baseline_metrics, build_matched_gru,
                               message_probes, param_counts, save_system)
from phase1.run_pilot import train_fused
from phase1.train_eval import (episode_ids_for, materialize,
                               metrics_over_horizons, n_windows,
                               partition_consistency, per_episode_metrics,
                               precompute_features, predict_modular,
                               predict_torch_model, take, train_torch_model)
from provenance import gather_provenance, write_results
from training.replay_buffer import ReplayBuffer

H = str(PRIMARY_HORIZON_S)
H_STEPS = HORIZON_STEPS[PRIMARY_HORIZON_S]

CONF = {
    "stride": {"train": 5, "val": 5, "test": 5},
    "epochs": 10, "batch_size": 256, "lr": 1e-3,
    "reservoir_seed_base": 200,     # disjoint from pilot's 100..104
    "model_seed_base": 2200,
    "n_bootstrap": 2000,
}


def latest_pilot_json() -> str:
    paths = sorted(glob.glob(os.path.join(P1_RESULTS, "pilot",
                                          "pilot_result_*.json")))
    if not paths:
        raise FileNotFoundError("no pilot result JSON; run Stage 2 first")
    return paths[-1]


# --------------------------------------------------------- message ablation


@torch.no_grad()
def relational_combined_loss(system, m_l, m_b, arrays,
                             batch_size: int = 1024) -> float:
    """Relational combined loss at 0.5 s given (possibly ablated) messages."""
    n = n_windows(arrays)
    fut_a = torch.from_numpy(arrays["future_actions"])
    fut_d = torch.from_numpy(arrays["future_dts"])
    tl = torch.from_numpy(arrays["future_lidar"][:, H_STEPS - 1])
    tb = torch.from_numpy(arrays["future_body"][:, H_STEPS - 1])
    losses, weights = [], []
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        act = system.action_encoder.at_horizons(fut_a[idx], fut_d[idx],
                                                system.horizon_steps)
        rel = system.relational(m_l[idx], m_b[idx], act)[PRIMARY_HORIZON_S]
        lm = torch.mean((rel[:, :16] - tl[idx]) ** 2)
        bm = torch.mean((rel[:, 16:] - tb[idx]) ** 2)
        losses.append(float(0.5 * lm + 0.5 * bm))
        weights.append(len(idx))
    return float(np.average(losses, weights=weights))


@torch.no_grad()
def message_ablations(system, feats, arrays, episode_ids, seed=0) -> dict:
    """Zero / cross-episode shuffle / moment-matched random replacement of
    each message stream; reports relational combined loss at 0.5 s."""
    system.eval()
    m_l = system.lidar_esn.message_from_features(feats["lidar"])
    m_b = system.body_esn.message_from_features(feats["body"])
    rng = np.random.default_rng(seed)
    n = m_l.shape[0]

    def cross_episode_perm():
        """Permutation where every window receives a message from a
        DIFFERENT episode (rejection resampling on the offenders)."""
        perm = rng.permutation(n)
        for _ in range(200):
            same = np.where(episode_ids[perm] == episode_ids)[0]
            if len(same) == 0:
                break
            perm[same] = rng.choice(n, size=len(same))
        return perm

    def rand_like(m):
        mu = m.mean(dim=0, keepdim=True)
        sd = m.std(dim=0, keepdim=True)
        g = torch.from_numpy(
            rng.standard_normal(tuple(m.shape)).astype(np.float32))
        return mu + sd * g

    out = {"intact": relational_combined_loss(system, m_l, m_b, arrays)}
    out["zero_m_lidar"] = relational_combined_loss(
        system, torch.zeros_like(m_l), m_b, arrays)
    out["zero_m_body"] = relational_combined_loss(
        system, m_l, torch.zeros_like(m_b), arrays)
    p = cross_episode_perm()
    out["shuffle_m_lidar"] = relational_combined_loss(
        system, m_l[p], m_b, arrays)
    p = cross_episode_perm()
    out["shuffle_m_body"] = relational_combined_loss(
        system, m_l, m_b[p], arrays)
    out["random_m_lidar"] = relational_combined_loss(
        system, rand_like(m_l), m_b, arrays)
    out["random_m_body"] = relational_combined_loss(
        system, m_l, rand_like(m_b), arrays)
    for k in list(out):
        if k != "intact":
            out[f"delta_{k}"] = out[k] - out["intact"]
    return out


# ------------------------------------------------------------- bootstraps


def hierarchical_bootstrap(per_seed_ep_losses: list[dict],
                           pers_ep_losses: dict, n_boot: int,
                           seed: int = 0) -> dict:
    """S with episodes resampled within seed, seeds as replicates.

    per_seed_ep_losses: one {episode_id: combined_loss} per training seed;
    pers_ep_losses: {episode_id: persistence combined loss}.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(per_seed_ep_losses)
    eps = [np.array(sorted(d)) for d in per_seed_ep_losses]
    draws = []
    for _ in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        s_vals = []
        for si in seed_idx:
            ep_ids = rng.choice(eps[si], size=len(eps[si]), replace=True)
            model = np.mean([per_seed_ep_losses[si][int(e)] for e in ep_ids])
            pers = np.mean([pers_ep_losses[int(e)] for e in ep_ids])
            s_vals.append(model / pers)
        draws.append(float(np.mean(s_vals)))
    draws = np.asarray(draws)
    return {"mean": float(draws.mean()),
            "ci95": [float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5))],
            "n_boot": n_boot, "n_seeds": n_seeds}


def paired_diff_bootstrap(a_losses: list[dict], b_losses: list[dict],
                          pers_ep_losses: dict, n_boot: int,
                          seed: int = 0) -> dict:
    """Hierarchical bootstrap of the paired per-seed S difference a - b
    (seeds paired by index, episodes resampled within seed)."""
    rng = np.random.default_rng(seed)
    n_seeds = len(a_losses)
    eps = [np.array(sorted(d)) for d in a_losses]
    draws = []
    for _ in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        d_vals = []
        for si in seed_idx:
            ep_ids = rng.choice(eps[si], size=len(eps[si]), replace=True)
            pers = np.mean([pers_ep_losses[int(e)] for e in ep_ids])
            sa = np.mean([a_losses[si][int(e)] for e in ep_ids]) / pers
            sb = np.mean([b_losses[si][int(e)] for e in ep_ids]) / pers
            d_vals.append(sa - sb)
        draws.append(float(np.mean(d_vals)))
    draws = np.asarray(draws)
    return {"mean": float(draws.mean()),
            "ci95": [float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5))],
            "n_boot": n_boot, "n_seeds": n_seeds}


def per_episode_persistence(arrays: dict, episode_ids: np.ndarray) -> dict:
    tl = arrays["future_lidar"][:, H_STEPS - 1]
    tb = arrays["future_body"][:, H_STEPS - 1]
    pl, pb = arrays["lidar"][:, -1], arrays["body"][:, -1]
    out = {}
    for ep in np.unique(episode_ids):
        m = episode_ids == ep
        lm = float(np.mean((pl[m] - tl[m]) ** 2))
        bm = float(np.mean((pb[m] - tb[m]) ** 2))
        out[int(ep)] = 0.5 * lm + 0.5 * bm
    return out


# ------------------------------------------------------------------- main


def run(pilot_json: str | None = None, log=print) -> str:
    torch.manual_seed(0)
    ensure_p1_dirs()
    t_start = time.perf_counter()
    cfg = load_config()

    pilot_path = pilot_json or latest_pilot_json()
    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot = json.load(f)
    chosen_cfg = pilot["chosen_config"]
    lidar_leak = pilot["chosen_leak"]["lidar"]
    body_leak = pilot["chosen_leak"]["body"]
    n_seeds = int(pilot["endpoints"]["confirmatory_n"])
    log(f"[conf] frozen config={chosen_cfg} leak=({lidar_leak},{body_leak}) "
        f"n={n_seeds} (from {os.path.basename(pilot_path)})")

    results = {"stage": "confirmatory", "pilot_json": pilot_path,
               "frozen": {"config": chosen_cfg,
                          "leak": pilot["chosen_leak"],
                          "n_seeds": n_seeds},
               "exclusions": [], "timing": {}}

    # --------------------------------------------------------------- data
    t0 = time.perf_counter()
    buffers = collect_stage("confirmatory", cfg, log=log)
    results["buffers"] = buffers
    results["timing"]["collection_s"] = time.perf_counter() - t0

    bufs = {k: ReplayBuffer.load(v["path"]) for k, v in buffers.items()}
    ds = {k: WindowDataset(bufs[k], range(len(bufs[k])),
                           stride=CONF["stride"][k])
          for k in ("train", "val", "test")}
    arrays = {k: materialize(d) for k, d in ds.items()}
    results["window_counts"] = {k: len(d) for k, d in ds.items()}
    results["dataset_fingerprints"] = {
        k: d.fingerprint(buffers[k]["sha256"]) for k, d in ds.items()}
    log(f"[conf] windows: {results['window_counts']}")
    test_ep_ids = episode_ids_for(ds["test"])
    pers_ep = per_episode_persistence(arrays["test"], test_ep_ids)

    # ---------------------------------------------------------- baselines
    max_wheel = float(cfg["env"]["max_wheel_speed"])
    base_test, linear_ar = baseline_metrics(arrays["train"], arrays["test"],
                                            max_wheel_speed=max_wheel)
    results["baselines"] = {"test": base_test}
    pers_test = base_test["persistence"]
    pers_L = pers_test[H]["combined"]

    seeds = [(CONF["reservoir_seed_base"] + i, CONF["model_seed_base"] + i)
             for i in range(n_seeds)]
    results["seeds"] = {"pairs": seeds}
    per_model: dict = {}
    ep_losses = {"fused": [], "isolated": [], "gru": []}
    trainable_ref = None

    # ------------------------------------------- modular fused + isolated
    for kind in ("fused", "isolated"):
        t0 = time.perf_counter()
        entries = []
        for rs, ms in seeds:
            r = train_fused(chosen_cfg, lidar_leak, body_leak,
                            kind == "fused", rs, ms, arrays,
                            log=lambda *a: None,
                            epochs=CONF["epochs"],
                            batch_size=CONF["batch_size"], lr=CONF["lr"])
            results["exclusions"].extend(r["exclusions"])
            system, info = r["system"], r["info"]
            feats_test = precompute_features(system, arrays["test"])
            stream = "rel" if kind == "fused" else "local"
            preds = predict_modular(system, feats_test, arrays["test"],
                                    stream)
            mtest = metrics_over_horizons(preds, arrays["test"])
            per_ep = per_episode_metrics(preds, arrays["test"], test_ep_ids)
            ep_losses[kind].append(per_ep)
            entry = {
                "seeds": {"reservoir": r["reservoir_seed_used"], "model": ms},
                "params": param_counts(system),
                "history": info["history"],
                "wall_clock_s": info["wall_clock_s"],
                "metrics_test": mtest,
                "test_S": mtest[H]["combined"] / pers_L,
                "per_episode_combined_0.5s": per_ep,
                "health": info["health"],
            }
            if kind == "fused":
                trainable_ref = entry["params"]["trainable"]
                entry["probes"] = message_probes(
                    system, info["features"]["train"],
                    info["features"]["val"], arrays["train"], arrays["val"])
                entry["message_ablation"] = message_ablations(
                    system, feats_test, arrays["test"], test_ep_ids,
                    seed=ms)
                if "physical" in (lidar_leak, body_leak):
                    entry["partition_consistency"] = {
                        "lidar": partition_consistency(system.lidar_esn,
                                                       arrays["val"]),
                        "body": partition_consistency(system.body_esn,
                                                      arrays["val"])}
                ck = os.path.join(P1_MODELS, f"conf_fused_rs{rs}.pt")
                entry["checkpoint"] = {
                    "path": ck,
                    "sha256": save_system(
                        ck, system,
                        {"esn": chosen_cfg, "leak": pilot["chosen_leak"]},
                        extra={"stage": "confirmatory",
                               "seeds": entry["seeds"]}, force=True)}
            entries.append(entry)
            log(f"[conf] {kind} rs={rs}: test S={entry['test_S']:.4f} "
                f"({info['wall_clock_s']:.0f}s)")
        per_model[f"modular_{kind}"] = entries
        results["timing"][f"{kind}_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------- GRU
    t0 = time.perf_counter()
    entries = []
    for rs, ms in seeds:
        gru = build_matched_gru(trainable_ref, model_seed=ms)
        info = train_torch_model(gru, arrays["train"], arrays["val"],
                                 CONF["epochs"], CONF["batch_size"],
                                 CONF["lr"], seed=ms, log=lambda *a: None)
        preds = predict_torch_model(gru, arrays["test"])
        mtest = metrics_over_horizons(preds, arrays["test"])
        per_ep = per_episode_metrics(preds, arrays["test"], test_ep_ids)
        ep_losses["gru"].append(per_ep)
        ck = os.path.join(P1_MODELS, f"conf_gru_ms{ms}.pt")
        from provenance import gather_provenance as _gp, save_checkpoint
        sha = save_checkpoint(
            ck, gru.state_dict(),
            _gp({"hidden": gru.hidden}, experiment_name=EXPERIMENT,
                extra={"stage": "confirmatory", "model_seed": ms}),
            {"experiment": EXPERIMENT, "hidden": gru.hidden}, force=True)
        entries.append({
            "seeds": {"model": ms}, "hidden": gru.hidden,
            "params": param_counts(gru),
            "history": info["history"],
            "wall_clock_s": info["wall_clock_s"],
            "metrics_test": mtest,
            "test_S": mtest[H]["combined"] / pers_L,
            "per_episode_combined_0.5s": per_ep,
            "checkpoint": {"path": ck, "sha256": sha}})
        log(f"[conf] gru ms={ms}: test S={entries[-1]['test_S']:.4f} "
            f"({info['wall_clock_s']:.0f}s)")
    per_model["gru_matched"] = entries
    results["timing"]["gru_s"] = time.perf_counter() - t0
    results["models"] = per_model

    # ------------------------------------------------------ efficiency
    def _time_forward(fn, n_repeat=50):
        fn()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            fn()
        return (time.perf_counter() - t0) / n_repeat * 1e3

    one = take(arrays["test"], np.arange(1))
    sys_f = per_model["modular_fused"]
    last_fused = train_fused(chosen_cfg, lidar_leak, body_leak, True,
                             seeds[0][0], seeds[0][1], arrays,
                             log=lambda *a: None, epochs=1,
                             batch_size=CONF["batch_size"],
                             lr=CONF["lr"])["system"]
    gru0 = build_matched_gru(trainable_ref, model_seed=seeds[0][1])
    results["efficiency"] = {
        "modular_forward_ms_per_window": _time_forward(
            lambda: last_fused(one)),
        "gru_forward_ms_per_window": _time_forward(lambda: gru0(one)),
        "modular_train_wall_clock_s": [e["wall_clock_s"] for e in sys_f],
        "gru_train_wall_clock_s": [e["wall_clock_s"]
                                   for e in per_model["gru_matched"]],
        "params": {"modular": sys_f[0]["params"],
                   "gru": per_model["gru_matched"][0]["params"]},
    }

    # ------------------------------------------------- endpoints + boots
    s_esn = [e["test_S"] for e in per_model["modular_fused"]]
    s_iso = [e["test_S"] for e in per_model["modular_isolated"]]
    s_gru = [e["test_S"] for e in per_model["gru_matched"]]
    c_vals = [(i - f) * pers_L for f, i in zip(s_esn, s_iso)]
    m_vals = [(g - f) * pers_L for f, g in zip(s_esn, s_gru)]
    results["endpoints"] = {
        "persistence_combined_0.5s_test": pers_L,
        "S_modular_per_seed": s_esn, "S_modular_mean": float(np.mean(s_esn)),
        "S_modular_std": float(np.std(s_esn, ddof=1)),
        "S_isolated_per_seed": s_iso,
        "S_isolated_mean": float(np.mean(s_iso)),
        "S_gru_per_seed": s_gru, "S_gru_mean": float(np.mean(s_gru)),
        "S_gru_std": float(np.std(s_gru, ddof=1)),
        "C_per_seed": c_vals, "C_mean": float(np.mean(c_vals)),
        "M_per_seed": m_vals, "M_mean": float(np.mean(m_vals)),
        "paired_S_diff_per_seed": [f - g for f, g in zip(s_esn, s_gru)],
    }
    nb = CONF["n_bootstrap"]
    results["bootstrap"] = {
        "S_modular": hierarchical_bootstrap(ep_losses["fused"], pers_ep,
                                            nb, seed=1),
        "S_isolated": hierarchical_bootstrap(ep_losses["isolated"], pers_ep,
                                             nb, seed=2),
        "S_gru": hierarchical_bootstrap(ep_losses["gru"], pers_ep, nb,
                                        seed=3),
        "S_modular_minus_S_gru": paired_diff_bootstrap(
            ep_losses["fused"], ep_losses["gru"], pers_ep, nb, seed=4),
        "S_isolated_minus_S_modular": paired_diff_bootstrap(
            ep_losses["isolated"], ep_losses["fused"], pers_ep, nb, seed=5),
    }

    results["provenance"] = gather_provenance(
        {"conf": CONF, "env": cfg["env"]}, experiment_name=EXPERIMENT,
        variant="confirmatory",
        seeds={"pairs": seeds})
    results["total_wall_clock_s"] = time.perf_counter() - t_start
    out = write_results("confirmatory_result", results,
                        results_dir=os.path.join(P1_RESULTS, "confirmatory"))
    log(f"[conf] endpoints: {results['endpoints']}")
    log(f"[conf] bootstrap: {results['bootstrap']}")
    log(f"[conf] wrote {out} ({results['total_wall_clock_s']:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-json", default=None)
    a = ap.parse_args()
    run(pilot_json=a.pilot_json)
