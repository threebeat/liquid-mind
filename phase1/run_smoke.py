"""Phase 1 smoke run (Stage 1).

30 episodes from the legacy buffer (env seeds 10_000..10_079, smoke-reserved),
one data seed, two reservoir seeds, a few epochs. Verifies finite states and
gradients, decreasing loss, checkpoint save/load round-trip, and a
deterministic rerun within tolerance, then writes ONE compact result JSON
under results/specialists_phase1_v1/smoke/ with parameter counts, train/val
losses, all baseline losses, runtime, exact splits, exact seeds, and full
provenance. No gate decision is made here or anywhere else in Phase 1.

Usage: .\\env\\python.exe -m phase1.run_smoke
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import DATA_DIR, load_config
from phase1 import (EXPERIMENT, P1_MODELS, P1_RESULTS, PRIMARY_HORIZON_S,
                    ensure_p1_dirs)
from phase1.dataset import (WindowDataset, make_episode_split,
                            save_split_manifest)
from phase1.experiment import (DEFAULT_ESN_CFG, baseline_metrics,
                               build_matched_gru, build_system, gru_metrics,
                               load_system, message_probes, modular_metrics,
                               param_counts, save_system, system_config_dict)
from phase1.train_eval import (materialize, n_windows, precompute_features,
                               predict_modular, s_ratio, take, train_modular,
                               train_torch_model, partition_consistency)
from provenance import file_checksum, gather_provenance, write_results
from training.replay_buffer import ReplayBuffer

SMOKE = {
    "n_episodes": 30, "n_train": 20, "n_val": 5, "n_test": 5,
    "data_seed": 123, "reservoir_seeds": [0, 1], "model_seed_offset": 900,
    "stride": 2, "epochs": 3, "batch_size": 256, "lr": 1e-3,
    "esn_cfg": dict(DEFAULT_ESN_CFG),
}


def load_smoke_buffer():
    path = os.path.join(DATA_DIR, "experience.npz")
    buf = ReplayBuffer.load(path, allow_legacy_buffer=True)
    return buf, path


def run(log=print) -> str:
    torch.manual_seed(0)
    ensure_p1_dirs()
    t_start = time.perf_counter()
    cfg = load_config()
    checks = {}

    buf, buf_path = load_smoke_buffer()
    buf_sha = file_checksum(buf_path)
    log(f"[smoke] legacy buffer: {len(buf)} episodes ({buf_sha[:12]})")

    # episode-level split, saved manifest (reused by every architecture)
    split = make_episode_split(SMOKE["n_episodes"], SMOKE["n_train"],
                               SMOKE["n_val"], SMOKE["n_test"],
                               seed=SMOKE["data_seed"])
    manifest_path = os.path.join(P1_RESULTS, "smoke", "split_manifest.json")
    save_split_manifest(manifest_path, split, buf_path)

    ds = {name: WindowDataset(buf, split[name], stride=SMOKE["stride"])
          for name in ("train", "val", "test")}
    arrays = {name: materialize(d) for name, d in ds.items()}
    fingerprints = {name: d.fingerprint(buf_sha) for name, d in ds.items()}
    log(f"[smoke] windows: " + ", ".join(f"{k}={len(d)}"
                                         for k, d in ds.items()))

    # ---------------------------------------------------------- baselines
    base, linear_ar = baseline_metrics(
        arrays["train"], arrays["test"],
        max_wheel_speed=float(cfg["env"]["max_wheel_speed"]))
    _, _ = baseline_metrics(arrays["train"], arrays["val"],
                            linear_ar=linear_ar)

    results = {"models": {}, "baselines": base, "checks": checks}

    # --------------------------------------------- modular ESN (2 seeds)
    model_seed_of = lambda rs: SMOKE["model_seed_offset"] + rs
    trainable_ref = None
    for rs in SMOKE["reservoir_seeds"]:
        name = f"modular_nominal_rs{rs}"
        system = build_system(rs, model_seed_of(rs), SMOKE["esn_cfg"])
        info = train_modular(system, arrays["train"], arrays["val"],
                             SMOKE["epochs"], SMOKE["batch_size"],
                             SMOKE["lr"], seed=model_seed_of(rs), log=log)
        feats_test = precompute_features(system, arrays["test"])
        entry = {
            "params": param_counts(system),
            "history": info["history"],
            "health": {k: {kk: vv for kk, vv in v.items()}
                       for k, v in info["health"].items()},
            "wall_clock_s": info["wall_clock_s"],
            "metrics_test": modular_metrics(system, arrays["test"],
                                            feats_test),
            "metrics_val": modular_metrics(system, arrays["val"],
                                           info["features"]["val"]),
            "probes": message_probes(system, info["features"]["train"],
                                     info["features"]["val"],
                                     arrays["train"], arrays["val"]),
            "config": system_config_dict(system),
            "seeds": {"reservoir": rs, "model": model_seed_of(rs),
                      "data": SMOKE["data_seed"]},
        }
        trainable_ref = entry["params"]["trainable"]
        # gradient-isolation check: reservoir buffers never receive grads
        checks[f"{name}_zero_trainable_reservoir"] = bool(
            all(not b.requires_grad for b in system.lidar_esn.buffers())
            and all(not b.requires_grad for b in system.body_esn.buffers()))
        checks[f"{name}_loss_decreased"] = bool(
            info["history"]["train_loss"][-1]
            < info["history"]["train_loss"][0])
        checks[f"{name}_finite"] = bool(
            info["health"]["lidar"]["ok"] and info["health"]["body"]["ok"]
            and all(np.isfinite(info["history"]["train_loss"])))
        # checkpoint round trip
        ck_path = os.path.join(P1_MODELS, f"smoke_{name}.pt")
        ck_sha = save_system(ck_path, system, SMOKE["esn_cfg"],
                             extra={"stage": "smoke", "name": name},
                             force=True)
        reloaded, _ = load_system(ck_path, rs, model_seed_of(rs),
                                  SMOKE["esn_cfg"])
        p0 = predict_modular(system, feats_test, arrays["test"], "rel")
        p1 = predict_modular(reloaded, feats_test, arrays["test"], "rel")
        rt = max(float(np.max(np.abs(p0["lidar"][h] - p1["lidar"][h])))
                 for h in p0["lidar"])
        checks[f"{name}_checkpoint_roundtrip"] = bool(rt < 1e-6)
        entry["checkpoint"] = {"path": ck_path, "sha256": ck_sha,
                               "roundtrip_max_abs_diff": rt}
        results["models"][name] = entry

    # deterministic rerun (seed 0) within tolerance
    rs = SMOKE["reservoir_seeds"][0]
    rerun = build_system(rs, model_seed_of(rs), SMOKE["esn_cfg"])
    info2 = train_modular(rerun, arrays["train"], arrays["val"],
                          SMOKE["epochs"], SMOKE["batch_size"], SMOKE["lr"],
                          seed=model_seed_of(rs), log=log)
    v_a = results["models"][f"modular_nominal_rs{rs}"]["history"]["val_loss"]
    v_b = info2["history"]["val_loss"]
    checks["deterministic_rerun"] = bool(
        max(abs(a - b) for a, b in zip(v_a, v_b)) < 1e-6)

    # ----------------------------------- physical-time leak variant (1 seed)
    name = "modular_physical_rs0"
    sys_phys = build_system(0, model_seed_of(0), SMOKE["esn_cfg"],
                            lidar_leak="physical", body_leak="physical")
    info_p = train_modular(sys_phys, arrays["train"], arrays["val"],
                           SMOKE["epochs"], SMOKE["batch_size"], SMOKE["lr"],
                           seed=model_seed_of(0), log=log)
    results["models"][name] = {
        "params": param_counts(sys_phys),
        "history": info_p["history"],
        "health": info_p["health"],
        "wall_clock_s": info_p["wall_clock_s"],
        "metrics_test": modular_metrics(sys_phys, arrays["test"]),
        "partition_consistency": {
            "lidar": partition_consistency(sys_phys.lidar_esn,
                                           arrays["val"]),
            "body": partition_consistency(sys_phys.body_esn, arrays["val"])},
        "config": system_config_dict(sys_phys),
        "seeds": {"reservoir": 0, "model": model_seed_of(0),
                  "data": SMOKE["data_seed"]},
    }

    # ----------------------------------------- isolated (no fusion, 1 seed)
    name = "modular_isolated_rs0"
    sys_iso = build_system(0, model_seed_of(0), SMOKE["esn_cfg"],
                           fusion=False)
    info_i = train_modular(sys_iso, arrays["train"], arrays["val"],
                           SMOKE["epochs"], SMOKE["batch_size"], SMOKE["lr"],
                           seed=model_seed_of(0), log=log)
    results["models"][name] = {
        "params": param_counts(sys_iso),
        "history": info_i["history"],
        "wall_clock_s": info_i["wall_clock_s"],
        "metrics_test": modular_metrics(sys_iso, arrays["test"]),
        "config": system_config_dict(sys_iso),
        "seeds": {"reservoir": 0, "model": model_seed_of(0),
                  "data": SMOKE["data_seed"]},
    }

    # -------------------------------------------- parameter-matched GRU
    gru = build_matched_gru(trainable_ref, model_seed=model_seed_of(0))
    info_g = train_torch_model(gru, arrays["train"], arrays["val"],
                               SMOKE["epochs"], SMOKE["batch_size"],
                               SMOKE["lr"], seed=model_seed_of(0), log=log)
    results["models"]["gru_matched"] = {
        "params": param_counts(gru), "hidden": gru.hidden,
        "history": info_g["history"],
        "wall_clock_s": info_g["wall_clock_s"],
        "metrics_test": gru_metrics(gru, arrays["test"]),
        "seeds": {"model": model_seed_of(0), "data": SMOKE["data_seed"]},
    }
    checks["gru_param_match_within_5pct"] = bool(
        abs(results["models"]["gru_matched"]["params"]["trainable"]
            - trainable_ref) / trainable_ref < 0.05)

    # ------------------------------------------------- raw-history MLP
    from phase1.baselines import RawHistoryMLP
    mlp = RawHistoryMLP(context_len=15, seed=model_seed_of(0))
    info_m = train_torch_model(mlp, arrays["train"], arrays["val"],
                               SMOKE["epochs"], SMOKE["batch_size"],
                               SMOKE["lr"], seed=model_seed_of(0), log=log)
    from phase1.train_eval import predict_torch_model, metrics_over_horizons
    results["models"]["raw_history_mlp"] = {
        "params": param_counts(mlp),
        "history": info_m["history"],
        "wall_clock_s": info_m["wall_clock_s"],
        "metrics_test": metrics_over_horizons(
            predict_torch_model(mlp, arrays["test"]), arrays["test"]),
        "seeds": {"model": model_seed_of(0), "data": SMOKE["data_seed"]},
    }

    # ------------------------------------------------ headline ratios
    pers = base["persistence"]
    summary = {}
    for name, entry in results["models"].items():
        mt = entry.get("metrics_test")
        if mt is None:
            continue
        flat = mt.get("rel") or mt.get("local") or mt
        summary[name] = {"S_0.5s": s_ratio(flat, pers)}
    iso = results["models"]["modular_isolated_rs0"]["metrics_test"]["local"]
    fused = results["models"]["modular_nominal_rs0"]["metrics_test"]["rel"]
    h = str(PRIMARY_HORIZON_S)
    summary["C_0.5s"] = iso[h]["combined"] - fused[h]["combined"]
    summary["M_0.5s"] = (
        results["models"]["gru_matched"]["metrics_test"][h]["combined"]
        - fused[h]["combined"])
    results["summary"] = summary

    results["provenance"] = gather_provenance(
        {"smoke": SMOKE, "env": cfg["env"]}, experiment_name=EXPERIMENT,
        variant="smoke",
        seeds={"data": SMOKE["data_seed"],
               "reservoir": SMOKE["reservoir_seeds"],
               "model": [model_seed_of(r) for r in SMOKE["reservoir_seeds"]]},
        extra={"buffer": {"path": buf_path, "sha256": buf_sha,
                          "legacy": True, "episodes": len(buf)},
               "split_manifest": manifest_path,
               "split": split,
               "dataset_fingerprints": fingerprints,
               "window_counts": {k: len(d) for k, d in ds.items()}})
    results["total_wall_clock_s"] = time.perf_counter() - t_start
    results["all_checks_passed"] = all(bool(v) for v in checks.values())

    out = write_results("smoke_result", results,
                        results_dir=os.path.join(P1_RESULTS, "smoke"))
    log(f"[smoke] checks: {checks}")
    log(f"[smoke] summary: {summary}")
    log(f"[smoke] wrote {out}")
    return out


if __name__ == "__main__":
    run()
