"""Phase 1 pilot (Stage 2).

Fresh disjoint-seed episodes (300 train / 75 val / 100 pilot-test). Bounded
ESN configuration selection exactly as preregistered
(docs/phase1_preregistration.md §5):

  A. ridge screen of all 54 grid combinations, 2 reservoir seeds each,
     ranked by validation combined loss at 0.5 s;
  B. top 3 retrained with the full Adam pipeline, 5 reservoir/model seeds,
     family chosen by mean validation S and FROZEN;
  C. 2x2 leak-mode ablation (nominal/physical x lidar/body) at the frozen
     family, 5 seeds, per-stream choice frozen.

Then 5-seed runs of: fused modular system (frozen config), isolated (fusion
off), parameter-matched GRU, raw-history MLP. Test-split evaluation happens
ONCE, after every selection is frozen on validation. Outputs pilot JSON with
per-model metrics, per-episode outcomes, timing, paired sigma_d and the
computed confirmatory n. No verdict is rendered.

Usage: .\\env\\python.exe -m phase1.run_pilot
"""
from __future__ import annotations

import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import load_config
from phase1 import (EXPERIMENT, P1_MODELS, P1_RESULTS, PRIMARY_HORIZON_S,
                    ensure_p1_dirs)
from phase1.baselines import RawHistoryMLP
from phase1.collect_data import collect_stage
from phase1.dataset import WindowDataset
from phase1.experiment import (baseline_metrics, build_matched_gru,
                               build_system, gru_metrics, message_probes,
                               modular_metrics, param_counts, save_system,
                               system_config_dict)
from phase1.gru_baseline import count_trainable
from phase1.train_eval import (episode_ids_for, materialize,
                               metrics_over_horizons, n_windows,
                               partition_consistency, per_episode_metrics,
                               precompute_features, predict_modular,
                               predict_torch_model, s_ratio, take,
                               train_modular, train_torch_model)
from provenance import file_checksum, gather_provenance, write_results
from training.replay_buffer import ReplayBuffer

H = str(PRIMARY_HORIZON_S)

PILOT = {
    "stride": {"train": 5, "val": 5, "test": 5},
    "epochs": 10, "batch_size": 256, "lr": 1e-3,
    "reservoir_seeds": [100, 101, 102, 103, 104],
    "model_seeds": [1100, 1101, 1102, 1103, 1104],
    "screen_seeds": [100, 101],
    "screen_train_windows": 8000,
    "screen_val_windows": 6000,
    "screen_lam": 1e-2,
    "top_k": 3,
    "delta": 0.05,        # prereg sample-size rule on S
    "min_n": 10,
}

GRID = [dict(spectral_radius=sr, alpha0=a, sparsity=sp, input_scale=isc)
        for sr, a, sp, isc in itertools.product(
            (0.7, 0.9, 1.1), (0.1, 0.3, 0.6), (0.05, 0.1), (0.1, 0.5, 1.0))]


# ------------------------------------------------------------ ridge screen


def _screen_features(system, arrays, steps) -> tuple[np.ndarray, np.ndarray]:
    feats = precompute_features(system, arrays)
    B = n_windows(arrays)
    x = np.concatenate([
        feats["lidar"].numpy(), feats["body"].numpy(),
        arrays["future_actions"][:, :steps].reshape(B, -1),
        arrays["future_dts"][:, :steps].reshape(B, -1),
        np.ones((B, 1), np.float32)], axis=1)
    y = np.concatenate([arrays["future_lidar"][:, steps - 1],
                        arrays["future_body"][:, steps - 1]], axis=1)
    return x.astype(np.float64), y.astype(np.float64)


def screen_grid(train_arrays, val_arrays, pers_val_combined, log=print):
    from phase1 import HORIZON_STEPS
    steps = HORIZON_STEPS[PRIMARY_HORIZON_S]
    rng = np.random.default_rng(7)
    tr_idx = rng.choice(n_windows(train_arrays),
                        min(PILOT["screen_train_windows"],
                            n_windows(train_arrays)), replace=False)
    va_idx = rng.choice(n_windows(val_arrays),
                        min(PILOT["screen_val_windows"],
                            n_windows(val_arrays)), replace=False)
    sub_tr = {k: v[tr_idx] for k, v in train_arrays.items()}
    sub_va = {k: v[va_idx] for k, v in val_arrays.items()}
    rows = []
    t0 = time.perf_counter()
    for gi, gcfg in enumerate(GRID):
        losses = []
        for rs in PILOT["screen_seeds"]:
            system = build_system(rs, 0, gcfg)
            x_tr, y_tr = _screen_features(system, sub_tr, steps)
            x_va, y_va = _screen_features(system, sub_va, steps)
            from phase1.baselines import mm, ridge_solve
            w = ridge_solve(x_tr, y_tr, PILOT["screen_lam"])
            pred = mm(x_va, w)
            lm = float(np.mean((pred[:, :16] - y_va[:, :16]) ** 2))
            bm = float(np.mean((pred[:, 16:] - y_va[:, 16:]) ** 2))
            losses.append(0.5 * lm + 0.5 * bm)
        rows.append({"config": gcfg, "val_combined_0.5s": losses,
                     "mean": float(np.mean(losses)),
                     "S_screen": float(np.mean(losses) / pers_val_combined)})
        if (gi + 1) % 9 == 0:
            log(f"[pilot-screen] {gi + 1}/{len(GRID)} configs "
                f"({time.perf_counter() - t0:.0f}s)")
    rows.sort(key=lambda r: r["mean"])
    return rows


# --------------------------------------------------------------- training


def train_fused(gcfg: dict, lidar_leak: str, body_leak: str, fusion: bool,
                rs: int, ms: int, arrays: dict, log=print,
                epochs: int | None = None, batch_size: int | None = None,
                lr: float | None = None) -> dict:
    """One modular run with preregistered exclusion/regeneration rules."""
    exclusions = []
    seed_shift = 0
    while True:
        system = build_system(rs + seed_shift, ms, gcfg,
                              lidar_leak=lidar_leak, body_leak=body_leak,
                              fusion=fusion)
        info = train_modular(system, arrays["train"], arrays["val"],
                             epochs or PILOT["epochs"],
                             batch_size or PILOT["batch_size"],
                             lr or PILOT["lr"], seed=ms, log=log)
        failed = (info["history"]["status"] != "ok"
                  or not info["health"]["lidar"]["ok"]
                  or not info["health"]["body"]["ok"])
        if not failed:
            break
        exclusions.append({
            "reservoir_seed": rs + seed_shift,
            "status": info["history"]["status"],
            "health": {k: v["failures"] for k, v in info["health"].items()}})
        log(f"[pilot] EXCLUSION at reservoir seed {rs + seed_shift}: "
            f"{exclusions[-1]}; regenerating with seed + 1000")
        seed_shift += 1000
        if seed_shift > 5000:
            raise RuntimeError("more than 5 consecutive reservoir "
                               "regenerations; investigate")
    return {"system": system, "info": info, "exclusions": exclusions,
            "reservoir_seed_used": rs + seed_shift}


def val_S(system, run_info, arrays, pers_val) -> float:
    stream = "rel" if system.fusion else "local"
    m = metrics_over_horizons(
        predict_modular(system, run_info["features"]["val"], arrays["val"],
                        stream), arrays["val"])
    return m[H]["combined"] / pers_val[H]["combined"]


# ------------------------------------------------------------------- main


def run(log=print) -> str:
    torch.manual_seed(0)
    ensure_p1_dirs()
    t_start = time.perf_counter()
    cfg = load_config()
    results = {"stage": "pilot", "exclusions": [], "timing": {}}

    # ------------------------------------------------------ data (reused)
    t0 = time.perf_counter()
    buffers = collect_stage("pilot", cfg, log=log)
    results["buffers"] = buffers
    results["timing"]["collection_s"] = time.perf_counter() - t0

    bufs = {k: ReplayBuffer.load(v["path"]) for k, v in buffers.items()}
    ds = {k: WindowDataset(bufs[k], range(len(bufs[k])),
                           stride=PILOT["stride"][k])
          for k in ("train", "val", "test")}
    arrays = {k: materialize(d) for k, d in ds.items()}
    results["window_counts"] = {k: len(d) for k, d in ds.items()}
    results["dataset_fingerprints"] = {
        k: d.fingerprint(buffers[k]["sha256"]) for k, d in ds.items()}
    log(f"[pilot] windows: {results['window_counts']}")

    # ---------------------------------------------------------- baselines
    t0 = time.perf_counter()
    max_wheel = float(cfg["env"]["max_wheel_speed"])
    base_val, linear_ar = baseline_metrics(arrays["train"], arrays["val"],
                                           max_wheel_speed=max_wheel)
    base_test, _ = baseline_metrics(arrays["train"], arrays["test"],
                                    max_wheel_speed=max_wheel,
                                    linear_ar=linear_ar)
    results["baselines"] = {"val": base_val, "test": base_test}
    results["timing"]["baselines_s"] = time.perf_counter() - t0
    pers_val, pers_test = base_val["persistence"], base_test["persistence"]

    # ------------------------------------------------- stage A: screening
    t0 = time.perf_counter()
    screen = screen_grid(arrays["train"], arrays["val"],
                         pers_val[H]["combined"], log=log)
    results["screen"] = screen
    results["timing"]["screen_s"] = time.perf_counter() - t0
    top = [r["config"] for r in screen[:PILOT["top_k"]]]
    log(f"[pilot] top-{PILOT['top_k']} configs: {top}")

    # --------------------------------------- stage B: top-k Adam, 5 seeds
    t0 = time.perf_counter()
    family_rows = []
    for ci, gcfg in enumerate(top):
        seeds_S, wall = [], []
        for rs, ms in zip(PILOT["reservoir_seeds"], PILOT["model_seeds"]):
            r = train_fused(gcfg, "nominal", "nominal", True, rs, ms,
                            arrays, log=lambda *a: None)
            results["exclusions"].extend(r["exclusions"])
            seeds_S.append(val_S(r["system"], r["info"], arrays, pers_val))
            wall.append(r["info"]["wall_clock_s"])
        family_rows.append({"config": gcfg,
                            "val_S_per_seed": seeds_S,
                            "mean_val_S": float(np.mean(seeds_S)),
                            "std_val_S": float(np.std(seeds_S, ddof=1)),
                            "wall_clock_s": wall})
        log(f"[pilot] family {ci}: mean val S={np.mean(seeds_S):.4f} "
            f"config={gcfg}")
    family_rows.sort(key=lambda r: r["mean_val_S"])
    chosen_cfg = family_rows[0]["config"]
    results["family_selection"] = family_rows
    results["chosen_config"] = chosen_cfg
    results["timing"]["family_selection_s"] = time.perf_counter() - t0
    log(f"[pilot] FROZEN config family: {chosen_cfg}")

    # ------------------------------------- stage C: 2x2 leak ablation
    t0 = time.perf_counter()
    leak_rows = []
    for ll, bl in itertools.product(("nominal", "physical"), repeat=2):
        seeds_S = []
        for rs, ms in zip(PILOT["reservoir_seeds"], PILOT["model_seeds"]):
            r = train_fused(chosen_cfg, ll, bl, True, rs, ms, arrays,
                            log=lambda *a: None)
            results["exclusions"].extend(r["exclusions"])
            seeds_S.append(val_S(r["system"], r["info"], arrays, pers_val))
        leak_rows.append({"lidar_leak": ll, "body_leak": bl,
                          "val_S_per_seed": seeds_S,
                          "mean_val_S": float(np.mean(seeds_S)),
                          "std_val_S": float(np.std(seeds_S, ddof=1))})
        log(f"[pilot] leak ({ll},{bl}): mean val S={np.mean(seeds_S):.4f}")
    leak_rows.sort(key=lambda r: r["mean_val_S"])
    lidar_leak = leak_rows[0]["lidar_leak"]
    body_leak = leak_rows[0]["body_leak"]
    results["leak_ablation"] = leak_rows
    results["chosen_leak"] = {"lidar": lidar_leak, "body": body_leak}
    results["timing"]["leak_ablation_s"] = time.perf_counter() - t0
    log(f"[pilot] FROZEN leak modes: lidar={lidar_leak} body={body_leak}")

    # ------------------- final pilot runs (frozen config, all model types)
    test_ep_ids = episode_ids_for(ds["test"])
    per_model: dict = {}
    trainable_ref = None

    def eval_modular_on_test(system, feats_test, stream):
        preds = predict_modular(system, feats_test, arrays["test"], stream)
        return (metrics_over_horizons(preds, arrays["test"]),
                per_episode_metrics(preds, arrays["test"], test_ep_ids))

    t0 = time.perf_counter()
    for kind in ("fused", "isolated"):
        entries = []
        for rs, ms in zip(PILOT["reservoir_seeds"], PILOT["model_seeds"]):
            r = train_fused(chosen_cfg, lidar_leak, body_leak,
                            kind == "fused", rs, ms, arrays,
                            log=lambda *a: None)
            results["exclusions"].extend(r["exclusions"])
            system, info = r["system"], r["info"]
            feats_test = precompute_features(system, arrays["test"])
            stream = "rel" if kind == "fused" else "local"
            mtest, per_ep = eval_modular_on_test(system, feats_test, stream)
            entry = {
                "seeds": {"reservoir": r["reservoir_seed_used"], "model": ms},
                "params": param_counts(system),
                "history": info["history"],
                "wall_clock_s": info["wall_clock_s"],
                "metrics_test": mtest,
                "test_S": mtest[H]["combined"] / pers_test[H]["combined"],
                "per_episode_combined_0.5s": per_ep,
                "health": info["health"],
            }
            if kind == "fused":
                entry["probes"] = message_probes(
                    system, info["features"]["train"],
                    info["features"]["val"], arrays["train"], arrays["val"])
                if "physical" in (lidar_leak, body_leak):
                    entry["partition_consistency"] = {
                        "lidar": partition_consistency(system.lidar_esn,
                                                       arrays["val"]),
                        "body": partition_consistency(system.body_esn,
                                                      arrays["val"])}
                ck = os.path.join(P1_MODELS, f"pilot_fused_rs{rs}.pt")
                entry["checkpoint"] = {
                    "path": ck,
                    "sha256": save_system(ck, system,
                                          {"esn": chosen_cfg,
                                           "leak": results["chosen_leak"]},
                                          extra={"stage": "pilot",
                                                 "seeds": entry["seeds"]},
                                          force=True)}
                trainable_ref = entry["params"]["trainable"]
            entries.append(entry)
            log(f"[pilot] {kind} rs={rs}: test S={entry['test_S']:.4f} "
                f"({info['wall_clock_s']:.0f}s)")
        per_model[f"modular_{kind}"] = entries
    results["timing"]["final_modular_s"] = time.perf_counter() - t0

    # GRU (parameter-matched), same splits/optimizer, 5 seeds
    t0 = time.perf_counter()
    entries = []
    for ms in PILOT["model_seeds"]:
        gru = build_matched_gru(trainable_ref, model_seed=ms)
        info = train_torch_model(gru, arrays["train"], arrays["val"],
                                 PILOT["epochs"], PILOT["batch_size"],
                                 PILOT["lr"], seed=ms, log=lambda *a: None)
        preds = predict_torch_model(gru, arrays["test"])
        mtest = metrics_over_horizons(preds, arrays["test"])
        entries.append({
            "seeds": {"model": ms}, "hidden": gru.hidden,
            "params": param_counts(gru),
            "history": info["history"],
            "wall_clock_s": info["wall_clock_s"],
            "metrics_test": mtest,
            "test_S": mtest[H]["combined"] / pers_test[H]["combined"],
            "per_episode_combined_0.5s": per_episode_metrics(
                preds, arrays["test"], test_ep_ids)})
        log(f"[pilot] gru ms={ms}: test S={entries[-1]['test_S']:.4f} "
            f"({info['wall_clock_s']:.0f}s)")
    per_model["gru_matched"] = entries
    results["timing"]["gru_s"] = time.perf_counter() - t0

    # raw-history MLP, 5 seeds
    t0 = time.perf_counter()
    entries = []
    for ms in PILOT["model_seeds"]:
        mlp = RawHistoryMLP(context_len=15, seed=ms)
        info = train_torch_model(mlp, arrays["train"], arrays["val"],
                                 PILOT["epochs"], PILOT["batch_size"],
                                 PILOT["lr"], seed=ms, log=lambda *a: None)
        preds = predict_torch_model(mlp, arrays["test"])
        mtest = metrics_over_horizons(preds, arrays["test"])
        entries.append({
            "seeds": {"model": ms}, "params": param_counts(mlp),
            "history": info["history"],
            "wall_clock_s": info["wall_clock_s"],
            "metrics_test": mtest,
            "test_S": mtest[H]["combined"] / pers_test[H]["combined"]})
    per_model["raw_history_mlp"] = entries
    results["timing"]["mlp_s"] = time.perf_counter() - t0
    results["models"] = per_model

    # -------------------------------------------- S / C / M and power
    s_esn = [e["test_S"] for e in per_model["modular_fused"]]
    s_iso = [e["test_S"] for e in per_model["modular_isolated"]]
    s_gru = [e["test_S"] for e in per_model["gru_matched"]]
    pers_L = pers_test[H]["combined"]
    c_vals = [(i - f) * pers_L for f, i in zip(s_esn, s_iso)]
    m_vals = [(g - f) * pers_L for f, g in zip(s_esn, s_gru)]
    d_vals = [f - g for f, g in zip(s_esn, s_gru)]     # paired S difference
    sigma_d = float(np.std(d_vals, ddof=1))
    n_rule = math.ceil((2.8 * sigma_d / PILOT["delta"]) ** 2)
    n_conf = max(n_rule, PILOT["min_n"])
    results["endpoints"] = {
        "persistence_combined_0.5s_test": pers_L,
        "S_modular_per_seed": s_esn,
        "S_isolated_per_seed": s_iso,
        "S_gru_per_seed": s_gru,
        "S_modular_mean": float(np.mean(s_esn)),
        "S_modular_std": float(np.std(s_esn, ddof=1)),
        "S_gru_mean": float(np.mean(s_gru)),
        "C_per_seed": c_vals, "C_mean": float(np.mean(c_vals)),
        "M_per_seed": m_vals, "M_mean": float(np.mean(m_vals)),
        "paired_S_diff_per_seed": d_vals,
        "sigma_d": sigma_d,
        "n_rule": n_rule, "delta": PILOT["delta"],
        "confirmatory_n": n_conf,
    }

    results["provenance"] = gather_provenance(
        {"pilot": PILOT, "env": cfg["env"]}, experiment_name=EXPERIMENT,
        variant="pilot",
        seeds={"reservoir": PILOT["reservoir_seeds"],
               "model": PILOT["model_seeds"]})
    results["total_wall_clock_s"] = time.perf_counter() - t_start
    out = write_results("pilot_result", results,
                        results_dir=os.path.join(P1_RESULTS, "pilot"))
    log(f"[pilot] endpoints: {results['endpoints']}")
    log(f"[pilot] wrote {out} ({results['total_wall_clock_s']:.0f}s total)")
    return out


if __name__ == "__main__":
    run()
