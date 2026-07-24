"""Shared experiment-level helpers for the Phase 1 stage runners (smoke,
pilot, confirmatory): system construction, parameter accounting, baseline
fitting/evaluation, message probes, and provenance-carrying checkpoints."""
from __future__ import annotations

import os

import numpy as np
import torch

from phase1 import HORIZON_STEPS, PRIMARY_HORIZON_S
from phase1.baselines import (LinearARBaseline, RawHistoryMLP,
                              kinematic_body_predictions,
                              persistence_predictions, ridge_probe)
from phase1.esn import make_body_specialist, make_lidar_specialist
from phase1.gru_baseline import (MonolithicGRU, count_total, count_trainable,
                                 matched_hidden_size)
from phase1.predictors import ModularSystem
from phase1.train_eval import (FRONT_RAYS, metrics_over_horizons, n_windows,
                               precompute_features, predict_modular,
                               predict_torch_model, take)
from provenance import (gather_provenance, load_checkpoint, save_checkpoint,
                        state_checksum)

DEFAULT_ESN_CFG = {
    "spectral_radius": 0.9, "alpha0": 0.3, "sparsity": 0.1,
    "input_scale": 0.5,
}


def build_system(reservoir_seed: int, model_seed: int,
                 esn_cfg: dict | None = None,
                 lidar_leak: str = "nominal", body_leak: str = "nominal",
                 fusion: bool = True, action_input: bool = False,
                 horizon_steps: dict | None = None) -> ModularSystem:
    c = dict(DEFAULT_ESN_CFG)
    c.update(esn_cfg or {})
    lidar = make_lidar_specialist(
        seed=reservoir_seed, leak_mode=lidar_leak,
        action_input=action_input, **c)
    body = make_body_specialist(
        seed=reservoir_seed + 500, leak_mode=body_leak, **c)
    return ModularSystem(lidar, body, model_seed=model_seed,
                         horizon_steps=horizon_steps, fusion=fusion)


def system_config_dict(system: ModularSystem) -> dict:
    return {"lidar_esn": system.lidar_esn.cfg.to_dict(),
            "body_esn": system.body_esn.cfg.to_dict(),
            "fusion": system.fusion,
            "action_input": bool(getattr(system.lidar_esn,
                                         "action_input", False)),
            "horizon_steps": {str(k): v
                              for k, v in system.horizon_steps.items()}}


def param_counts(module) -> dict:
    return {"trainable": count_trainable(module),
            "total": count_total(module)}


def build_matched_gru(target_trainable: int, model_seed: int) -> MonolithicGRU:
    hidden = matched_hidden_size(target_trainable)
    return MonolithicGRU(hidden, seed=model_seed)


# ------------------------------------------------------------- baselines


def baseline_metrics(train_arrays: dict, eval_arrays: dict,
                     max_wheel_speed: float = 20.0,
                     linear_ar: LinearARBaseline | None = None) -> tuple:
    """(metrics dict, fitted linear AR). Persistence / linear AR / kinematic
    (body only; lidar via persistence for its combined score)."""
    out = {}
    pers = persistence_predictions(eval_arrays)
    out["persistence"] = metrics_over_horizons(pers, eval_arrays)

    if linear_ar is None:
        linear_ar = LinearARBaseline().fit(train_arrays)
    out["linear_ar"] = metrics_over_horizons(linear_ar.predict(eval_arrays),
                                             eval_arrays)

    kin_body = kinematic_body_predictions(eval_arrays, max_wheel_speed)
    kin = {"lidar": pers["lidar"], "body": kin_body}   # rays persist
    out["kinematic"] = metrics_over_horizons(kin, eval_arrays)
    return out, linear_ar


# ---------------------------------------------------------------- probes


@torch.no_grad()
def message_probes(system: ModularSystem, feats_train: dict,
                   feats_val: dict, train_arrays: dict,
                   val_arrays: dict) -> dict:
    """Linear (ridge) probes of the 16-d messages: current and future front
    clearance from m_L; future body from m_B (0.5 s endpoint)."""
    steps = HORIZON_STEPS[PRIMARY_HORIZON_S]
    m_l_tr = system.lidar_esn.message_from_features(feats_train["lidar"]).numpy()
    m_l_va = system.lidar_esn.message_from_features(feats_val["lidar"]).numpy()
    m_b_tr = system.body_esn.message_from_features(feats_train["body"]).numpy()
    m_b_va = system.body_esn.message_from_features(feats_val["body"]).numpy()

    fc_now_tr = train_arrays["lidar"][:, -1, FRONT_RAYS].min(axis=1)
    fc_now_va = val_arrays["lidar"][:, -1, FRONT_RAYS].min(axis=1)
    fc_fut_tr = train_arrays["future_lidar"][:, steps - 1][:, FRONT_RAYS].min(axis=1)
    fc_fut_va = val_arrays["future_lidar"][:, steps - 1][:, FRONT_RAYS].min(axis=1)
    body_fut_tr = train_arrays["future_body"][:, steps - 1]
    body_fut_va = val_arrays["future_body"][:, steps - 1]

    return {
        "m_lidar_to_front_clearance_now": ridge_probe(
            m_l_tr, fc_now_tr, m_l_va, fc_now_va),
        "m_lidar_to_front_clearance_0.5s": ridge_probe(
            m_l_tr, fc_fut_tr, m_l_va, fc_fut_va),
        "m_body_to_body_0.5s": ridge_probe(
            m_b_tr, body_fut_tr, m_b_va, body_fut_va),
    }


# ------------------------------------------------------------ checkpoints


def system_compat(system: ModularSystem) -> dict:
    return {"experiment": "specialists_phase1_v1",
            "lidar_units": system.lidar_esn.cfg.n_units,
            "body_units": system.body_esn.cfg.n_units,
            "fusion": system.fusion}


def save_system(path: str, system: ModularSystem, config: dict,
                extra: dict | None = None, force: bool = False) -> str:
    meta = gather_provenance(config, experiment_name="specialists_phase1_v1",
                             extra=extra or {})
    return save_checkpoint(path, system.state_dict(), meta,
                           system_compat(system), force=force)


def load_system(path: str, reservoir_seed: int, model_seed: int,
                esn_cfg: dict | None = None, lidar_leak="nominal",
                body_leak="nominal", fusion=True,
                action_input=False) -> tuple:
    system = build_system(reservoir_seed, model_seed, esn_cfg, lidar_leak,
                          body_leak, fusion, action_input)
    state, meta = load_checkpoint(path, expected_compat=system_compat(system))
    system.load_state_dict(state)
    system.eval()
    return system, meta


# ---------------------------------------------------------- model metrics


def modular_metrics(system: ModularSystem, arrays: dict,
                    feats: dict | None = None) -> dict:
    if feats is None:
        feats = precompute_features(system, arrays)
    out = {"local": metrics_over_horizons(
        predict_modular(system, feats, arrays, "local"), arrays)}
    if system.fusion:
        out["rel"] = metrics_over_horizons(
            predict_modular(system, feats, arrays, "rel"), arrays)
    return out


def gru_metrics(model: MonolithicGRU, arrays: dict) -> dict:
    return metrics_over_horizons(predict_torch_model(model, arrays), arrays)
