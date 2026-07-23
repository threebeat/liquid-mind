"""Factorial experiment manifest I/O and agent reconstruction helpers."""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

import torch

from common import MODELS_DIR, ROOT, load_config
from provenance import file_checksum


def manifest_path(smoke: bool = False, stamp: str | None = None) -> str:
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if smoke else stamp
    return os.path.join(MODELS_DIR, f"factorial_manifest_{tag}.json")


def cell_entry(cell: dict, checkpoint: str, training_seed: int,
               experiment: str, smoke: bool, resolved_cfg: dict,
               extra: dict | None = None) -> dict:
    """One row of a factorial manifest."""
    meta = {}
    state_checksum = None
    if os.path.exists(checkpoint):
        try:
            payload = torch.load(checkpoint, weights_only=True)
            if isinstance(payload, dict) and "meta" in payload:
                meta = payload["meta"]
                state_checksum = meta.get("state_checksum")
        except Exception:
            pass
    a = resolved_cfg.get("agent", {})
    entry = {
        "name": cell["name"],
        "checkpoint": (os.path.relpath(checkpoint, ROOT)
                       if os.path.isabs(checkpoint) else checkpoint),
        "experiment": experiment,
        "training_seed": int(training_seed),
        "smoke": bool(smoke),
        "variant_factors": {
            "snn_time_aware": bool(cell["snn_time_aware"]),
            "cfc_time_aware": bool(cell["cfc_time_aware"]),
            "mask_direct_dt": bool(cell["mask_direct_dt"]),
            "hierarchical": bool(cell.get("hierarchical", False)),
        },
        "resolved_agent": {
            "snn_time_aware": bool(a.get("snn_time_aware", True)),
            "cfc_time_aware": bool(a.get("cfc_time_aware", True)),
            "mask_direct_dt": bool(a.get("mask_direct_dt", True)),
            "policy_input": a.get("policy_input", "spikes_obs"),
            "snn_semantics": a.get("snn_semantics", "event_count"),
            "timing_convention": a.get("timing_convention", "causal"),
            "use_input_adapter": bool(a.get("use_input_adapter", True)),
            "adapter_dim": int(a.get("adapter_dim", 32)),
        },
        "state_checksum": state_checksum,
        "file_sha256": (file_checksum(checkpoint)
                        if os.path.exists(checkpoint) else None),
        "compat": meta.get("compat"),
    }
    if extra:
        entry.update(extra)
    return entry


def write_manifest(path: str, entries: list[dict],
                   smoke: bool = False, meta: dict | None = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cells": len(entries),
        "cells": entries,
    }
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_latest_manifest(smoke: bool | None = None) -> str | None:
    """Return the newest factorial_manifest_*.json under models/."""
    if not os.path.isdir(MODELS_DIR):
        return None
    cands = []
    for name in os.listdir(MODELS_DIR):
        if not name.startswith("factorial_manifest_") or not name.endswith(".json"):
            continue
        if smoke is True and "smoke" not in name:
            continue
        if smoke is False and "smoke" in name:
            continue
        cands.append(os.path.join(MODELS_DIR, name))
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def config_from_checkpoint(path: str, base: dict | None = None) -> dict:
    """Rebuild a full config from checkpoint metadata (preferred) or base."""
    cfg = copy.deepcopy(base if base is not None else load_config())
    payload = torch.load(path, weights_only=True)
    if not (isinstance(payload, dict) and "meta" in payload):
        return cfg
    meta = payload["meta"]
    stored = meta.get("config")
    if isinstance(stored, dict) and stored:
        return copy.deepcopy(stored)
    compat = meta.get("compat") or {}
    a = cfg.setdefault("agent", {})
    for key in ("snn_time_aware", "cfc_time_aware", "mask_direct_dt",
                "policy_input", "snn_semantics", "timing_convention",
                "use_input_adapter", "adapter_dim", "snn_neurons",
                "lnn_units", "latent_dim"):
        if key in compat and compat[key] is not None:
            a[key] = compat[key]
    return cfg


def config_from_cell(cell: dict | Any, base: dict | None = None) -> dict:
    """Apply variant factors from a manifest cell (or cell dict) onto base."""
    cfg = copy.deepcopy(base if base is not None else load_config())
    factors = cell.get("variant_factors") or cell.get("resolved_agent") or cell
    a = cfg.setdefault("agent", {})
    for key in ("snn_time_aware", "cfc_time_aware", "mask_direct_dt",
                "policy_input", "snn_semantics", "timing_convention",
                "use_input_adapter", "adapter_dim"):
        if key in factors and factors[key] is not None:
            a[key] = factors[key]
    resolved = cell.get("resolved_agent") or {}
    for key, val in resolved.items():
        if val is not None:
            a[key] = val
    return cfg


def load_agent_for_checkpoint(checkpoint: str, allow_legacy: bool = False,
                              base_cfg: dict | None = None,
                              cell: dict | None = None):
    """Construct a HybridAgent matching the checkpoint and load weights."""
    from agent.hybrid_agent import HybridAgent
    path = checkpoint
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if cell is not None:
        cfg = config_from_cell(cell, base_cfg)
        # Prefer full stored config when present
        try:
            cfg = config_from_checkpoint(path, cfg)
        except Exception:
            pass
    else:
        cfg = config_from_checkpoint(path, base_cfg)
    agent = HybridAgent(cfg, mode="reactive")
    agent.load(path, allow_legacy=allow_legacy)
    return agent, cfg
