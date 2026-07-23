"""Factorial experiment manifest I/O, verification and agent reconstruction.

Manifest (schema_version 2) conventions:
  - "cells" holds one entry per TRAINING RUN with explicit identity fields
    (cell_id, training_seed, run_id, experiment, attempt) — identity is
    never inferred by parsing strings;
  - status lifecycle: planned -> training -> completed | failed, or
    skipped_valid when an existing artifact matches the PLANNED spec;
    every transition is recorded in status_history with timestamps;
  - retraining with --force increments `attempt` and keeps the previous
    record in `attempt_history` — historical run records are never
    silently replaced in a scientific manifest;
  - writes are atomic (tmp + fsync + os.replace): an interrupted write can
    never corrupt the previous manifest;
  - verification compares the artifact against BOTH its recorded manifest
    checksums and the independently planned specification, so
    skipped_valid means "matches the experiment we intended to run", not
    merely "internally self-consistent".
"""
from __future__ import annotations

import copy
import json
import os
import time

import torch

from common import MODELS_DIR, ROOT, load_config
from provenance import SEMANTICS_VERSION, file_checksum

MANIFEST_SCHEMA_VERSION = 2


def manifest_path(smoke: bool = False, stamp: str | None = None) -> str:
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if smoke else stamp
    return os.path.join(MODELS_DIR, f"factorial_manifest_{tag}.json")


def run_identity(cell_name: str, training_seed: int) -> str:
    return f"{cell_name}__s{int(training_seed)}"


def planned_spec(cell: dict, training_seed: int, smoke: bool,
                 resolved_cfg: dict) -> dict:
    """Independently planned specification a finished artifact must match."""
    tr = resolved_cfg.get("training", {})
    return {
        "snn_time_aware": bool(cell["snn_time_aware"]),
        "cfc_time_aware": bool(cell["cfc_time_aware"]),
        "mask_direct_dt": bool(cell["mask_direct_dt"]),
        "training_seed": int(training_seed),
        "smoke": bool(smoke),
        "budget": {
            "generations": int(tr["cma_generations"]),
            "population": int(tr["cma_population"]),
            "episodes_per_candidate": int(tr["episodes_per_candidate"]),
            "validation_episodes": int(tr.get("validation_episodes", 12)),
        },
        "semantics_version": SEMANTICS_VERSION,
        "mode": "hierarchical" if cell.get("hierarchical") else "reactive",
    }


def artifact_fields(checkpoint: str) -> dict:
    """Checksums/compat read from an existing artifact (None when absent)."""
    out = {"state_checksum": None, "file_sha256": None, "compat": None}
    if not os.path.exists(checkpoint):
        return out
    out["file_sha256"] = file_checksum(checkpoint)
    try:
        payload = torch.load(checkpoint, weights_only=True)
    except Exception:
        return out
    if isinstance(payload, dict) and "meta" in payload:
        m = payload["meta"]
        out["state_checksum"] = m.get("state_checksum")
        out["compat"] = m.get("compat")
    return out


def cell_entry(cell: dict, checkpoint: str, training_seed: int,
               experiment: str, smoke: bool, resolved_cfg: dict,
               attempt: int = 1, status: str = "completed",
               extra: dict | None = None) -> dict:
    """One RUN entry of a factorial manifest (explicit identity fields)."""
    a = resolved_cfg.get("agent", {})
    abs_ckpt = checkpoint if os.path.isabs(checkpoint) \
        else os.path.join(ROOT, checkpoint)
    entry = {
        "cell_id": cell["name"],
        "name": cell["name"],                      # legacy alias for cell_id
        "training_seed": int(training_seed),
        "run_id": run_identity(cell["name"], training_seed),
        "experiment": experiment,
        "attempt": int(attempt),
        "status": status,
        "status_history": [],
        "attempt_history": [],
        "smoke": bool(smoke),
        "checkpoint": (os.path.relpath(checkpoint, ROOT)
                       if os.path.isabs(checkpoint) else checkpoint),
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
        "planned_spec": planned_spec(cell, training_seed, smoke, resolved_cfg),
    }
    entry.update(artifact_fields(abs_ckpt))
    if extra:
        entry.update(extra)
    return entry


def set_run_status(entry: dict, new_status: str, error: str | None = None,
                   checkpoint: str | None = None) -> None:
    """Record a status transition (timestamp, attempt, from/to, error)."""
    entry.setdefault("status_history", []).append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attempt": int(entry.get("attempt", 1)),
        "from": entry.get("status"),
        "to": new_status,
        "error": (str(error)[:500] if error else None),
        "checkpoint": checkpoint,
    })
    entry["status"] = new_status


def write_manifest(path: str, entries: list[dict],
                   smoke: bool = False, meta: dict | None = None) -> str:
    """Atomic write: serialize to tmp, fsync, os.replace. An interrupted
    write never corrupts a previously valid manifest at `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "smoke": bool(smoke),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cells": len(entries),
        "cells": entries,
    }
    if meta:
        payload["meta"] = meta
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
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


# ------------------------------------------------------------- verification


def verify_manifest_cell(cell: dict, expected_spec: dict | None = None) -> dict:
    """Verify one manifest run entry against its artifact AND the planned
    experiment specification (entry["planned_spec"] unless expected_spec is
    given). Raises ValueError listing every disagreement — never silently
    accepts a replaced, edited, or wrong-experiment artifact."""
    ckpt = cell["checkpoint"]
    path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest checkpoint missing: {path}")
    problems: list[str] = []

    actual_sha = file_checksum(path)
    if cell.get("file_sha256") and actual_sha != cell["file_sha256"]:
        problems.append(f"file_sha256: manifest={cell['file_sha256'][:12]}... "
                        f"actual={actual_sha[:12]}...")

    payload = torch.load(path, weights_only=True)
    if not (isinstance(payload, dict) and "meta" in payload):
        raise ValueError(f"{path} has no provenance metadata; cannot verify "
                         f"against the manifest")
    meta = payload["meta"]

    if cell.get("state_checksum") \
            and meta.get("state_checksum") != cell["state_checksum"]:
        problems.append("state_checksum: manifest disagrees with checkpoint")
    if cell.get("compat") is not None and meta.get("compat") != cell["compat"]:
        problems.append("compat block: manifest disagrees with checkpoint")
    if cell.get("experiment") and meta.get("experiment") != cell["experiment"]:
        problems.append(f"experiment: manifest={cell['experiment']!r} "
                        f"checkpoint={meta.get('experiment')!r}")
    seeds = meta.get("seeds") or {}
    if cell.get("training_seed") is not None \
            and seeds.get("cma_seed") is not None \
            and int(seeds["cma_seed"]) != int(cell["training_seed"]):
        problems.append(f"training_seed: manifest={cell['training_seed']} "
                        f"checkpoint cma_seed={seeds['cma_seed']}")

    mcfg_agent = (meta.get("config") or {}).get("agent") or {}
    factors = cell.get("variant_factors") or {}
    for k in ("snn_time_aware", "cfc_time_aware", "mask_direct_dt"):
        if k in factors and k in mcfg_agent \
                and bool(mcfg_agent[k]) != bool(factors[k]):
            problems.append(f"factor {k}: manifest={factors[k]} "
                            f"checkpoint config={mcfg_agent[k]}")

    # --- planned-spec agreement (non-tautological: independent of the
    # metadata copied into the entry) ---
    spec = expected_spec if expected_spec is not None \
        else cell.get("planned_spec")
    if spec:
        for k in ("snn_time_aware", "cfc_time_aware", "mask_direct_dt"):
            if spec.get(k) is not None and k in mcfg_agent \
                    and bool(mcfg_agent[k]) != bool(spec[k]):
                problems.append(f"planned {k}={spec[k]} but checkpoint "
                                f"config has {mcfg_agent[k]}")
        if spec.get("training_seed") is not None:
            if seeds.get("cma_seed") is None:
                problems.append("planned training_seed set but checkpoint "
                                "records no cma_seed")
            elif int(seeds["cma_seed"]) != int(spec["training_seed"]):
                problems.append(f"planned training_seed={spec['training_seed']}"
                                f" but checkpoint cma_seed={seeds['cma_seed']}")
        if spec.get("smoke") is not None:
            if "smoke" not in meta:
                problems.append("planned run_kind/smoke set but checkpoint "
                                "metadata records none (pre-run_kind artifact;"
                                " retrain with --force)")
            elif bool(meta["smoke"]) != bool(spec["smoke"]):
                problems.append(f"planned smoke={spec['smoke']} but "
                                f"checkpoint smoke={meta['smoke']}")
        if spec.get("budget"):
            mb = meta.get("budget") or {}
            for bk, bv in spec["budget"].items():
                if mb.get(bk) != bv:
                    problems.append(f"planned budget.{bk}={bv} but "
                                    f"checkpoint has {mb.get(bk)}")
        if spec.get("semantics_version") is not None \
                and meta.get("semantics_version") != spec["semantics_version"]:
            problems.append(f"planned semantics_version="
                            f"{spec['semantics_version']} but checkpoint has "
                            f"{meta.get('semantics_version')}")
        if spec.get("mode") and meta.get("mode") != spec["mode"]:
            problems.append(f"planned mode={spec['mode']!r} but checkpoint "
                            f"mode={meta.get('mode')!r}")

    if problems:
        raise ValueError(
            f"manifest verification FAILED for "
            f"{cell.get('run_id', cell.get('cell_id', path))}:\n  "
            + "\n  ".join(problems))
    return {"verified": True, "file_sha256": actual_sha,
            "experiment": meta.get("experiment"),
            "run_kind": meta.get("run_kind"),
            "training_seed": seeds.get("cma_seed")}


# --------------------------------------------------- agent reconstruction


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


def config_from_cell(cell: dict, base: dict | None = None) -> dict:
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
    """Construct a HybridAgent matching the checkpoint and load weights.

    Returns (agent, cfg, config_source). Semantics-v3 checkpoints MUST carry
    a resolved config in metadata; failure to read it is an error, never a
    silent fallback. Rebuilding from cell factors requires an explicit
    allow_legacy=True and is reported as "cell_factors_legacy".
    """
    from agent.hybrid_agent import HybridAgent
    path = checkpoint
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    payload = torch.load(path, weights_only=True)
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if meta is None:
        raise ValueError(f"{path} is a bare state dict without provenance "
                         f"metadata; import it as legacy first")
    stored = meta.get("config")
    if isinstance(stored, dict) and stored.get("agent"):
        cfg = copy.deepcopy(stored)
        config_source = "checkpoint_meta"
    elif allow_legacy:
        cfg = (config_from_cell(cell, base_cfg) if cell is not None
               else config_from_checkpoint(path, base_cfg))
        config_source = "cell_factors_legacy"
    else:
        raise ValueError(
            f"{path} carries no resolved config in its metadata (required "
            f"for semantics_version {SEMANTICS_VERSION} artifacts). Rebuild "
            f"from manifest cell factors only deliberately, with "
            f"allow_legacy=True.")
    agent = HybridAgent(cfg, mode="reactive")
    agent.load(path, allow_legacy=allow_legacy)
    return agent, cfg, config_source
