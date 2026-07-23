"""Provenance and artifact-safety helpers (Priority 0).

Every checkpoint and result produced by this project records where it came
from: git commit, dirty state, fully resolved configuration, seeds, package
versions, parameter counts and the timing distribution it was trained /
evaluated under. Checkpoints carry a compatibility block that is validated
on load. Provenance currently covers agent/world-model `.pt` checkpoints and
(from semantics_version 3) fingerprinted replay buffers; Stable-Baselines
PPO zips and pre-v3 experience.npz files still need explicit legacy paths.

Conventions:
  - A checkpoint file is a single .pt containing {"state": ..., "meta": ...}.
    meta["compat"] holds the keys that must match the loading configuration;
    meta["state_checksum"] is a SHA-256 over a canonical tensor encoding
    (sorted names + dtype/shape + contiguous bytes) and is verified on load.
  - Legacy (pre-provenance) artifacts are wrapped via import_legacy_checkpoint
    and marked meta["legacy"] = True. They load only when the caller passes
    allow_legacy=True, so they cannot be mistaken for new experiments.
  - save_checkpoint refuses to overwrite an existing file unless force=True.
  - Result JSONs are written to timestamped paths and never overwrite.
"""
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from common import RESULTS_DIR, ROOT

# Bump whenever environment, observation, reward, model or timing semantics
# change in a way that invalidates previously trained artifacts.
#   1: original snapshot (single-spike LIF, endpoint collision charge,
#      horizon overshoot, raw dt in obs consumed unmasked).
#   2: event-count LIF, causal timing convention, per-substep collision
#      integration, exact physical horizon, dt masking, input adapters.
#   3: shared 54-d sensory bus adapters, replay-buffer provenance,
#      verified state checksums, strengthened WM/hierarchy gates.
SEMANTICS_VERSION = 3

_PACKAGES = ("torch", "numpy", "gymnasium", "ncps", "cmaes",
             "stable_baselines3", "pybullet", "yaml")


# --------------------------------------------------------------- environment


def git_info() -> dict:
    def _run(args):
        try:
            return subprocess.check_output(
                ["git"] + args, cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = _run(["status", "--porcelain"])
    return {"commit": _run(["rev-parse", "HEAD"]),
            "dirty": bool(status) if status is not None else None}


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in _PACKAGES:
        try:
            out[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def _sanitize(obj):
    """Make metadata JSON- and weights_only-serializable: numpy scalars ->
    python, anything exotic (e.g. torch's TorchVersion) -> str."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    return str(obj)


def gather_provenance(config: dict, experiment_name: str, variant: str = "",
                      seeds=None, extra: dict | None = None) -> dict:
    meta = {
        "experiment": experiment_name,
        "variant": variant,
        "semantics_version": SEMANTICS_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_info(),
        "packages": package_versions(),
        "platform": platform.platform(),
        "config": copy.deepcopy(config),
        "seeds": seeds,
    }
    if extra:
        meta.update(extra)
    return _sanitize(meta)


# ---------------------------------------------------------------- checksums


def _flatten_tensors(obj, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Walk nested state dicts and collect (dotted_name, tensor) pairs."""
    out = []
    if isinstance(obj, torch.Tensor):
        out.append((prefix or "tensor", obj))
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_tensors(obj[k], key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.extend(_flatten_tensors(v, key))
    return out


def state_checksum(state: dict) -> str:
    """SHA-256 over a canonical encoding of tensor contents.

    Hashes sorted tensor names with dtype, shape, and raw contiguous bytes
    rather than torch.save() output, so the digest is stable across library
    versions that change pickle framing.
    """
    h = hashlib.sha256()
    for name, tensor in _flatten_tensors(state):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(t.dtype).encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(b"\0")
        h.update(t.numpy().tobytes())
        h.update(b"\0")
    return h.hexdigest()


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_ref(path: str) -> dict:
    """Reference block for result JSONs: exactly which artifact was used."""
    ref = {"path": os.path.relpath(path, ROOT) if os.path.isabs(path) else path,
           "sha256": file_checksum(path) if os.path.exists(path) else None}
    try:
        payload = torch.load(path, weights_only=True)
        if isinstance(payload, dict) and "meta" in payload:
            m = payload["meta"]
            ref.update({"experiment": m.get("experiment"),
                        "legacy": bool(m.get("legacy", False)),
                        "timestamp": m.get("timestamp")})
    except Exception:
        pass
    return ref


# -------------------------------------------------------------- checkpoints


def save_checkpoint(path: str, state: dict, meta: dict, compat: dict,
                    force: bool = False) -> str:
    """Save {"state", "meta"} with compat + checksum. Returns file SHA-256."""
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"refusing to overwrite existing checkpoint {path}. "
            f"Pass force=True (CLI: --force) or choose a new experiment name.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(_sanitize(meta))
    meta["compat"] = dict(_sanitize(compat))
    meta["state_checksum"] = state_checksum(state)
    torch.save({"state": state, "meta": meta}, path)
    return file_checksum(path)


def load_checkpoint(path: str, expected_compat: dict | None = None,
                    allow_legacy: bool = False):
    """Load and validate a checkpoint. Returns (state, meta).

    Raises with an actionable message when the file has no provenance
    metadata, is a legacy artifact loaded without allow_legacy, or its
    compat block disagrees with the current configuration.
    """
    payload = torch.load(path, weights_only=True)
    if not (isinstance(payload, dict) and "state" in payload
            and "meta" in payload):
        raise ValueError(
            f"{path} is a bare state dict without provenance metadata. "
            f"If this is a pre-provenance artifact, import it explicitly: "
            f"`python main.py import-legacy` and load the imported copy "
            f"with allow_legacy=True.")
    meta = payload["meta"]
    state = payload["state"]
    if meta.get("legacy", False):
        if not allow_legacy:
            raise ValueError(
                f"{path} is a LEGACY artifact (trained before "
                f"semantics_version {SEMANTICS_VERSION}). Its training "
                f"semantics differ from the current code. Load it only "
                f"deliberately, with allow_legacy=True.")
        return state, meta
    stored = meta.get("state_checksum")
    if stored is None:
        if not allow_legacy:
            raise ValueError(
                f"{path} has no state_checksum (pre-verification artifact). "
                f"Re-save under the current provenance module, or load "
                f"deliberately with allow_legacy=True.")
    else:
        actual_sum = state_checksum(state)
        if actual_sum != stored:
            raise ValueError(
                f"checkpoint {path} failed state_checksum verification:\n"
                f"  recorded={stored}\n  recomputed={actual_sum}\n"
                f"The file may be corrupted or its weights were modified.")
    if expected_compat is not None:
        actual = meta.get("compat", {})
        diffs = {k: {"checkpoint": actual.get(k), "current": v}
                 for k, v in _sanitize(expected_compat).items()
                 if actual.get(k) != v}
        if diffs:
            lines = "\n".join(f"  {k}: checkpoint={v['checkpoint']!r} "
                              f"current={v['current']!r}"
                              for k, v in diffs.items())
            raise ValueError(
                f"checkpoint {path} is incompatible with the current "
                f"configuration:\n{lines}\n"
                f"Either restore the matching config, retrain under the new "
                f"semantics, or (for old artifacts) re-import as legacy.")
    return state, meta


def import_legacy_checkpoint(src: str, dst: str, note: str = "",
                             force: bool = False) -> str:
    """Wrap a bare pre-provenance state dict as an explicit legacy artifact."""
    state = torch.load(src, weights_only=True)
    if isinstance(state, dict) and "state" in state and "meta" in state:
        raise ValueError(f"{src} already carries provenance metadata")
    meta = gather_provenance(
        {}, experiment_name=f"legacy:{os.path.basename(src)}",
        extra={"legacy": True, "note": note,
               "imported_from": os.path.relpath(src, ROOT),
               "legacy_semantics_version": 1})
    return save_checkpoint(dst, state, meta, compat={"legacy": True},
                           force=force)


# ------------------------------------------------------------------ results


def write_results(stem: str, payload: dict, results_dir: str = RESULTS_DIR) -> str:
    """Write a result JSON to a fresh timestamped path (never overwrite)."""
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{stem}_{ts}.json")
    n = 1
    while os.path.exists(path):
        path = os.path.join(results_dir, f"{stem}_{ts}_{n}.json")
        n += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, indent=2)
    return path


def count_parameters(params) -> int:
    return int(sum(p.numel() for p in params))
