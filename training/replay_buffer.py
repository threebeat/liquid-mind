"""Episodic replay buffer for world-model training.

Stores whole episodes of (obs, action, dt) plus event timestamps, and
samples chunk-level transitions either by decision count (legacy) or by
target PHYSICAL duration (variable numbers of action events per chunk).

Transition timing fields (preparing for asynchronous sensing):
    t_capture[k]   simulated time at which obs[k] was captured
                   (t_capture[0] = 0 at reset: assimilation at elapsed 0);
    t_delivery[k]  time the measurement reached the agent — currently equal
                   to t_capture (synchronized Gym stepping); kept as an
                   explicit field so delayed/stale delivery can be injected
                   later without a schema change.
Derived per-step quantities: action k is issued at t_capture[k] (after the
delivery of obs[k]), held for dts[k], and the next measurement is captured
at t_capture[k+1]. All sensors within an observation are currently captured
together and always valid (per-sensor validity/age arrives with the
streaming environment).

From semantics_version 3, buffers carry an embedded metadata block and a
fingerprinted filename; incompatible buffers are refused on load unless
allow_legacy_buffer=True.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

from provenance import SEMANTICS_VERSION, _sanitize, file_checksum, git_info

BUFFER_SCHEMA_VERSION = 2


class ReplayBuffer:
    def __init__(self, meta: dict | None = None):
        self.episodes: list[dict] = []
        self.meta: dict = dict(meta or {})

    def add_episode(self, obs: np.ndarray, actions: np.ndarray,
                    dts: np.ndarray, t_capture: np.ndarray | None = None,
                    t_delivery: np.ndarray | None = None):
        """obs: (T+1, obs_dim), actions: (T, act_dim), dts: (T,),
        t_capture/t_delivery: (T+1,) or None (reconstructed from dts)."""
        dts = np.asarray(dts, dtype=np.float32)
        if t_capture is None:
            t_capture = np.concatenate([[0.0], np.cumsum(dts)]).astype(np.float32)
        if t_delivery is None:
            t_delivery = np.asarray(t_capture, dtype=np.float32).copy()
        self.episodes.append({
            "obs": np.asarray(obs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "dts": dts,
            "t_capture": np.asarray(t_capture, dtype=np.float32),
            "t_delivery": np.asarray(t_delivery, dtype=np.float32)})

    def __len__(self):
        return len(self.episodes)

    def n_steps(self):
        return sum(len(e["actions"]) for e in self.episodes)

    # ------------------------------------------------------------- sampling

    def sample_chunks(self, batch: int, chunk: int, rng: np.random.Generator):
        """Legacy fixed-decision-count chunks."""
        obs_t, act_mean, dt_sum, obs_next = [], [], [], []
        usable = [e for e in self.episodes if len(e["actions"]) > chunk]
        for _ in range(batch):
            e = usable[rng.integers(len(usable))]
            t = int(rng.integers(0, len(e["actions"]) - chunk))
            a = e["actions"][t:t + chunk]
            d = e["dts"][t:t + chunk]
            obs_t.append(e["obs"][t])
            # duration-weighted mean: a command held for 66 ms counts more
            # than one held for 16 ms (matters under irregular timing)
            act_mean.append((a * d[:, None]).sum(axis=0) / d.sum())
            dt_sum.append([d.sum()])
            obs_next.append(e["obs"][t + chunk])
        return (np.array(obs_t), np.array(act_mean),
                np.array(dt_sum, dtype=np.float32), np.array(obs_next))

    @staticmethod
    def steps_for_duration(dts: np.ndarray, start: int, seconds: float):
        """Number of decisions from `start` whose durations first reach
        `seconds` of physical time, or None if the episode ends first."""
        acc, k = 0.0, start
        n = len(dts)
        while k < n and acc < seconds - 1e-9:
            acc += float(dts[k])
            k += 1
        if acc < seconds - 1e-9:
            return None
        return k - start, acc

    def sample_chunks_by_duration(self, batch: int, seconds: float,
                                  rng: np.random.Generator):
        """Chunks defined by target PHYSICAL duration: a variable number of
        action events is aggregated until >= `seconds` elapsed. Returns
        (obs_t, duration-weighted mean action, actual duration, obs_next).

        NOTE: action order inside the chunk is discarded (duration-weighted
        mean). An order-sensitive action-duration encoder is a next-stage
        baseline comparison, not yet implemented.
        """
        obs_t, act_mean, dt_sum, obs_next = [], [], [], []
        usable = [e for e in self.episodes
                  if float(e["dts"].sum()) > 2.0 * seconds]
        while len(obs_t) < batch:
            e = usable[rng.integers(len(usable))]
            t = int(rng.integers(0, len(e["actions"])))
            got = self.steps_for_duration(e["dts"], t, seconds)
            if got is None:
                continue
            k, actual = got
            a = e["actions"][t:t + k]
            d = e["dts"][t:t + k]
            obs_t.append(e["obs"][t])
            act_mean.append((a * d[:, None]).sum(axis=0) / d.sum())
            dt_sum.append([actual])
            obs_next.append(e["obs"][t + k])
        return (np.array(obs_t), np.array(act_mean),
                np.array(dt_sum, dtype=np.float32), np.array(obs_next))

    # ----------------------------------------------------------- provenance

    @staticmethod
    def build_meta(config: dict, seed_range: tuple[int, int] | None = None,
                   policy_checkpoint: str | None = None,
                   extra: dict | None = None) -> dict:
        meta = {
            "schema_version": BUFFER_SCHEMA_VERSION,
            "semantics_version": SEMANTICS_VERSION,
            "git": git_info(),
            "config": {
                "env": config.get("env"),
                "agent": {
                    k: config.get("agent", {}).get(k)
                    for k in ("policy_input", "snn_semantics",
                              "timing_convention", "mask_direct_dt",
                              "snn_time_aware", "cfc_time_aware",
                              "use_input_adapter", "adapter_dim")
                },
                "world_model": {
                    k: config.get("world_model", {}).get(k)
                    for k in ("chunk_seconds", "chunk_steps")
                },
            },
            "seed_range": list(seed_range) if seed_range else None,
            "policy_checkpoint": policy_checkpoint,
        }
        if extra:
            meta.update(extra)
        return _sanitize(meta)

    def expected_compat(self) -> dict:
        return {
            "schema_version": BUFFER_SCHEMA_VERSION,
            "semantics_version": SEMANTICS_VERSION,
        }

    @staticmethod
    def fingerprint_name(meta: dict, data_dir: str) -> str:
        """Fingerprinted buffer path: experience_sv{N}_{sha8}.npz."""
        blob = json.dumps(_sanitize(meta), sort_keys=True).encode("utf-8")
        sha8 = hashlib.sha256(blob).hexdigest()[:8]
        sv = meta.get("semantics_version", SEMANTICS_VERSION)
        return os.path.join(data_dir, f"experience_sv{sv}_{sha8}.npz")

    def save(self, path: str, meta: dict | None = None):
        if meta is not None:
            self.meta = dict(meta)
        self.meta.setdefault("schema_version", BUFFER_SCHEMA_VERSION)
        self.meta.setdefault("semantics_version", SEMANTICS_VERSION)
        meta_json = json.dumps(_sanitize(self.meta), sort_keys=True)
        np.savez_compressed(
            path,
            n=len(self.episodes),
            meta_json=np.asarray(meta_json),
            **{f"obs_{i}": e["obs"] for i, e in enumerate(self.episodes)},
            **{f"act_{i}": e["actions"] for i, e in enumerate(self.episodes)},
            **{f"dt_{i}": e["dts"] for i, e in enumerate(self.episodes)},
            **{f"tcap_{i}": e["t_capture"] for i, e in enumerate(self.episodes)},
            **{f"tdel_{i}": e["t_delivery"] for i, e in enumerate(self.episodes)})
        # Record content checksum after write for load-time reporting
        self.meta["file_sha256"] = file_checksum(path)
        return path

    @classmethod
    def load(cls, path: str, expected_compat: dict | None = None,
             allow_legacy_buffer: bool = False) -> "ReplayBuffer":
        data = np.load(path, allow_pickle=False)
        meta: dict[str, Any] = {}
        if "meta_json" in data:
            raw = data["meta_json"]
            # np.savez stores a 0-d unicode/object array or bytes
            if hasattr(raw, "item"):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            meta = json.loads(str(raw))
        else:
            if not allow_legacy_buffer:
                raise ValueError(
                    f"replay buffer {path} has no provenance metadata "
                    f"(pre-semantics_version-3 artifact). Refuse silent "
                    f"reuse across semantic changes. Re-collect experience, "
                    f"or pass allow_legacy_buffer=True / --allow-legacy-buffer "
                    f"and record the override in the world-model metadata.")
            meta = {"legacy_buffer": True, "schema_version": 1,
                    "semantics_version": None, "path": path}

        if expected_compat is None and not meta.get("legacy_buffer"):
            expected_compat = {
                "schema_version": BUFFER_SCHEMA_VERSION,
                "semantics_version": SEMANTICS_VERSION,
            }
        if expected_compat is not None and not meta.get("legacy_buffer"):
            diffs = {k: {"buffer": meta.get(k), "current": v}
                     for k, v in expected_compat.items()
                     if meta.get(k) != v}
            if diffs:
                if not allow_legacy_buffer:
                    lines = "\n".join(
                        f"  {k}: buffer={v['buffer']!r} current={v['current']!r}"
                        for k, v in diffs.items())
                    raise ValueError(
                        f"replay buffer {path} is incompatible with the "
                        f"current semantics:\n{lines}\n"
                        f"Re-collect under the new semantics, or pass "
                        f"--allow-legacy-buffer to override deliberately.")
                meta["legacy_override"] = True
                meta["compat_diffs"] = diffs

        buf = cls(meta=meta)
        for i in range(int(data["n"])):
            tcap = data[f"tcap_{i}"] if f"tcap_{i}" in data else None
            tdel = data[f"tdel_{i}"] if f"tdel_{i}" in data else None
            buf.add_episode(data[f"obs_{i}"], data[f"act_{i}"],
                            data[f"dt_{i}"], tcap, tdel)
        return buf
