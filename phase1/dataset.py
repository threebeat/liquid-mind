"""Ordered-window dataset over ReplayBuffer episodes (Phase 1).

Each sample is an ORDERED window anchored at observation index t of one
episode (never crossing an episode boundary):

    lidar          [L, 16]   obs[..., 0:16] over the context
    body           [L, 2]    obs[..., 19:21] (fwd speed/1.2, yaw rate/3)
    actions        [L, 2]    a_j issued at each context step (ordered,
                             never averaged)
    prev_actions   [L, 2]    a_{j-1} (zero before the first decision) --
                             the body specialist's declared action input
    dts            [L, 1]    seconds elapsed since the previous observation
                             (nominal dt for the episode's first obs)
    valid_mask     [L]       1 where the context step exists (left-padded
                             windows near the episode start)
    future_lidar   [Hmax,16] obs[t+1 .. t+Hmax, 0:16]
    future_body    [Hmax, 2]
    future_actions [Hmax, 2] a_t .. a_{t+Hmax-1} (ordered)
    future_dts     [Hmax, 1] seconds

Horizon endpoints (0.25/0.5/1.0/2.0 s at nominal 30 Hz -> 8/15/30/60
decisions) index into the future arrays. Splits are by COMPLETE episode with
a saved manifest (episode indices + buffer checksum) reused identically by
every architecture. Everything is deterministic given a seed.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from phase1 import CONTEXT_LEN, HORIZON_STEPS, MAX_HORIZON_STEPS, NOMINAL_DT
from provenance import _sanitize, file_checksum

MIN_CONTEXT = 5  # anchors need at least this many real context steps


# ----------------------------------------------------------------- splits


def make_episode_split(n_episodes: int, n_train: int, n_val: int,
                       n_test: int, seed: int) -> dict:
    """Random disjoint split by complete episode."""
    if n_train + n_val + n_test > n_episodes:
        raise ValueError(f"split {n_train}+{n_val}+{n_test} exceeds "
                         f"{n_episodes} episodes")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes)
    return {
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "train": sorted(int(i) for i in order[:n_train]),
        "val": sorted(int(i) for i in order[n_train:n_train + n_val]),
        "test": sorted(int(i) for i in
                       order[n_train + n_val:n_train + n_val + n_test]),
    }


def save_split_manifest(path: str, split: dict, buffer_path: str) -> dict:
    manifest = dict(split)
    manifest["buffer_path"] = os.path.basename(buffer_path)
    manifest["buffer_sha256"] = file_checksum(buffer_path)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(manifest[a]) & set(manifest[b]):
            raise ValueError(f"split leak between {a} and {b}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(manifest), f, indent=2)
    return manifest


def load_split_manifest(path: str, buffer_path: str | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if buffer_path is not None:
        actual = file_checksum(buffer_path)
        if actual != manifest["buffer_sha256"]:
            raise ValueError(
                f"split manifest {path} was built for buffer sha256 "
                f"{manifest['buffer_sha256'][:12]}..., but {buffer_path} "
                f"has {actual[:12]}...; refuse mismatched splits")
    return manifest


# ---------------------------------------------------------------- dataset


class WindowDataset:
    """Materializes ordered windows for a fixed set of episode indices."""

    def __init__(self, buffer, episode_indices, context_len: int = CONTEXT_LEN,
                 max_horizon: int = MAX_HORIZON_STEPS, stride: int = 1,
                 min_context: int = MIN_CONTEXT,
                 nominal_dt: float = NOMINAL_DT):
        self.L = int(context_len)
        self.H = int(max_horizon)
        self.stride = int(stride)
        self.min_context = int(min_context)
        self.nominal_dt = float(nominal_dt)
        self.episode_indices = [int(i) for i in episode_indices]
        self.episodes = [buffer.episodes[i] for i in self.episode_indices]
        # anchors: (local_episode_idx, t). Window context ends at obs t,
        # future covers obs t+1..t+H -- entirely inside one episode.
        self.anchors: list[tuple[int, int]] = []
        for li, e in enumerate(self.episodes):
            T = len(e["actions"])            # obs has T+1 rows
            t0 = self.min_context - 1
            for t in range(t0, T - self.H + 1, self.stride):
                self.anchors.append((li, t))

    def __len__(self):
        return len(self.anchors)

    # ------------------------------------------------------------ windows

    def _dt_at_obs(self, e, j: int) -> float:
        """Seconds elapsed before obs j was captured (nominal for j = 0)."""
        return float(e["dts"][j - 1]) if j >= 1 else self.nominal_dt

    def window(self, idx: int) -> dict:
        li, t = self.anchors[idx]
        e = self.episodes[li]
        L, H = self.L, self.H
        out = {
            "lidar": np.zeros((L, 16), np.float32),
            "body": np.zeros((L, 2), np.float32),
            "actions": np.zeros((L, 2), np.float32),
            "prev_actions": np.zeros((L, 2), np.float32),
            "dts": np.zeros((L, 1), np.float32),
            "valid_mask": np.zeros((L,), np.float32),
        }
        obs, acts = e["obs"], e["actions"]
        for i in range(L):
            j = t - L + 1 + i
            if j < 0:
                continue
            out["lidar"][i] = obs[j, 0:16]
            out["body"][i] = obs[j, 19:21]
            out["actions"][i] = acts[j]
            if j >= 1:
                out["prev_actions"][i] = acts[j - 1]
            out["dts"][i, 0] = self._dt_at_obs(e, j)
            out["valid_mask"][i] = 1.0
        out["future_lidar"] = obs[t + 1:t + 1 + H, 0:16].astype(np.float32)
        out["future_body"] = obs[t + 1:t + 1 + H, 19:21].astype(np.float32)
        out["future_actions"] = acts[t:t + H].astype(np.float32)
        out["future_dts"] = e["dts"][t:t + H].astype(np.float32)[:, None]
        return out

    def batch(self, indices) -> dict:
        """Stacked windows: dict of [B, ...] float32 arrays."""
        wins = [self.window(int(i)) for i in indices]
        return {k: np.stack([w[k] for w in wins]) for k in wins[0]}

    def epoch_order(self, epoch_seed: int) -> np.ndarray:
        return np.random.default_rng(epoch_seed).permutation(len(self))

    # -------------------------------------------------------- provenance

    def fingerprint(self, buffer_sha256: str) -> str:
        blob = json.dumps({
            "buffer_sha256": buffer_sha256,
            "episode_indices": self.episode_indices,
            "context_len": self.L, "max_horizon": self.H,
            "stride": self.stride, "min_context": self.min_context,
            "horizon_steps": {str(k): v for k, v in HORIZON_STEPS.items()},
            "nominal_dt": self.nominal_dt,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def horizon_targets(batch: dict, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint targets at a horizon of `steps` decisions."""
    return (batch["future_lidar"][:, steps - 1],
            batch["future_body"][:, steps - 1])
