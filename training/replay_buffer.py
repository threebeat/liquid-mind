"""Episodic replay buffer for world-model training.

Stores whole episodes of (obs, action, dt) and samples chunk-level
transitions: (obs_t, mean action over the chunk, total chunk duration,
obs_{t+chunk}).
"""
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.episodes: list[dict] = []

    def add_episode(self, obs: np.ndarray, actions: np.ndarray,
                    dts: np.ndarray):
        """obs: (T+1, obs_dim), actions: (T, act_dim), dts: (T,)"""
        self.episodes.append({
            "obs": np.asarray(obs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "dts": np.asarray(dts, dtype=np.float32)})

    def __len__(self):
        return len(self.episodes)

    def n_steps(self):
        return sum(len(e["actions"]) for e in self.episodes)

    def sample_chunks(self, batch: int, chunk: int, rng: np.random.Generator):
        obs_t, act_mean, dt_sum, obs_next = [], [], [], []
        usable = [e for e in self.episodes if len(e["actions"]) > chunk]
        for _ in range(batch):
            e = usable[rng.integers(len(usable))]
            t = int(rng.integers(0, len(e["actions"]) - chunk))
            obs_t.append(e["obs"][t])
            act_mean.append(e["actions"][t:t + chunk].mean(axis=0))
            dt_sum.append([e["dts"][t:t + chunk].sum()])
            obs_next.append(e["obs"][t + chunk])
        return (np.array(obs_t), np.array(act_mean),
                np.array(dt_sum, dtype=np.float32), np.array(obs_next))

    def save(self, path: str):
        np.savez_compressed(
            path,
            n=len(self.episodes),
            **{f"obs_{i}": e["obs"] for i, e in enumerate(self.episodes)},
            **{f"act_{i}": e["actions"] for i, e in enumerate(self.episodes)},
            **{f"dt_{i}": e["dts"] for i, e in enumerate(self.episodes)})

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        data = np.load(path)
        buf = cls()
        for i in range(int(data["n"])):
            buf.add_episode(data[f"obs_{i}"], data[f"act_{i}"], data[f"dt_{i}"])
        return buf
