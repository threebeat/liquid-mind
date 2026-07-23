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
"""
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.episodes: list[dict] = []

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
        (obs_t, duration-weighted mean action, actual duration, obs_next)."""
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

    # ---------------------------------------------------------------- io

    def save(self, path: str):
        np.savez_compressed(
            path,
            n=len(self.episodes),
            **{f"obs_{i}": e["obs"] for i, e in enumerate(self.episodes)},
            **{f"act_{i}": e["actions"] for i, e in enumerate(self.episodes)},
            **{f"dt_{i}": e["dts"] for i, e in enumerate(self.episodes)},
            **{f"tcap_{i}": e["t_capture"] for i, e in enumerate(self.episodes)},
            **{f"tdel_{i}": e["t_delivery"] for i, e in enumerate(self.episodes)})

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        data = np.load(path)
        buf = cls()
        for i in range(int(data["n"])):
            tcap = data[f"tcap_{i}"] if f"tcap_{i}" in data else None
            tdel = data[f"tdel_{i}"] if f"tdel_{i}" in data else None
            buf.add_episode(data[f"obs_{i}"], data[f"act_{i}"],
                            data[f"dt_{i}"], tcap, tdel)
        return buf
