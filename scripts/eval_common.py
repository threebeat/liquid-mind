"""Shared evaluation machinery (Priority 5).

Every evaluation episode produces a full record (seeds, checkpoints, reward,
success, distances, path geometry, collision integrals, action statistics,
termination reason) that is saved verbatim in the result JSON — aggregates
are always recomputable and paired comparisons are always possible.

Statistics:
  - bootstrap confidence intervals for means and medians;
  - Wilson score intervals for success proportions;
  - paired bootstrap intervals for reward / final-distance differences on
    identical seeds;
  - paired success discordance counts + exact McNemar (binomial) test.

No normal approximations on 10-episode pilots.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ------------------------------------------------------------ episode runs


def run_episode(env, act_fn, reset_fn, seed: int,
                planner_seed: int | None = None,
                timing_seed: int | None = None,
                planner=None) -> dict:
    """Run one episode and return the full per-episode record.

    act_fn(obs, dt) -> action; reset_fn() resets agent state.
    The environment's np_random drives both layout and timing; a distinct
    timing seed would require a separate stream (recorded for provenance
    even while identical to the env seed).

    If `planner` is a CEMPlanner (or object with .seed), it is seeded from
    planner_seed via its private torch.Generator — not the global RNG.
    """
    if planner is not None and planner_seed is not None and hasattr(planner, "seed"):
        planner.seed(int(planner_seed))
    obs, info = env.reset(seed=int(seed))
    reset_fn()
    start_dist = float(info["goal_dist"])
    prev_pos = np.asarray(info["pos"])
    prev_action = None

    total = 0.0
    path_len = 0.0
    contact_duration = 0.0
    collision_entries = 0
    action_mags, action_switches = [], []
    steps = 0
    done = False
    while not done:
        action = np.asarray(act_fn(obs, env._last_dt), dtype=np.float64)
        obs, r, term, trunc, info = env.step(action)
        total += r
        steps += 1
        pos = np.asarray(info["pos"])
        path_len += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        contact_duration += float(info.get("contact_duration", 0.0))
        collision_entries += int(info.get("collision_entries", 0))
        action_mags.append(float(np.abs(action).mean()))
        if prev_action is not None:
            action_switches.append(float(np.abs(action - prev_action).mean()))
        prev_action = action
        done = term or trunc

    if term:
        reason = "success"
    elif env._step_count >= env.max_episode_steps:
        reason = "step_cap"
    else:
        reason = "time_limit"
    return {
        "env_seed": int(seed),
        "timing_seed": int(timing_seed if timing_seed is not None else seed),
        "planner_seed": (int(planner_seed) if planner_seed is not None
                         else None),
        "reward": float(total),
        "success": bool(info["is_success"]),
        "final_goal_dist": float(info["goal_dist"]),
        "sim_duration": float(info.get("sim_time", 0.0)),
        "decision_count": steps,
        "path_length": float(path_len),
        "straight_line_start_dist": start_dist,
        "path_efficiency": float(start_dist / max(path_len, 1e-6)),
        "collision_duration": float(contact_duration),
        "collision_entries": int(collision_entries),
        "time_to_goal": (float(info.get("sim_time", 0.0)) if term else None),
        "mean_action_magnitude": float(np.mean(action_mags)) if action_mags
        else 0.0,
        "action_switching_rate": (float(np.mean(action_switches))
                                  if action_switches else 0.0),
        "termination": reason,
    }


# ---------------------------------------------------------------- statistics


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"point": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return {"point": float(stat(values)),
            "lo": float(np.percentile(boots, 100 * alpha / 2)),
            "hi": float(np.percentile(boots, 100 * (1 - alpha / 2)))}


def wilson_interval(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a success proportion."""
    if n == 0:
        return {"point": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"point": p, "lo": max(0.0, center - half),
            "hi": min(1.0, center + half), "k": int(k), "n": int(n)}


def paired_bootstrap_diff(a, b, n_boot: int = 10_000, alpha: float = 0.05,
                          seed: int = 0) -> dict:
    """CI on mean(a - b) for values paired on identical seeds."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b), "paired comparison needs identical seed lists"
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean_diff": float(d.mean()), "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "n_pairs": len(d)}


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value against p=0.5."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    pk = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= pk + 1e-15)))


def mcnemar_exact(success_a, success_b) -> dict:
    """Exact McNemar test on paired binary outcomes (identical seeds).
    b01 = A failed but B succeeded; b10 = A succeeded but B failed."""
    a = np.asarray(success_a, dtype=bool)
    b = np.asarray(success_b, dtype=bool)
    assert len(a) == len(b)
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    return {"a_only_success": b10, "b_only_success": b01,
            "discordant": b01 + b10,
            "p_value": _binom_two_sided_p(min(b01, b10), b01 + b10)}


def summarize(records: list[dict]) -> dict:
    """Aggregate a list of per-episode records with proper uncertainty."""
    rw = [r["reward"] for r in records]
    fd = [r["final_goal_dist"] for r in records]
    k = sum(r["success"] for r in records)
    return {
        "episodes": len(records),
        "reward_mean": bootstrap_ci(rw, np.mean),
        "reward_median": bootstrap_ci(rw, np.median),
        "success": wilson_interval(k, len(records)),
        "final_goal_dist_mean": bootstrap_ci(fd, np.mean),
        "path_efficiency_mean": bootstrap_ci(
            [r["path_efficiency"] for r in records], np.mean),
        "collision_duration_mean": bootstrap_ci(
            [r["collision_duration"] for r in records], np.mean),
    }


def aggregate_by_env_seed(records: list[dict]) -> list[dict]:
    """Collapse CEM repeats into one inferential unit per environment seed.

    Numeric metrics are averaged; success becomes a rate in [0, 1] and a
    binary majority vote (for McNemar). Within-seed reward variance is kept
    when multiple repeats exist.
    """
    by_seed: dict[int, list[dict]] = {}
    for r in records:
        by_seed.setdefault(int(r["env_seed"]), []).append(r)
    out = []
    for seed in sorted(by_seed):
        group = by_seed[seed]
        rewards = [float(r["reward"]) for r in group]
        successes = [float(bool(r["success"])) for r in group]
        dists = [float(r["final_goal_dist"]) for r in group]
        coll = [float(r["collision_duration"]) for r in group]
        rate = float(np.mean(successes))
        out.append({
            "env_seed": seed,
            "n_repeats": len(group),
            "reward": float(np.mean(rewards)),
            "reward_std_within": (float(np.std(rewards))
                                  if len(rewards) > 1 else 0.0),
            "success": rate >= 0.5,
            "success_rate": rate,
            "final_goal_dist": float(np.mean(dists)),
            "collision_duration": float(np.mean(coll)),
            "path_efficiency": float(np.mean(
                [r["path_efficiency"] for r in group])),
            "planner_seeds": [r.get("planner_seed") for r in group],
            "cem_repeats": [r.get("cem_repeat") for r in group],
        })
    return out


def within_env_planner_variance(records: list[dict]) -> dict:
    """Report within-environment planner stochasticity when cem_repeats > 1."""
    by_seed: dict[int, list[dict]] = {}
    for r in records:
        by_seed.setdefault(int(r["env_seed"]), []).append(r)
    multi = [g for g in by_seed.values() if len(g) > 1]
    if not multi:
        return {"n_env_with_repeats": 0}
    reward_stds = [float(np.std([r["reward"] for r in g])) for g in multi]
    success_stds = [float(np.std([float(r["success"]) for r in g]))
                    for g in multi]
    return {
        "n_env_with_repeats": len(multi),
        "mean_within_env_reward_std": float(np.mean(reward_stds)),
        "mean_within_env_success_std": float(np.mean(success_stds)),
    }


def compare(records_a: list[dict], records_b: list[dict],
            name_a: str, name_b: str,
            aggregate_repeats: bool = True) -> dict:
    """Paired comparison of two variants on identical environment seeds.

    When cem_repeats > 1 produced multiple records per env_seed, repeats are
    aggregated within each seed before pairing (environment seed is the
    inferential unit).
    """
    a_recs = (aggregate_by_env_seed(records_a) if aggregate_repeats
              else records_a)
    b_recs = (aggregate_by_env_seed(records_b) if aggregate_repeats
              else records_b)
    sa = {r["env_seed"]: r for r in a_recs}
    sb = {r["env_seed"]: r for r in b_recs}
    seeds = sorted(set(sa) & set(sb))
    a = [sa[s] for s in seeds]
    b = [sb[s] for s in seeds]
    success_a = [r.get("success_rate", float(r["success"])) for r in a]
    success_b = [r.get("success_rate", float(r["success"])) for r in b]
    return {
        "a": name_a, "b": name_b, "n_pairs": len(seeds),
        "aggregated_repeats": bool(aggregate_repeats),
        "reward_diff": paired_bootstrap_diff(
            [r["reward"] for r in a], [r["reward"] for r in b]),
        "final_dist_diff": paired_bootstrap_diff(
            [r["final_goal_dist"] for r in a],
            [r["final_goal_dist"] for r in b]),
        "success_rate_diff": paired_bootstrap_diff(success_a, success_b),
        "collision_duration_diff": paired_bootstrap_diff(
            [r["collision_duration"] for r in a],
            [r["collision_duration"] for r in b]),
        "success_mcnemar": mcnemar_exact(
            [r["success"] for r in a], [r["success"] for r in b]),
    }
