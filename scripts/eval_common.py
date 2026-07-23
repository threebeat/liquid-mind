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
    b01 = A failed but B succeeded; b10 = A succeeded but B failed.
    Pairs where either outcome is None (majority ties under repeated runs)
    are dropped and counted in dropped_tied_pairs."""
    assert len(success_a) == len(success_b)
    pairs = [(x, y) for x, y in zip(success_a, success_b)
             if x is not None and y is not None]
    dropped = len(success_a) - len(pairs)
    a = np.asarray([p[0] for p in pairs], dtype=bool)
    b = np.asarray([p[1] for p in pairs], dtype=bool)
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    return {"a_only_success": b10, "b_only_success": b01,
            "discordant": b01 + b10,
            "dropped_tied_pairs": dropped,
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

    Numeric metrics are averaged; success becomes a fractional rate in
    [0, 1]. The majority-vote binary `success` is DESCRIPTIVE only: it
    requires a strict majority (rate > 0.5); an exact 50% tie under an even
    repeat count yields success=None and success_tied=True — a tie is never
    classified as a success.
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
        if rate > 0.5:
            majority = True
        elif rate < 0.5:
            majority = False
        else:
            majority = None                      # tie: never counted a success
        out.append({
            "env_seed": seed,
            "n_repeats": len(group),
            "reward": float(np.mean(rewards)),
            "reward_std_within": (float(np.std(rewards))
                                  if len(rewards) > 1 else 0.0),
            "success": majority,
            "success_tied": majority is None,
            "success_rate": rate,
            "final_goal_dist": float(np.mean(dists)),
            "collision_duration": float(np.mean(coll)),
            "path_efficiency": float(np.mean(
                [r["path_efficiency"] for r in group])),
            "planner_seeds": [r.get("planner_seed") for r in group],
            "cem_repeats": [r.get("cem_repeat") for r in group],
        })
    return out


def summarize_env_level(records: list[dict], n_boot: int = 10_000,
                        seed: int = 0) -> dict:
    """Environment-level inferential summary: repeats are aggregated within
    each environment seed FIRST, then env seeds are the bootstrap units.

    Success: with one repeat per environment the outcomes are binary and a
    Wilson interval applies. With repeats > 1 the per-environment values are
    fractional frequencies (0.0/0.5/1.0, ...), so an environment-cluster
    bootstrap of the mean frequency is used instead — Wilson would be wrong.
    """
    units = aggregate_by_env_seed(records)
    if not units:
        return {"env_units": 0}
    max_rep = max(u["n_repeats"] for u in units)
    out = {
        "env_units": len(units),
        "max_repeats": int(max_rep),
        "note": "environment seed is the inferential unit; CEM repeats are "
                "aggregated within each seed",
        "reward_mean": bootstrap_ci([u["reward"] for u in units], np.mean,
                                    n_boot=n_boot, seed=seed),
        "final_goal_dist_mean": bootstrap_ci(
            [u["final_goal_dist"] for u in units], np.mean,
            n_boot=n_boot, seed=seed),
        "collision_duration_mean": bootstrap_ci(
            [u["collision_duration"] for u in units], np.mean,
            n_boot=n_boot, seed=seed),
        "path_efficiency_mean": bootstrap_ci(
            [u["path_efficiency"] for u in units], np.mean,
            n_boot=n_boot, seed=seed),
    }
    if max_rep == 1:
        k = sum(1 for u in units if u["success"])
        out["success"] = wilson_interval(k, len(units))
        out["success_rate_mean"] = None
    else:
        out["success"] = None
        out["success_rate_mean"] = bootstrap_ci(
            [u["success_rate"] for u in units], np.mean,
            n_boot=n_boot, seed=seed)
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
    inferential unit). The paired success-frequency difference is then the
    sole confirmatory success statistic; McNemar on majority-collapsed
    repeated stochastic runs is NOT confirmatory and is reported only as
    descriptive, marked applicable=False.
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
    success_a = [r.get("success_rate", float(bool(r["success"]))) for r in a]
    success_b = [r.get("success_rate", float(bool(r["success"]))) for r in b]
    max_rep = max([int(r.get("n_repeats", 1)) for r in a + b], default=1)
    mcn_raw = mcnemar_exact([r["success"] for r in a],
                            [r["success"] for r in b])
    if max_rep > 1:
        mcnemar = {"applicable": False,
                   "reason": "repeated stochastic planner runs: majority-"
                             "collapsed McNemar is not confirmatory; use "
                             "success_rate_diff",
                   "descriptive": mcn_raw}
    else:
        mcnemar = {"applicable": True, **mcn_raw}
    return {
        "a": name_a, "b": name_b, "n_pairs": len(seeds),
        "aggregated_repeats": bool(aggregate_repeats),
        "max_repeats": max_rep,
        "reward_diff": paired_bootstrap_diff(
            [r["reward"] for r in a], [r["reward"] for r in b]),
        "final_dist_diff": paired_bootstrap_diff(
            [r["final_goal_dist"] for r in a],
            [r["final_goal_dist"] for r in b]),
        "success_rate_diff": paired_bootstrap_diff(success_a, success_b),
        "collision_duration_diff": paired_bootstrap_diff(
            [r["collision_duration"] for r in a],
            [r["collision_duration"] for r in b]),
        "success_mcnemar": mcnemar,
    }


# ---------------------------------------- degradation / difference-in-diff

# Sign convention for robustness/degradation effects: values are transformed
# so that HIGHER always means MORE ROBUST (less degradation). Cost metrics
# (final_goal_dist, collision_duration; lower is better) are negated.
METRIC_HIGHER_BETTER = {"reward": True, "success_rate": True,
                        "final_goal_dist": False, "collision_duration": False}
SIGN_CONVENTION = ("higher robustness value = less degradation; in cell "
                   "contrasts and factorial effects, positive = first/"
                   "physical variant more robust (cost metrics negated)")


def robustness_by_env(level_records: list[dict], fixed_records: list[dict],
                      metric: str):
    """Per-environment robustness R[s] = sign * (Y_level[s] - Y_fixed[s]).

    Repeats are aggregated within env seed first. sign = +1 for higher-is-
    better metrics, -1 for cost metrics, so higher R always means more
    robust. Returns (dict env_seed -> R, missing-seed report).
    """
    sign = 1.0 if METRIC_HIGHER_BETTER[metric] else -1.0
    la = {u["env_seed"]: u for u in aggregate_by_env_seed(level_records)}
    fx = {u["env_seed"]: u for u in aggregate_by_env_seed(fixed_records)}
    shared = sorted(set(la) & set(fx))
    missing = {"level_only": sorted(set(la) - set(fx)),
               "fixed_only": sorted(set(fx) - set(la))}
    out = {s: sign * (float(la[s][metric]) - float(fx[s][metric]))
           for s in shared}
    return out, missing


def paired_did(rob_a: dict, rob_b: dict, n_boot: int = 10_000,
               seed: int = 0) -> dict:
    """Difference-in-differences: robustness of A minus robustness of B on
    shared environment seeds. Positive = A more robust. Missing seeds are
    REPORTED, never silently intersected away."""
    shared = sorted(set(rob_a) & set(rob_b))
    missing_in_b = sorted(set(rob_a) - set(rob_b))
    missing_in_a = sorted(set(rob_b) - set(rob_a))
    if not shared:
        return {"mean_diff": None, "lo": None, "hi": None,
                "excludes_zero": False, "n_pairs": 0,
                "missing_in_a": missing_in_a, "missing_in_b": missing_in_b}
    d = paired_bootstrap_diff([rob_a[s] for s in shared],
                              [rob_b[s] for s in shared],
                              n_boot=n_boot, seed=seed)
    d["missing_in_a"] = missing_in_a
    d["missing_in_b"] = missing_in_b
    return d


def factorial_effects(d00: dict, d10: dict, d01: dict, d11: dict,
                      n_boot: int = 10_000, seed: int = 0) -> dict:
    """2x2 factorial effects on per-environment robustness dicts.

    D_ij = robustness with SNN factor i and CfC factor j (1 = physical).
    Positive = physical variant more robust.

        SNN main effect = ((D10 - D00) + (D11 - D01)) / 2
        CfC main effect = ((D01 - D00) + (D11 - D10)) / 2
        Interaction     =  D11 - D10 - D01 + D00

    Effects are computed per shared environment seed and bootstrapped over
    environments (env-paired within one training replicate).
    """
    shared = sorted(set(d00) & set(d10) & set(d01) & set(d11))
    if not shared:
        return {"n_env": 0, "snn_main_effect": None,
                "cfc_main_effect": None, "interaction": None}
    a00 = np.asarray([d00[s] for s in shared], dtype=np.float64)
    a10 = np.asarray([d10[s] for s in shared], dtype=np.float64)
    a01 = np.asarray([d01[s] for s in shared], dtype=np.float64)
    a11 = np.asarray([d11[s] for s in shared], dtype=np.float64)
    snn = ((a10 - a00) + (a11 - a01)) / 2.0
    cfc = ((a01 - a00) + (a11 - a10)) / 2.0
    inter = a11 - a10 - a01 + a00

    def _boot(v, s):
        ci = bootstrap_ci(v, np.mean, n_boot=n_boot, seed=s)
        ci["excludes_zero"] = bool(ci["lo"] is not None
                                   and (ci["lo"] > 0 or ci["hi"] < 0))
        return ci

    return {"n_env": len(shared),
            "snn_main_effect": _boot(snn, seed),
            "cfc_main_effect": _boot(cfc, seed + 1),
            "interaction": _boot(inter, seed + 2)}
