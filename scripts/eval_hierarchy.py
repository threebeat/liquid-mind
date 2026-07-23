"""Hierarchy attribution evaluation (Priorities 5, 6, 7).

Runs the following variants on IDENTICAL paired environment seeds
(default 100) on the U-trap layout:

    reactive              original reactive checkpoint (liquid_policy.pt);
    equal_extra_reactive  equal-extra-budget reactive (when checkpoint exists);
    hier_planner          hierarchical checkpoint, real CEM planner active;
    hier_zero             hierarchical checkpoint, subgoal forced to zero;
    hier_random           hierarchical checkpoint, random subgoals;
    hier_shuffled         hierarchical checkpoint, shuffled subgoal sequences;
    hier_heuristic        hierarchical checkpoint, geometric lidar waypoint.

Gate status is passed | failed | incomplete:
  - incomplete until equal-extra reactive is present and evaluated;
  - primary endpoint: paired success improvement;
  - secondary: final-distance improvement;
  - safety: collision duration not unacceptably worse;
  - attribution: planner beats zero/random/shuffled/heuristic;
  - optimization control: planner beats original + equal-extra reactive.

CEM repeats are aggregated within each environment seed before paired
testing; every repeat's planner log is retained.
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import MODELS_DIR, ensure_dirs, load_config
from provenance import checkpoint_ref, gather_provenance, write_results
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv
from scripts.eval_common import (compare, run_episode, summarize,
                                 within_env_planner_variance)

PLANNER_SEED_BASE = 313_000
ENV_SEED_BASE = 777_000

# Safety: planner collision-duration mean increase vs control must stay
# below this absolute seconds threshold when claiming a pass.
COLLISION_WORSEN_MAX_S = 1.0


def _load_wm(cfg, override_gate: bool, allow_legacy: bool):
    from training.train_world_model import load_world_model
    return load_world_model(cfg, override_gate=override_gate,
                            allow_legacy=allow_legacy)


def _make_agent(cfg, mode, policy_file, wm, allow_legacy):
    path = (policy_file if os.path.isabs(policy_file)
            else os.path.join(MODELS_DIR, policy_file))
    agent = HybridAgent(cfg, mode=mode, world_model=wm)
    agent.load(path, allow_legacy=allow_legacy)
    return agent


def _find_equal_extra(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    patterns = [
        os.path.join(MODELS_DIR, "factorial_reactive-extra-budget*.pt"),
        os.path.join(MODELS_DIR, "*reactive-extra-budget*.pt"),
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def _run_variant(cfg, agent, episodes, layout, cem_repeats: int = 1):
    env = NavEnv(cfg, layout=layout)
    records, planner_logs, subgoal_seqs = [], [], []
    for ep in range(episodes):
        env_seed = ENV_SEED_BASE + ep
        ep_subgoals = []
        for rep in range(cem_repeats):
            planner_seed = PLANNER_SEED_BASE + ep * 97 + rep
            agent._subgoal_rng = np.random.default_rng(planner_seed)
            planner = agent.planner if agent.planner is not None else None
            rec = run_episode(env, agent.act, agent.reset, env_seed,
                              planner_seed=planner_seed, planner=planner)
            rec["cem_repeat"] = rep
            records.append(rec)
            if agent.log_planner:
                # Copy before the next reset() clears planner_log
                log_copy = list(agent.planner_log)
                planner_logs.append(log_copy)
                ep_subgoals.append([e["chosen_subgoal"] for e in log_copy])
        if ep_subgoals:
            # Prefer the first repeat's sequence for shuffled replay
            subgoal_seqs.append(ep_subgoals[0])
    env.close()
    return records, planner_logs, subgoal_seqs


def _planner_validation(planner_logs: list[list[dict]]) -> dict:
    """On-policy world-model check: prediction made at plan k for the next
    macro-step versus the realized readout observed at plan k+1."""
    rows = []
    for ep_log in planner_logs:
        for e in ep_log:
            if "prev_predicted_readout" not in e:
                continue
            p = np.asarray(e["prev_predicted_readout"], dtype=np.float64)
            r = np.asarray(e["realized_readout"], dtype=np.float64)
            rows.append({
                "goal_err_m": abs(p[0] - r[0]) * 10.0,
                "bearing_err": abs(float(np.angle(np.exp(
                    1j * (math.atan2(p[1], p[2]) - math.atan2(r[1], r[2])))))),
                "quad_err": float(np.mean(np.abs(p[3:] - r[3:]))),
                "best_score": e["best_score"],
            })
    if not rows:
        return {"n": 0}
    scores = np.asarray([x["best_score"] for x in rows])
    terciles = np.percentile(scores, [33.3, 66.7])
    strata = {"low_score": [], "mid_score": [], "high_score": []}
    for x in rows:
        if x["best_score"] <= terciles[0]:
            strata["low_score"].append(x)
        elif x["best_score"] <= terciles[1]:
            strata["mid_score"].append(x)
        else:
            strata["high_score"].append(x)

    def _agg(sub):
        return {"n": len(sub),
                "goal_err_m": float(np.mean([x["goal_err_m"] for x in sub]))
                if sub else None,
                "bearing_err_rad": float(np.mean([x["bearing_err"]
                                                  for x in sub])) if sub else None,
                "quad_err": float(np.mean([x["quad_err"] for x in sub]))
                if sub else None}
    out = {"overall": _agg(rows)}
    out.update({k: _agg(v) for k, v in strata.items()})
    return out


def _beat_control(c: dict) -> dict:
    """Primary: success_rate improvement with CI excluding 0 (or McNemar).
    Secondary: final-distance improvement. Safety: collision not much worse.
    """
    sdiff = c["success_rate_diff"]
    primary = bool(sdiff["mean_diff"] > 0 and sdiff["excludes_zero"])
    # Also accept clear McNemar evidence when rates are sparse
    mcn = c["success_mcnemar"]
    if not primary and mcn["discordant"] > 0 and mcn["p_value"] < 0.05:
        primary = mcn["a_only_success"] > mcn["b_only_success"]
    fdiff = c["final_dist_diff"]
    secondary = bool(fdiff["mean_diff"] < 0 and fdiff["excludes_zero"])
    cdiff = c["collision_duration_diff"]
    safety_ok = bool(cdiff["mean_diff"] <= COLLISION_WORSEN_MAX_S)
    return {
        "primary_success": primary,
        "secondary_final_dist": secondary,
        "safety_collision_ok": safety_ok,
        "passed": bool(primary and safety_ok),
    }


def main(episodes: int = 100, layout: str = "u_trap",
         override_wm_gate: bool = False, allow_legacy: bool = False,
         cem_repeats: int = 1,
         equal_extra_checkpoint: str | None = None):
    cfg = load_config()
    ensure_dirs()
    wm, wm_meta = _load_wm(cfg, override_wm_gate, allow_legacy)

    reactive_file = "liquid_policy.pt"
    hier_file = "hier_policy.pt"
    equal_extra_path = _find_equal_extra(equal_extra_checkpoint)
    results, all_records = {}, {}

    # 1) real planner first: collects subgoal norms/sequences for controls
    agent = _make_agent(cfg, "hierarchical", hier_file, wm, allow_legacy)
    agent.subgoal_source = "planner"
    agent.log_planner = True
    print(f"[eval-hier] hier_planner ({episodes} episodes on {layout}, "
          f"cem_repeats={cem_repeats})...")
    recs, planner_logs, subgoal_seqs = _run_variant(
        cfg, agent, episodes, layout, cem_repeats)
    all_records["hier_planner"] = recs
    results["planner_validation_on_policy"] = _planner_validation(planner_logs)
    results["planner_logs_summary"] = {
        "invocations": int(sum(len(l) for l in planner_logs)),
        "n_repeat_logs": len(planner_logs),
        "mean_best_score": float(np.mean(
            [e["best_score"] for l in planner_logs for e in l]))
        if planner_logs else None}
    results["planner_within_env_variance"] = within_env_planner_variance(recs)

    norms = [float(np.linalg.norm(g)) for l in subgoal_seqs for g in l]
    mean_norm = float(np.mean(norms)) if norms else 1.0

    # 2) controls on the SAME seeds (including shuffled with cem_repeats)
    def control(name, source, setup=None):
        a = _make_agent(cfg, "hierarchical", hier_file, wm, allow_legacy)
        a.subgoal_source = source
        if setup:
            setup(a)
        print(f"[eval-hier] {name}...")
        r, _, _ = _run_variant(cfg, a, episodes, layout, cem_repeats)
        all_records[name] = r

    control("hier_zero", "zero")
    control("hier_random", "random",
            lambda a: setattr(a, "random_subgoal_norm", mean_norm))

    a = _make_agent(cfg, "hierarchical", hier_file, wm, allow_legacy)
    a.subgoal_source = "shuffled"
    print("[eval-hier] hier_shuffled...")
    env = NavEnv(cfg, layout=layout)
    recs = []
    for ep in range(episodes):
        if subgoal_seqs:
            a.scripted_subgoals = subgoal_seqs[(ep + 1) % len(subgoal_seqs)]
        for rep in range(cem_repeats):
            planner_seed = PLANNER_SEED_BASE + ep * 97 + rep
            a._subgoal_rng = np.random.default_rng(planner_seed)
            rec = run_episode(env, a.act, a.reset, ENV_SEED_BASE + ep,
                              planner_seed=planner_seed, planner=a.planner)
            rec["cem_repeat"] = rep
            recs.append(rec)
    env.close()
    all_records["hier_shuffled"] = recs

    control("hier_heuristic", "heuristic")

    # 3) reactive checkpoints on the same seeds
    a = _make_agent(cfg, "reactive", reactive_file, None, allow_legacy)
    print("[eval-hier] reactive...")
    recs, _, _ = _run_variant(cfg, a, episodes, layout, cem_repeats)
    all_records["reactive"] = recs

    missing_controls = []
    if equal_extra_path:
        print(f"[eval-hier] equal_extra_reactive ({equal_extra_path})...")
        a = _make_agent(cfg, "reactive", equal_extra_path, None, allow_legacy)
        recs, _, _ = _run_variant(cfg, a, episodes, layout, cem_repeats)
        all_records["equal_extra_reactive"] = recs
    else:
        missing_controls.append("equal_extra_reactive")
        print("[eval-hier] equal_extra_reactive MISSING — gate incomplete "
              "(train via run_timing_factorial.py --include-extra-reactive)")

    # ------------------------------------------------------------- report
    results["variants"] = {}
    for name, recs in all_records.items():
        s = summarize(recs)
        results["variants"][name] = {"summary": s, "episodes": recs}
        print(f"[eval-hier] {name:22s}: "
              f"R={s['reward_mean']['point']:7.2f} "
              f"[{s['reward_mean']['lo']:.2f},{s['reward_mean']['hi']:.2f}]  "
              f"success={s['success']['k']}/{s['success']['n']} "
              f"[{s['success']['lo']:.2f},{s['success']['hi']:.2f}]  "
              f"final_dist={s['final_goal_dist_mean']['point']:.2f} m")

    required = ["reactive", "hier_zero", "hier_random", "hier_shuffled",
                "hier_heuristic", "equal_extra_reactive"]
    results["comparisons"] = {}
    criteria = {}
    for other in required:
        if other not in all_records:
            criteria[other] = {"available": False, "passed": False}
            continue
        c = compare(all_records["hier_planner"], all_records[other],
                    "hier_planner", other)
        results["comparisons"][f"hier_planner_vs_{other}"] = c
        beat = _beat_control(c)
        criteria[other] = {"available": True, **beat}
        d = c["success_rate_diff"]
        print(f"[eval-hier] planner vs {other:22s}: "
              f"dSuccess={d['mean_diff']:+.3f} [{d['lo']:.3f},{d['hi']:.3f}] "
              f"{'(excludes 0)' if d['excludes_zero'] else '(includes 0)'}  "
              f"McNemar p={c['success_mcnemar']['p_value']:.3f}  "
              f"beat={'YES' if beat['passed'] else 'no'}")

    if missing_controls:
        status = "incomplete"
        passed = False
    elif all(criteria[k]["passed"] for k in required):
        status = "passed"
        passed = True
    else:
        status = "failed"
        passed = False

    results["hierarchy_gate"] = {
        "status": status,
        "passed": passed,
        "criteria": criteria,
        "missing_controls": missing_controls,
        "primary_endpoint": "paired_success_rate",
        "note": ("Gate remains incomplete until equal-extra-reactive exists "
                 "and is evaluated. Pass requires beating all six controls "
                 "on success (primary) without unacceptable collision "
                 "increase."),
    }
    print(f"[eval-hier] hierarchy gate status={status.upper()} "
          f"(passed={passed})")

    ckpts = {
        "reactive": checkpoint_ref(os.path.join(MODELS_DIR, reactive_file)),
        "hierarchical": checkpoint_ref(os.path.join(MODELS_DIR, hier_file)),
        "world_model": checkpoint_ref(os.path.join(MODELS_DIR, "world_model.pt")),
    }
    if equal_extra_path:
        ckpts["equal_extra_reactive"] = checkpoint_ref(equal_extra_path)

    wm_gate = wm_meta.get("gate", {})
    results["meta"] = gather_provenance(
        cfg, experiment_name=f"eval_hierarchy_{layout}",
        seeds={"env_seed_base": ENV_SEED_BASE,
               "planner_seed_base": PLANNER_SEED_BASE,
               "episodes": episodes, "cem_repeats": cem_repeats},
        extra={"checkpoints": ckpts,
               "wm_gate": wm_gate.get("passed")
               if isinstance(wm_gate, dict) else None,
               "wm_gate_status": (wm_gate.get("status")
                                  if isinstance(wm_gate, dict) else None)})
    out = write_results(f"hierarchy_{layout}", results)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--layout", default="u_trap",
                    choices=["u_trap", "random"])
    ap.add_argument("--override-wm-gate", action="store_true")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="permit explicitly imported legacy checkpoints")
    ap.add_argument("--cem-repeats", type=int, default=1,
                    help="CEM seeds per environment seed (planner "
                         "stochasticity vs environment variability)")
    ap.add_argument("--equal-extra-checkpoint", type=str, default=None,
                    help="path to equal-extra-budget reactive checkpoint")
    a = ap.parse_args()
    main(a.episodes, a.layout, a.override_wm_gate, a.allow_legacy,
         a.cem_repeats, a.equal_extra_checkpoint)
