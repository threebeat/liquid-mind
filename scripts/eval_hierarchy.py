"""Hierarchy attribution evaluation (Priorities 5, 6, 7).

Runs the following variants on IDENTICAL paired environment seeds
(default 100) on the U-trap layout:

    reactive        original reactive checkpoint (liquid_policy.pt);
    hier_planner    hierarchical checkpoint, real CEM planner active;
    hier_zero       hierarchical checkpoint, subgoal forced to zero
                    (planner-off control);
    hier_random     hierarchical checkpoint, random subgoals matched in
                    norm and update frequency to the real planner;
    hier_shuffled   hierarchical checkpoint, planner subgoal sequences
                    replayed from a DIFFERENT episode;
    hier_heuristic  hierarchical checkpoint, simple geometric lidar
                    waypoint heuristic as the subgoal source.

These separate: extra optimization, shaping-reward learning, "any changing
subgoal" effects, heuristic-vs-learned-model value, and actual planning.
Success claims require the planner to beat the planner-off, random/shuffled
and heuristic controls — see the paired statistics in the output.

Every planner invocation is logged (candidate-score distribution, chosen
subgoal, predicted readout trajectory) and the realized grounded readout at
the next invocation is compared against the prediction: on-policy
world-model validation on CEM-selected action distributions, stratified by
planner score.

All per-episode records are kept in the result JSON; aggregates use
bootstrap/Wilson intervals and paired McNemar tests. NOTE: language stays
"pilot" until the preregistered gates pass.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import MODELS_DIR, ensure_dirs, load_config
from provenance import checkpoint_ref, gather_provenance, write_results
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv
from scripts.eval_common import compare, run_episode, summarize

PLANNER_SEED_BASE = 313_000
ENV_SEED_BASE = 777_000


def _load_wm(cfg, override_gate: bool, allow_legacy: bool):
    from training.train_world_model import load_world_model
    return load_world_model(cfg, override_gate=override_gate,
                            allow_legacy=allow_legacy)


def _make_agent(cfg, mode, policy_file, wm, allow_legacy):
    agent = HybridAgent(cfg, mode=mode, world_model=wm)
    agent.load(os.path.join(MODELS_DIR, policy_file),
               allow_legacy=allow_legacy)
    return agent


def _run_variant(cfg, agent, episodes, layout, cem_repeats: int = 1):
    env = NavEnv(cfg, layout=layout)
    records, planner_logs, subgoal_seqs = [], [], []
    for ep in range(episodes):
        env_seed = ENV_SEED_BASE + ep
        for rep in range(cem_repeats):
            planner_seed = PLANNER_SEED_BASE + ep * 97 + rep
            agent._subgoal_rng = np.random.default_rng(planner_seed)
            rec = run_episode(env, agent.act, agent.reset, env_seed,
                              planner_seed=planner_seed)
            rec["cem_repeat"] = rep
            records.append(rec)
        if agent.log_planner:
            planner_logs.append(list(agent.planner_log))
            subgoal_seqs.append(
                [e["chosen_subgoal"] for e in agent.planner_log])
    env.close()
    return records, planner_logs, subgoal_seqs


def _planner_validation(planner_logs: list[list[dict]]) -> dict:
    """On-policy world-model check: prediction made at plan k for the next
    macro-step versus the realized readout observed at plan k+1, stratified
    by planner score (does the model err more when the planner is
    confident/unconfident, or on novel action distributions?)."""
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


def main(episodes: int = 100, layout: str = "u_trap",
         override_wm_gate: bool = False, allow_legacy: bool = False,
         cem_repeats: int = 1):
    cfg = load_config()
    ensure_dirs()
    wm, wm_meta = _load_wm(cfg, override_wm_gate, allow_legacy)

    reactive_file = "liquid_policy.pt"
    hier_file = "hier_policy.pt"
    results, all_records = {}, {}

    # 1) real planner first: collects subgoal norms/sequences for controls
    agent = _make_agent(cfg, "hierarchical", hier_file, wm, allow_legacy)
    agent.subgoal_source = "planner"
    agent.log_planner = True
    print(f"[eval-hier] hier_planner ({episodes} episodes on {layout})...")
    recs, planner_logs, subgoal_seqs = _run_variant(
        cfg, agent, episodes, layout, cem_repeats)
    all_records["hier_planner"] = recs
    results["planner_validation_on_policy"] = _planner_validation(planner_logs)
    results["planner_logs_summary"] = {
        "invocations": int(sum(len(l) for l in planner_logs)),
        "mean_best_score": float(np.mean(
            [e["best_score"] for l in planner_logs for e in l]))
        if planner_logs else None}

    norms = [float(np.linalg.norm(g)) for l in subgoal_seqs for g in l]
    mean_norm = float(np.mean(norms)) if norms else 1.0

    # 2) controls on the SAME seeds
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

    # shuffled: episode i replays the subgoal sequence recorded in episode
    # (i+1) mod N of the real planner runs
    class _ShuffleWrap:
        def __init__(self, agent):
            self.agent = agent
            self.ep = -1

        def reset(self):
            self.ep += 1
            if subgoal_seqs:
                seq = subgoal_seqs[(self.ep + 1) % len(subgoal_seqs)]
                self.agent.scripted_subgoals = seq
            self.agent.reset()

    a = _make_agent(cfg, "hierarchical", hier_file, wm, allow_legacy)
    a.subgoal_source = "shuffled"
    wrap = _ShuffleWrap(a)
    print("[eval-hier] hier_shuffled...")
    env = NavEnv(cfg, layout=layout)
    recs = []
    for ep in range(episodes):
        planner_seed = PLANNER_SEED_BASE + ep * 97
        rec = run_episode(env, a.act, wrap.reset, ENV_SEED_BASE + ep,
                          planner_seed=planner_seed)
        recs.append(rec)
    env.close()
    all_records["hier_shuffled"] = recs

    control("hier_heuristic", "heuristic")

    # 3) reactive checkpoint on the same seeds
    a = _make_agent(cfg, "reactive", reactive_file, None, allow_legacy)
    print("[eval-hier] reactive...")
    recs, _, _ = _run_variant(cfg, a, episodes, layout, cem_repeats)
    all_records["reactive"] = recs

    # ------------------------------------------------------------- report
    results["variants"] = {}
    for name, recs in all_records.items():
        s = summarize(recs)
        results["variants"][name] = {"summary": s, "episodes": recs}
        print(f"[eval-hier] {name:15s}: "
              f"R={s['reward_mean']['point']:7.2f} "
              f"[{s['reward_mean']['lo']:.2f},{s['reward_mean']['hi']:.2f}]  "
              f"success={s['success']['k']}/{s['success']['n']} "
              f"[{s['success']['lo']:.2f},{s['success']['hi']:.2f}]  "
              f"final_dist={s['final_goal_dist_mean']['point']:.2f} m")

    results["comparisons"] = {}
    for other in ("reactive", "hier_zero", "hier_random", "hier_shuffled",
                  "hier_heuristic"):
        c = compare(all_records["hier_planner"], all_records[other],
                    "hier_planner", other)
        results["comparisons"][f"hier_planner_vs_{other}"] = c
        d = c["reward_diff"]
        print(f"[eval-hier] planner vs {other:15s}: "
              f"dR={d['mean_diff']:+7.2f} [{d['lo']:.2f},{d['hi']:.2f}] "
              f"{'(excludes 0)' if d['excludes_zero'] else '(includes 0)'}  "
              f"McNemar p={c['success_mcnemar']['p_value']:.3f} "
              f"(discordant {c['success_mcnemar']['discordant']})")

    # hierarchy gate: planner must beat ALL controls
    gates = {}
    for other in ("hier_zero", "hier_random", "hier_shuffled",
                  "hier_heuristic"):
        c = results["comparisons"][f"hier_planner_vs_{other}"]
        gates[other] = bool(c["reward_diff"]["mean_diff"] > 0
                            and c["reward_diff"]["excludes_zero"])
    results["hierarchy_gate"] = {
        "criteria": gates, "passed": all(gates.values()),
        "note": ("PILOT RESULT unless passed: the equal-extra-"
                 "reactive-training control still requires a separate "
                 "training run (see scripts/run_timing_factorial.py).")}
    print(f"[eval-hier] hierarchy gate "
          f"{'PASSED' if results['hierarchy_gate']['passed'] else 'NOT passed'}"
          f" — treat these numbers as a pilot until every control is beaten"
          f" (including the pending equal-extra-training control).")

    results["meta"] = gather_provenance(
        cfg, experiment_name=f"eval_hierarchy_{layout}",
        seeds={"env_seed_base": ENV_SEED_BASE,
               "planner_seed_base": PLANNER_SEED_BASE,
               "episodes": episodes, "cem_repeats": cem_repeats},
        extra={"checkpoints": {
            "reactive": checkpoint_ref(os.path.join(MODELS_DIR, reactive_file)),
            "hierarchical": checkpoint_ref(os.path.join(MODELS_DIR, hier_file)),
            "world_model": checkpoint_ref(
                os.path.join(MODELS_DIR, "world_model.pt"))},
            "wm_gate": wm_meta.get("gate", {}).get("passed")
            if isinstance(wm_meta.get("gate"), dict) else None})
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
    a = ap.parse_args()
    main(a.episodes, a.layout, a.override_wm_gate, a.allow_legacy,
         a.cem_repeats)
