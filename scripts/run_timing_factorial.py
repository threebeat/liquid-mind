"""Timing factorial runner (Priority 3).

Enumerates the confound-free timing ablation:

    SNN propagation   CfC propagation   Raw dt channel
    physical          physical          masked
    physical          nominal           masked
    nominal           physical          masked
    nominal           nominal           masked
    physical          physical          visible   (control)
    nominal           nominal           visible   (control)

All cells share: the same environment semantics, the same training seed
bank, common random numbers within each generation, a held-out validation
seed bank for checkpoint selection, a shared 54-d sensory bus + identical
adapter shapes (fixed downstream architecture; not total-parameter matching),
and equal optimizer budgets. Each cell can be trained with multiple
independent seeds (--seeds N) for architecture-level claims; every
(cell, seed) pair is a separate manifest RUN with an explicit run_id.

Manifest lifecycle: the full planned run list is written upfront with
status "planned"; each run transitions to training -> completed | failed
(or skipped_valid when an existing artifact verifies against the PLANNED
spec), and every transition is persisted atomically. On a failure the
error is recorded and later independent cells continue by default; the
process exits nonzero at the end if any run failed (--fail-fast stops at
the first failure).

--preflight validates the whole plan without training (unique run IDs,
instantiable cell configs and parameter counts, budgets, timing, expected
paths, smoke-manifest validity) and exits nonzero on any failure.

Nothing is trained without an explicit --run flag; the default is a
dry-run that prints the experiment table. Use --smoke for a minutes-scale
end-to-end shakeout before committing to full budgets.
"""
import argparse
import copy
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import MODELS_DIR, ROOT, ensure_dirs, load_config
from scripts.factorial_io import (artifact_fields, cell_entry,
                                  discover_latest_manifest, load_manifest,
                                  manifest_path, run_identity,
                                  set_run_status, verify_manifest_cell,
                                  write_manifest)


def factorial_cells():
    cells = []
    for snn_phys, cfc_phys in itertools.product((True, False), repeat=2):
        cells.append({
            "name": f"{'phys' if snn_phys else 'nom'}snn"
                    f"-{'phys' if cfc_phys else 'nom'}cfc-masked",
            "snn_time_aware": snn_phys, "cfc_time_aware": cfc_phys,
            "mask_direct_dt": True, "hierarchical": False})
    # dt-visible controls: does explicit timing-conditioned action selection
    # explain robustness, rather than continuous internal state?
    for phys in (True, False):
        tag = "phys" if phys else "nom"
        cells.append({
            "name": f"{tag}snn-{tag}cfc-visible",
            "snn_time_aware": phys, "cfc_time_aware": phys,
            "mask_direct_dt": False, "hierarchical": False})
    return cells


def extra_reactive_cell():
    """Equal-extra-training control (Priority 6): warm-start the reactive
    policy from its own checkpoint and give it the hierarchical policy's
    additional budget, with no planner and no shaping."""
    return {"name": "reactive-extra-budget",
            "snn_time_aware": True, "cfc_time_aware": True,
            "mask_direct_dt": True, "hierarchical": False,
            "warm_from": "liquid_policy.pt"}


def cell_config(cfg: dict, cell: dict, smoke: bool, train_jitter: bool) -> dict:
    c = copy.deepcopy(cfg)
    a = c["agent"]
    a["snn_time_aware"] = cell["snn_time_aware"]
    a["cfc_time_aware"] = cell["cfc_time_aware"]
    a["mask_direct_dt"] = cell["mask_direct_dt"]
    c["training"]["train_irregular_dt"] = train_jitter
    if smoke:
        c["training"]["cma_generations"] = 2
        c["training"]["cma_population"] = 4
        c["training"]["episodes_per_candidate"] = 1
        c["training"]["validation_episodes"] = 2
    return c


def build_plans(cells, seeds, smoke, cfg, train_jitter):
    """One plan per (cell, training seed): explicit identity, no parsing."""
    plans = []
    for cell in cells:
        for seed in range(seeds):
            exp = f"factorial_{cell['name']}_s{seed}"
            if smoke:
                exp += "_smoke"
            plans.append({
                "cell": cell,
                "seed": seed,
                "experiment": exp,
                "run_id": run_identity(cell["name"], seed),
                "checkpoint": os.path.join(MODELS_DIR, exp + ".pt"),
                "config": cell_config(cfg, cell, smoke, train_jitter),
            })
    return plans


EXPECTED_SMOKE_SEEDS = (0, 1)


def check_smoke_manifest_complete(failures: list[str]) -> None:
    """STRICT Gate C' completeness: a full run requires the latest smoke
    manifest to contain exactly the 6 expected cells x seeds {0, 1} = 12
    runs, all completed/skipped_valid, all passing integrity verification,
    all semantics v3. An incomplete-but-not-invalid manifest must fail."""
    from provenance import SEMANTICS_VERSION

    smoke_man = discover_latest_manifest(smoke=True)
    if smoke_man is None:
        failures.append("full run requires a complete Gate C' smoke "
                        "manifest; none found (run --smoke --run first)")
        return
    try:
        man = load_manifest(smoke_man)
    except Exception as e:
        failures.append(f"smoke manifest {smoke_man} unreadable: {e}")
        return
    expected = {(c["name"], s) for c in factorial_cells()
                for s in EXPECTED_SMOKE_SEEDS}
    entries = {(e.get("cell_id") or e.get("name"), e.get("training_seed")): e
               for e in man.get("cells", [])}
    missing = sorted(f"{c}__s{s}" for c, s in expected - set(entries))
    if missing:
        failures.append(f"smoke manifest missing expected runs: {missing}")
    n_verified = 0
    for key in sorted(expected & set(entries)):
        entry = entries[key]
        rid = entry.get("run_id", f"{key[0]}__s{key[1]}")
        status = entry.get("status", "completed")
        if status not in ("completed", "skipped_valid"):
            failures.append(f"smoke run {rid} has status {status!r} "
                            f"(need completed/skipped_valid)")
            continue
        try:
            v = verify_manifest_cell(entry)
        except (ValueError, FileNotFoundError) as e:
            failures.append(f"smoke run {rid} fails integrity "
                            f"verification: {e}")
            continue
        if v.get("semantics_version") != SEMANTICS_VERSION:
            failures.append(f"smoke run {rid} has semantics_version "
                            f"{v.get('semantics_version')} != "
                            f"{SEMANTICS_VERSION}")
            continue
        n_verified += 1
    print(f"[preflight] Gate C' smoke manifest {smoke_man}: "
          f"{n_verified}/{len(expected)} expected runs verified "
          f"({'COMPLETE' if n_verified == len(expected) else 'INCOMPLETE'})")


def preflight(plans, smoke: bool) -> int:
    """Validate the planned factorial without training. Returns exit code."""
    import torch
    from agent.hybrid_agent import HybridAgent

    failures = []
    print(f"[preflight] {len({p['cell']['name'] for p in plans})} cells x "
          f"{len({p['seed'] for p in plans})} seeds = {len(plans)} runs "
          f"({'smoke' if smoke else 'full'} budgets)")

    run_ids = [p["run_id"] for p in plans]
    if len(set(run_ids)) != len(run_ids):
        dupes = sorted({r for r in run_ids if run_ids.count(r) > 1})
        failures.append(f"duplicate run_ids: {dupes}")

    # one template agent per cell config: parameter counts + budgets
    by_cell = {}
    for p in plans:
        by_cell.setdefault(p["cell"]["name"], p)
    for name, p in by_cell.items():
        c = p["config"]
        tr = c["training"]
        try:
            torch.manual_seed(0)
            n_params = HybridAgent(c, mode="reactive").n_parameters()
        except Exception as e:
            failures.append(f"cell {name}: agent instantiation failed: {e}")
            continue
        print(f"[preflight] {name:34s} params={n_params:6d} "
              f"budget=(gen={tr['cma_generations']}, "
              f"pop={tr['cma_population']}, "
              f"ep/cand={tr['episodes_per_candidate']}, "
              f"val={tr.get('validation_episodes', 12)}) "
              f"train_jitter={tr.get('train_irregular_dt', False)} "
              f"substeps=[{c['env'].get('substeps_min')},"
              f"{c['env'].get('substeps_max')}]")

    n_existing = 0
    for p in plans:
        exists = os.path.exists(p["checkpoint"])
        n_existing += int(exists)
        print(f"[preflight] {p['run_id']:44s} -> "
              f"models/{os.path.basename(p['checkpoint'])} "
              f"{'EXISTS' if exists else 'to train'}")
    print(f"[preflight] {n_existing}/{len(plans)} checkpoints already exist")

    from scripts.eval_dt_robustness import LADDER
    print(f"[preflight] evaluation: {len(plans)} runs x {len(LADDER)} ladder "
          f"levels x ~50 episodes = ~{len(plans) * len(LADDER) * 50} episodes")

    if smoke:
        # Smoke preflight: informational check of any existing smoke manifest
        smoke_man = discover_latest_manifest(smoke=True)
        if smoke_man is None:
            print("[preflight] no smoke manifest found (run --smoke --run "
                  "first to validate plumbing)")
        else:
            try:
                man = load_manifest(smoke_man)
                bad = 0
                for entry in man.get("cells", []):
                    if entry.get("status", "completed") not in (
                            "completed", "skipped_valid"):
                        continue
                    try:
                        verify_manifest_cell(entry)
                    except (ValueError, FileNotFoundError) as e:
                        bad += 1
                        failures.append(
                            f"smoke manifest entry {entry.get('run_id')} "
                            f"fails verification: {e}")
                print(f"[preflight] smoke manifest {smoke_man}: "
                      f"{'OK' if bad == 0 else f'{bad} entries FAIL verification'}")
            except Exception as e:
                failures.append(f"smoke manifest {smoke_man} unreadable: {e}")
    else:
        # Full preflight: the Gate C' smoke manifest must be COMPLETE
        check_smoke_manifest_complete(failures)

    if failures:
        print("\n[preflight] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[preflight] all checks passed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="actually train (default: dry-run print the table)")
    ap.add_argument("--preflight", action="store_true",
                    help="validate the planned factorial without training")
    ap.add_argument("--smoke", action="store_true",
                    help="minutes-scale budgets for an end-to-end shakeout")
    ap.add_argument("--seeds", type=int, default=1,
                    help="independent training seeds per cell")
    ap.add_argument("--cells", type=str, default=None,
                    help="comma-separated cell-name filter")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--train-jitter", action="store_true",
                    help="train under irregular timing (default: fixed rate)")
    ap.add_argument("--include-extra-reactive", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failed run (default: continue "
                    "to independent cells and exit nonzero at the end)")
    ap.add_argument("--stamp", type=str, default=None,
                    help="manifest timestamp to reuse (resume a full run "
                    "into the SAME manifest, preserving attempt history)")
    ap.add_argument("--allow-underpowered-full", action="store_true",
                    help="diagnostic override of the --seeds >= 5 "
                    "requirement for full runs; recorded in the manifest")
    args = ap.parse_args()

    if (args.run and not args.smoke and args.seeds < 5
            and not args.allow_underpowered_full):
        ap.error("full Gate D requires --seeds >= 5 (preregistered "
                 "replicates); use --allow-underpowered-full only for "
                 "deliberate diagnostics — the override is recorded in "
                 "the manifest")

    cfg = load_config()
    ensure_dirs()
    cells = factorial_cells()
    if args.include_extra_reactive:
        cells.append(extra_reactive_cell())
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",")}
        cells = [c for c in cells if c["name"] in wanted]

    plans = build_plans(cells, args.seeds, args.smoke, cfg, args.train_jitter)

    if args.preflight:
        sys.exit(preflight(plans, args.smoke))

    print(f"{'run_id':44s} {'snn':8s} {'cfc':8s} {'raw dt':8s} checkpoint")
    for p in plans:
        cell = p["cell"]
        print(f"{p['run_id']:44s} "
              f"{'phys' if cell['snn_time_aware'] else 'nominal':8s} "
              f"{'phys' if cell['cfc_time_aware'] else 'nominal':8s} "
              f"{'masked' if cell['mask_direct_dt'] else 'VISIBLE':8s} "
              f"models/{os.path.basename(p['checkpoint'])}")

    if not args.run:
        print("\ndry-run only. Re-run with --run to train "
              "(--smoke first is strongly recommended; --preflight to "
              "validate the plan).")
        return

    from training.train_policy import train
    stamp = args.stamp or time.strftime("%Y%m%d_%H%M%S")
    man_path = manifest_path(smoke=args.smoke,
                             stamp=None if args.smoke else stamp)
    man_meta = {"stamp": None if args.smoke else stamp}
    if args.allow_underpowered_full and not args.smoke:
        man_meta["underpowered_override"] = True
        man_meta["seeds_requested"] = args.seeds

    # Merge prior manifest so retraining with --force increments `attempt`
    # and preserves the historical record instead of silently replacing it.
    prior = {}
    if os.path.exists(man_path):
        try:
            prior = {e.get("run_id"): e
                     for e in load_manifest(man_path).get("cells", [])}
        except Exception:
            prior = {}

    entries = []
    for p in plans:
        prev = prior.get(p["run_id"])
        attempt = 1
        history = []
        if prev is not None:
            history = list(prev.get("attempt_history", []))
            attempt = int(prev.get("attempt", 1))
            will_retrain = args.force or not os.path.exists(p["checkpoint"])
            if will_retrain and prev.get("status") not in (None, "planned"):
                attempt += 1
                history.append({k: prev.get(k) for k in
                                ("attempt", "status", "status_history",
                                 "experiment", "file_sha256",
                                 "state_checksum")})
        entry = cell_entry(p["cell"], p["checkpoint"], p["seed"],
                           p["experiment"], args.smoke, p["config"],
                           attempt=attempt, status="planned")
        entry["attempt_history"] = history
        set_run_status(entry, "planned")
        entries.append(entry)
    write_manifest(man_path, entries, smoke=args.smoke, meta=man_meta)
    print(f"[factorial] planned manifest written: {man_path}")

    any_failed = False
    for entry, p in zip(entries, plans):
        ckpt = p["checkpoint"]
        run_id = p["run_id"]

        if os.path.exists(ckpt) and not args.force:
            # skipped_valid must mean "matches the experiment we intended
            # to run" — verify against the independently planned spec.
            entry.update(artifact_fields(ckpt))
            try:
                verify_manifest_cell(entry)
                set_run_status(entry, "skipped_valid",
                               checkpoint=entry["checkpoint"])
                print(f"[factorial] SKIP {run_id}: existing checkpoint "
                      f"verifies against the planned spec")
            except (ValueError, FileNotFoundError) as e:
                set_run_status(entry, "failed",
                               error=f"existing checkpoint does not match "
                                     f"planned spec: {e}")
                any_failed = True
                print(f"[factorial] FAIL {run_id}: existing checkpoint does "
                      f"not match the planned spec (--force to retrain)\n"
                      f"  {e}")
            write_manifest(man_path, entries, smoke=args.smoke, meta=man_meta)
            if any_failed and args.fail_fast:
                break
            continue

        set_run_status(entry, "training")
        write_manifest(man_path, entries, smoke=args.smoke, meta=man_meta)
        print(f"\n[factorial] === training {run_id} "
              f"(experiment {p['experiment']}, attempt {entry['attempt']}) ===")
        try:
            train(hierarchical=p["cell"].get("hierarchical", False),
                  workers=args.workers, config=p["config"],
                  experiment_name=p["experiment"], seed=p["seed"],
                  force=args.force, warm_from=p["cell"].get("warm_from"),
                  run_kind="smoke" if args.smoke else "full")
            entry.update(artifact_fields(ckpt))
            set_run_status(entry, "completed", checkpoint=entry["checkpoint"])
        except Exception as e:                          # noqa: BLE001
            set_run_status(entry, "failed",
                           error=f"{type(e).__name__}: {e}")
            any_failed = True
            print(f"[factorial] FAILED {run_id}: {type(e).__name__}: {e}")
        write_manifest(man_path, entries, smoke=args.smoke, meta=man_meta)
        if any_failed and args.fail_fast:
            print("[factorial] --fail-fast: stopping at first failure")
            break

    write_manifest(man_path, entries, smoke=args.smoke, meta=man_meta)
    by_status = {}
    for e in entries:
        by_status.setdefault(e["status"], []).append(e["run_id"])
    print(f"\n[factorial] done. Manifest: {man_path}")
    for st, ids in sorted(by_status.items()):
        print(f"  {st:14s} {len(ids):2d}: {', '.join(ids)}")
    print("Evaluate with:\n"
          f"  python scripts/eval_dt_robustness.py --manifest {man_path}\n"
          "  python scripts/eval_dt_robustness.py --all-factorial")
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
