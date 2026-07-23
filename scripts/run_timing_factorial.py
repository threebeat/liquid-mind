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
independent seeds (--seeds N) for architecture-level claims.

Writes a factorial manifest (models/factorial_manifest_*.json) that
eval_dt_robustness.py consumes via --manifest / --all-factorial.

Also registers (off by default) the "reactive_extra_budget" attribution
control: the reactive policy given the same ADDITIONAL training budget the
hierarchical policy received — required before claiming planner value
(--include-extra-reactive to enable).

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

from common import MODELS_DIR, ensure_dirs, load_config
from scripts.factorial_io import cell_entry, manifest_path, write_manifest


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="actually train (default: dry-run print the table)")
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
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    cells = factorial_cells()
    if args.include_extra_reactive:
        cells.append(extra_reactive_cell())
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",")}
        cells = [c for c in cells if c["name"] in wanted]

    print(f"{'cell':34s} {'snn':8s} {'cfc':8s} {'raw dt':8s} "
          f"{'seeds':6s} checkpoint")
    plans = []
    for cell in cells:
        for seed in range(args.seeds):
            exp = f"factorial_{cell['name']}" + (f"_s{seed}"
                                                 if args.seeds > 1 else "")
            if args.smoke:
                exp += "_smoke"
            plans.append((cell, seed, exp))
            print(f"{cell['name']:34s} "
                  f"{'phys' if cell['snn_time_aware'] else 'nominal':8s} "
                  f"{'phys' if cell['cfc_time_aware'] else 'nominal':8s} "
                  f"{'masked' if cell['mask_direct_dt'] else 'VISIBLE':8s} "
                  f"{seed:<6d} models/{exp}.pt")

    if not args.run:
        print("\ndry-run only. Re-run with --run to train "
              "(--smoke first is strongly recommended).")
        return

    from training.train_policy import train
    stamp = time.strftime("%Y%m%d_%H%M%S")
    man_path = manifest_path(smoke=args.smoke,
                             stamp=None if args.smoke else stamp)
    entries = []
    for cell, seed, exp in plans:
        ckpt = os.path.join(MODELS_DIR, exp + ".pt")
        c = cell_config(cfg, cell, args.smoke, args.train_jitter)
        if os.path.exists(ckpt) and not args.force:
            print(f"[factorial] SKIP {exp}: checkpoint exists "
                  f"(--force to retrain); still recording in manifest")
        else:
            print(f"\n[factorial] === training {exp} (seed {seed}) ===")
            train(hierarchical=cell.get("hierarchical", False),
                  workers=args.workers, config=c, experiment_name=exp,
                  seed=seed, force=args.force,
                  warm_from=cell.get("warm_from"))
        if os.path.exists(ckpt):
            entries.append(cell_entry(cell, ckpt, seed, exp, args.smoke, c))
            write_manifest(man_path, entries, smoke=args.smoke,
                           meta={"updated": time.strftime("%Y-%m-%dT%H:%M:%S")})

    write_manifest(man_path, entries, smoke=args.smoke)
    print(f"\n[factorial] done. Manifest: {man_path}")
    print("Evaluate with:\n"
          f"  python scripts/eval_dt_robustness.py --manifest {man_path}\n"
          "  python scripts/eval_dt_robustness.py --all-factorial")


if __name__ == "__main__":
    main()
