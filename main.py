"""Liquid-Mind command line.

    python main.py check           # Phase 0: env smoke test (headless random agent)
    python main.py import-legacy   # wrap pre-provenance artifacts as legacy
    python main.py train-baseline  # Phase 1: PPO + MLP
    python main.py train-policy    # Phase 2: CMA-ES over SNN + liquid policy
    python main.py eval-dt         # Phase 2c: timing-disturbance ladder
    python main.py train-wm        # Phase 3: JEPA world model (+ planning gate)
    python main.py train-hier      # Phase 4: goal-conditioned policy
    python main.py eval-hier       # Phase 4: attribution-controlled comparison
    python main.py demo            # live GUI demo (see scripts/run_live.py flags)

Run `python -m pytest tests` before any training run: deterministic tests
must pass first (see README).
"""
import argparse
import os
import sys

from common import MODELS_DIR, load_config


def check():
    import numpy as np
    from environment.nav_env import NavEnv
    cfg = load_config()
    env = NavEnv(cfg, irregular_dt=True)
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape
        total, done, steps = 0.0, False, 0
        while not done:
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
            total += r
            steps += 1
            done = term or trunc
        print(f"[check] episode {ep}: steps={steps} reward={total:.2f} "
              f"final_goal_dist={info['goal_dist']:.2f} dt_last={info['dt']:.4f} "
              f"end_time={info['sim_time']:.4f}s")
    env.close()
    print("[check] environment OK")


def import_legacy(force: bool = False):
    """Wrap the pre-provenance artifacts in models/ as explicit legacy
    checkpoints under models/legacy/. Originals are left untouched."""
    from provenance import import_legacy_checkpoint
    legacy_dir = os.path.join(MODELS_DIR, "legacy")
    candidates = ["liquid_policy.pt", "hier_policy.pt", "world_model.pt"]
    done = 0
    for name in candidates:
        src = os.path.join(MODELS_DIR, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(legacy_dir,
                           name.replace(".pt", "_legacy.pt"))
        try:
            sha = import_legacy_checkpoint(src, dst, force=force,
                                           note="pre-provenance artifact "
                                                "(semantics_version 1)")
            print(f"[legacy] {src} -> {dst} (sha256 {sha[:12]}...)")
            done += 1
        except (ValueError, FileExistsError) as e:
            print(f"[legacy] skip {name}: {e}")
    if done == 0:
        print("[legacy] nothing imported")
    else:
        print(f"[legacy] {done} artifact(s) imported. Originals preserved; "
              f"the imported copies load only with allow_legacy=True and "
              f"can never be mistaken for new experiments.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[
        "check", "import-legacy", "train-baseline", "train-policy",
        "eval-dt", "train-wm", "train-hier", "eval-hier", "demo"])
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--layout", type=str, default="u_trap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment-name", type=str, default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing artifact")
    ap.add_argument("--override-wm-gate", action="store_true",
                    help="use a failed/ungated world model (diagnostics only)")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="permit explicitly imported legacy checkpoints")
    args, extra = ap.parse_known_args()

    if args.command == "check":
        check()
    elif args.command == "import-legacy":
        import_legacy(force=args.force)
    elif args.command == "train-baseline":
        from training.train_baseline import train
        train(args.timesteps, force=args.force)
    elif args.command == "train-policy":
        from training.train_policy import train
        train(args.generations, hierarchical=False, workers=args.workers,
              experiment_name=args.experiment_name, seed=args.seed,
              force=args.force)
    elif args.command == "train-hier":
        from training.train_policy import train
        train(args.generations, hierarchical=True, workers=args.workers,
              experiment_name=args.experiment_name, seed=args.seed,
              force=args.force, override_wm_gate=args.override_wm_gate)
    elif args.command == "train-wm":
        from training.train_world_model import train
        train(force=args.force)
    elif args.command == "eval-dt":
        from scripts.eval_dt_robustness import main as run
        run(args.episodes or 50, allow_legacy=args.allow_legacy)
    elif args.command == "eval-hier":
        from scripts.eval_hierarchy import main as run
        run(args.episodes or 100, layout=args.layout,
            override_wm_gate=args.override_wm_gate,
            allow_legacy=args.allow_legacy)
    elif args.command == "demo":
        sys.argv = [sys.argv[0]] + extra
        from scripts.run_live import main as run
        run()


if __name__ == "__main__":
    main()
