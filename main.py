"""Liquid-Mind command line.

    python main.py check           # Phase 0: env smoke test (headless random agent)
    python main.py train-baseline  # Phase 1: PPO + MLP
    python main.py train-policy    # Phase 2: CMA-ES over SNN + liquid policy
    python main.py eval-dt         # Phase 2c: irregular-timing robustness
    python main.py train-wm        # Phase 3: JEPA world model
    python main.py train-hier      # Phase 4: goal-conditioned policy
    python main.py eval-hier       # Phase 4: U-trap comparison
    python main.py demo            # live GUI demo (see scripts/run_live.py flags)
"""
import argparse
import sys

from common import load_config


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
              f"final_goal_dist={info['goal_dist']:.2f} dt_last={info['dt']:.4f}")
    env.close()
    print("[check] environment OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[
        "check", "train-baseline", "train-policy", "eval-dt",
        "train-wm", "train-hier", "eval-hier", "demo"])
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args, extra = ap.parse_known_args()

    if args.command == "check":
        check()
    elif args.command == "train-baseline":
        from training.train_baseline import train
        train(args.timesteps)
    elif args.command == "train-policy":
        from training.train_policy import train
        train(args.generations, hierarchical=False, workers=args.workers)
    elif args.command == "train-hier":
        from training.train_policy import train
        train(args.generations, hierarchical=True, workers=args.workers)
    elif args.command == "train-wm":
        from training.train_world_model import train
        train()
    elif args.command == "eval-dt":
        from scripts.eval_dt_robustness import main as run
        run()
    elif args.command == "eval-hier":
        from scripts.eval_hierarchy import main as run
        run()
    elif args.command == "demo":
        sys.argv = [sys.argv[0]] + extra
        from scripts.run_live import main as run
        run()


if __name__ == "__main__":
    main()
