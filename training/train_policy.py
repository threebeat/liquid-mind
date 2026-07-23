"""Phase 2 (and Phase 4): evolve the SNN + liquid policy with CMA-ES.

Why evolution instead of backprop: the policy is a stateful continuous-time
recurrent network driving a stateful spiking encoder. Backprop-through-time
across 600 physics steps on CPU is slow and fragile; separable CMA-ES treats
the whole agent as a black box and works well for networks of this size
(~10k parameters).

Phase 4 (--hierarchical): same evolution, but the agent runs its JEPA planner
during rollouts and the fitness adds a shaping term for moving toward the
planner's latent subgoal.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from cmaes import SepCMA

from common import MODELS_DIR, ensure_dirs, load_config
from agent.hybrid_agent import HybridAgent
from agent.world_model import WorldModel
from environment.nav_env import OBS_DIM, NavEnv

_ENV = None
_AGENT = None


def _flatten(params) -> np.ndarray:
    # ncps CfC params can be non-contiguous, so reshape (not view) + copy.
    return np.concatenate([p.detach().reshape(-1).cpu().numpy()
                           for p in params]).astype(np.float64)


def _unflatten(vec: np.ndarray, params) -> None:
    t = torch.from_numpy(np.asarray(vec, dtype=np.float32))
    i = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(t[i:i + n].reshape(p.shape))
            i += n


def _make_agent(cfg: dict, hierarchical: bool) -> HybridAgent:
    wm = None
    if hierarchical:
        wm = WorldModel(OBS_DIM, int(cfg["agent"]["latent_dim"]),
                        hidden=int(cfg["world_model"]["hidden_dim"]))
        wm.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "world_model.pt"), weights_only=True))
        wm.eval()
    return HybridAgent(cfg, mode="hierarchical" if hierarchical else "reactive",
                       world_model=wm)


def _worker_init(cfg: dict, hierarchical: bool):
    global _ENV, _AGENT
    torch.set_num_threads(1)
    _ENV = NavEnv(cfg)
    _AGENT = _make_agent(cfg, hierarchical)


def _evaluate(args):
    """Fitness of one parameter vector: negative mean episodic reward."""
    params, seeds, shaping = args
    _unflatten(params, _AGENT.parameters())
    total = 0.0
    for seed in seeds:
        obs, _ = _ENV.reset(seed=int(seed))
        _AGENT.reset()
        ep = 0.0
        done = False
        while not done:
            action = _AGENT.act(obs, _ENV._last_dt)
            # measure shaping before/after the step against the SAME subgoal
            # (act() may have just changed the target; comparing distances to
            # different targets would reward the target moving, not the robot)
            ld_before = (_AGENT.latent_dist_to_subgoal(obs) if shaping else 0.0)
            obs, r, term, trunc, _ = _ENV.step(action)
            if shaping:
                r += 0.1 * (ld_before - _AGENT.latent_dist_to_subgoal(obs))
            ep += r
            done = term or trunc
        total += ep
    return -total / len(seeds)


def train(generations: int | None = None, hierarchical: bool = False,
          workers: int | None = None, config: dict | None = None):
    cfg = config or load_config()
    ensure_dirs()
    tr = cfg["training"]
    generations = generations or int(tr["cma_generations"])
    workers = workers if workers is not None else int(tr["workers"])
    popsize = int(tr["cma_population"])
    n_ep = int(tr["episodes_per_candidate"])

    template = _make_agent(cfg, hierarchical)
    if hierarchical:
        warm = os.path.join(MODELS_DIR, "liquid_policy.pt")
        if os.path.exists(warm):
            template.load(warm)
            print("[cma] warm-starting from reactive liquid policy")
    x0 = _flatten(template.parameters())
    dim = x0.size
    print(f"[cma] evolving {dim} parameters, pop={popsize}, "
          f"gens={generations}, hierarchical={hierarchical}")

    opt = SepCMA(mean=x0.astype(np.float64), sigma=float(tr["cma_sigma"]),
                 population_size=popsize)
    stem = "hier_policy" if hierarchical else "liquid_policy"
    out_path = os.path.join(MODELS_DIR, stem + template.variant_tag() + ".pt")

    pool = None
    if workers > 1:
        pool = mp.Pool(workers, initializer=_worker_init,
                       initargs=(cfg, hierarchical))
    else:
        _worker_init(cfg, hierarchical)

    rng = np.random.default_rng(0)
    best_fit, best_x = np.inf, x0
    t0 = time.time()
    for gen in range(generations):
        seeds = rng.integers(0, 1_000_000, size=n_ep)   # same for all candidates
        cands = [opt.ask() for _ in range(popsize)]
        jobs = [(c.astype(np.float32), seeds, hierarchical) for c in cands]
        fits = pool.map(_evaluate, jobs) if pool else [_evaluate(j) for j in jobs]
        opt.tell(list(zip(cands, fits)))

        gen_best = float(np.min(fits))
        if gen_best < best_fit:
            best_fit = gen_best
            best_x = cands[int(np.argmin(fits))]
            _unflatten(best_x, template.parameters())
            template.save(out_path)
        print(f"[cma] gen {gen + 1}/{generations}  "
              f"best_reward={-gen_best:.2f}  all_time={-best_fit:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"[cma] saved best agent to {out_path} (reward {-best_fit:.2f})")
    return -best_fit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train(a.generations, a.hierarchical, a.workers)
