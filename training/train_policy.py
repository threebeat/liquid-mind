"""Phase 2 (and Phase 4): evolve the SNN + liquid policy with CMA-ES.

Why evolution instead of backprop: the policy is a stateful continuous-time
recurrent network driving a stateful spiking encoder. Backprop-through-time
across 600 physics steps on CPU is slow and fragile; separable CMA-ES treats
the whole agent as a black box and works well for networks of this size
(~10k parameters).

Selection (Priority 11 subset): candidates are screened with common random
numbers (the same environment seeds for every candidate within a
generation, drawn from a TRAINING seed bank). Each generation's winner is
re-evaluated on a disjoint VALIDATION seed bank, and the checkpoint saved is
the validation-best candidate — not the noisiest observed training best.
Training and validation fitness histories are stored in the checkpoint
metadata.

Phase 4 (--hierarchical): same evolution, but the agent runs its JEPA
planner during rollouts and the fitness adds a shaping term for moving
toward the planner's latent subgoal. The world model must have passed its
planning gate (see train_world_model.load_world_model) unless
--override-wm-gate is given. Warm-starting from the reactive policy records
the parent checkpoint checksum in the metadata.
"""
import argparse
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from cmaes import SepCMA

from common import MODELS_DIR, ensure_dirs, load_config
from provenance import checkpoint_ref, gather_provenance
from agent.hybrid_agent import HybridAgent
from environment.nav_env import NavEnv

# seed banks: training seeds and validation seeds must never overlap
TRAIN_SEED_MAX = 800_000
VALIDATION_SEED_BASE = 900_000


def seed_training_run(seed: int) -> dict:
    """Reset EVERY RNG a training run depends on, before any module is
    constructed. This makes the initial parameter vector a deterministic
    function of the training seed — matched seeds across factorial cells
    start from the same initialization regardless of prior RNG history
    (preflight instantiation, cell ordering, resumed runs).

    Returns the complete seeding policy for checkpoint metadata.
    """
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"training_seed": seed, "torch_seed": seed, "numpy_seed": seed,
            "python_seed": seed, "cma_seed": seed}

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


def _make_agent(cfg: dict, hierarchical: bool,
                override_wm_gate: bool = False) -> HybridAgent:
    wm = None
    if hierarchical:
        from training.train_world_model import load_world_model
        wm, _ = load_world_model(cfg, override_gate=override_wm_gate)
    return HybridAgent(cfg, mode="hierarchical" if hierarchical else "reactive",
                       world_model=wm)


def _worker_init(cfg: dict, hierarchical: bool, override_wm_gate: bool):
    global _ENV, _AGENT
    torch.set_num_threads(1)
    irregular = bool(cfg["training"].get("train_irregular_dt", False))
    _ENV = NavEnv(cfg, irregular_dt=irregular)
    _AGENT = _make_agent(cfg, hierarchical, override_wm_gate)


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
          workers: int | None = None, config: dict | None = None,
          experiment_name: str | None = None, seed: int = 0,
          force: bool = False, warm_start: bool = True,
          override_wm_gate: bool = False, warm_from: str | None = None,
          run_kind: str = "full"):
    assert run_kind in ("full", "smoke")
    # FIRST: deterministic initialization. Must precede _make_agent() (and
    # any torch module construction) so the initial parameter vector depends
    # only on the training seed. Worker processes need no seeding for
    # reproducibility: every candidate's parameters are overwritten by
    # _unflatten before evaluation.
    seed_policy = seed_training_run(seed)
    cfg = config or load_config()
    ensure_dirs()
    tr = cfg["training"]
    generations = generations or int(tr["cma_generations"])
    workers = workers if workers is not None else int(tr["workers"])
    popsize = int(tr["cma_population"])
    n_ep = int(tr["episodes_per_candidate"])
    n_val = int(tr.get("validation_episodes", 12))
    val_seeds = np.arange(VALIDATION_SEED_BASE, VALIDATION_SEED_BASE + n_val)

    template = _make_agent(cfg, hierarchical, override_wm_gate)
    stem = "hier_policy" if hierarchical else "liquid_policy"
    experiment_name = experiment_name or (stem + template.variant_tag())
    out_path = os.path.join(MODELS_DIR, experiment_name + ".pt")
    if os.path.exists(out_path) and not force:
        raise FileExistsError(
            f"{out_path} already exists; refusing to overwrite a previous "
            f"experiment. Re-run with --force or a new experiment name.")

    parent_ref = None
    if warm_from is not None:
        # explicit warm start (e.g. the equal-extra-budget reactive control)
        warm = warm_from if os.path.isabs(warm_from) \
            else os.path.join(MODELS_DIR, warm_from)
        template.load(warm)       # compat-validated: same flags required
        parent_ref = checkpoint_ref(warm)
        print(f"[cma] warm-starting from {warm}")
    elif hierarchical and warm_start:
        warm = os.path.join(MODELS_DIR,
                            "liquid_policy" + template.variant_tag() + ".pt")
        if not os.path.exists(warm):
            warm = os.path.join(MODELS_DIR, "liquid_policy.pt")
        if os.path.exists(warm):
            template.load(warm)   # compat-validated: same flags required
            parent_ref = checkpoint_ref(warm)
            print(f"[cma] warm-starting from reactive policy {warm}")
        else:
            print("[cma] no reactive policy found; training from scratch")
    wm_ref = (checkpoint_ref(os.path.join(MODELS_DIR, "world_model.pt"))
              if hierarchical else None)

    x0 = _flatten(template.parameters())
    dim = x0.size
    print(f"[cma] evolving {dim} parameters, pop={popsize}, "
          f"gens={generations}, hierarchical={hierarchical}, "
          f"experiment={experiment_name}")

    opt = SepCMA(mean=x0.astype(np.float64), sigma=float(tr["cma_sigma"]),
                 population_size=popsize, seed=seed)

    pool = None
    if workers > 1:
        pool = mp.Pool(workers, initializer=_worker_init,
                       initargs=(cfg, hierarchical, override_wm_gate))
    else:
        _worker_init(cfg, hierarchical, override_wm_gate)

    def _eval_jobs(jobs):
        return pool.map(_evaluate, jobs) if pool else [_evaluate(j) for j in jobs]

    rng = np.random.default_rng(seed)
    best_train_fit = np.inf
    best_val_fit, best_val_gen = np.inf, -1
    history = []
    saved = False
    t0 = time.time()
    for gen in range(generations):
        # common random numbers: identical seeds for every candidate, drawn
        # from the training bank (never overlapping the validation bank)
        seeds = rng.integers(0, TRAIN_SEED_MAX, size=n_ep)
        cands = [opt.ask() for _ in range(popsize)]
        jobs = [(c.astype(np.float32), seeds, hierarchical) for c in cands]
        fits = _eval_jobs(jobs)
        opt.tell(list(zip(cands, fits)))

        gen_best = float(np.min(fits))
        gen_winner = cands[int(np.argmin(fits))]
        best_train_fit = min(best_train_fit, gen_best)

        # re-evaluate the generation winner on the held-out validation bank;
        # the saved checkpoint is the VALIDATION best, not the training best
        val_fit = _eval_jobs([(gen_winner.astype(np.float32), val_seeds,
                               hierarchical)])[0]
        history.append({"generation": gen + 1, "train_best": -gen_best,
                        "validation": -float(val_fit)})
        if val_fit < best_val_fit:
            best_val_fit = float(val_fit)
            best_val_gen = gen + 1
            _unflatten(gen_winner, template.parameters())
            meta = gather_provenance(
                cfg, experiment_name=experiment_name,
                variant=template.variant_tag(),
                seeds={**seed_policy,
                       "validation_seeds": val_seeds.tolist()},
                extra={"mode": template.mode,
                       "optimizer": "SepCMA",
                       # run_kind lives in METADATA so downstream consumers
                       # (equal-extra validation, manifest verification)
                       # never have to infer smoke status from filenames
                       "run_kind": run_kind,
                       "smoke": run_kind == "smoke",
                       "parameter_count": template.n_parameters(),
                       "budget": {"generations": generations,
                                  "population": popsize,
                                  "episodes_per_candidate": n_ep,
                                  "validation_episodes": n_val},
                       "timing": {
                           "train_irregular_dt":
                               bool(tr.get("train_irregular_dt", False)),
                           "substeps_min": int(cfg["env"]["substeps_min"]),
                           "substeps_max": int(cfg["env"]["substeps_max"]),
                           "nominal_substeps":
                               int(cfg["env"]["control_substeps"])},
                       "selection": "validation_best",
                       "validation_reward": -best_val_fit,
                       "validation_best_generation": best_val_gen,
                       "train_best_reward": -best_train_fit,
                       "history": history,
                       "parent_checkpoint": parent_ref,
                       "world_model_checkpoint": wm_ref})
            # within-run updates overwrite our own file (guarded at start)
            template.save(out_path, meta=meta, force=True)
            saved = True
        print(f"[cma] gen {gen + 1}/{generations}  "
              f"train_best={-gen_best:.2f}  val={-val_fit:.2f}  "
              f"val_best={-best_val_fit:.2f}@g{best_val_gen}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    if pool:
        pool.close()
        pool.join()
    if saved:
        print(f"[cma] saved validation-best agent to {out_path} "
              f"(validation reward {-best_val_fit:.2f}, "
              f"generation {best_val_gen})")
    return -best_val_fit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--experiment-name", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--override-wm-gate", action="store_true")
    a = ap.parse_args()
    train(a.generations, a.hierarchical, a.workers,
          experiment_name=a.experiment_name, seed=a.seed, force=a.force,
          warm_start=not a.no_warm_start, override_wm_gate=a.override_wm_gate)
