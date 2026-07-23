"""Phase 1: PPO + plain MLP baseline (stable-baselines3).

Purpose: prove the environment is learnable and set a score for the liquid
agent to beat. Nothing exotic here on purpose.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from common import MODELS_DIR, ensure_dirs, load_config
from provenance import file_checksum, gather_provenance
from environment.nav_env import NavEnv


def train(timesteps: int | None = None, config: dict | None = None,
          force: bool = False):
    cfg = config or load_config()
    ensure_dirs()
    timesteps = timesteps or int(cfg["training"]["baseline_timesteps"])
    n_envs = int(cfg["training"]["baseline_n_envs"])
    path = os.path.join(MODELS_DIR, "ppo_baseline.zip")
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{path} already exists; refusing to overwrite. Re-run with "
            f"--force to replace it.")

    vec = make_vec_env(lambda: NavEnv(cfg), n_envs=n_envs)
    model = PPO("MlpPolicy", vec, verbose=1, seed=0,
                n_steps=1024, batch_size=256, learning_rate=3e-4,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(path)
    vec.close()

    eval_env = Monitor(NavEnv(cfg))
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
    eval_env.close()

    # SB3 zips carry no project metadata; write a provenance sidecar
    n_params = int(sum(p.numel() for p in model.policy.parameters()))
    meta = gather_provenance(
        cfg, experiment_name="ppo_baseline",
        extra={"optimizer": "PPO", "timesteps": timesteps,
               "parameter_count": n_params,
               "eval": {"mean_reward": float(mean_r),
                        "std_reward": float(std_r), "episodes": 20},
               "checkpoint_sha256": file_checksum(path)})
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[baseline] saved {path} (+ {path}.meta.json)")
    print(f"[baseline] eval over 20 episodes: {mean_r:.2f} +/- {std_r:.2f}")
    return mean_r


if __name__ == "__main__":
    train()
