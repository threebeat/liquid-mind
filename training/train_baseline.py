"""Phase 1: PPO + plain MLP baseline (stable-baselines3).

Purpose: prove the environment is learnable and set a score for the liquid
agent to beat. Nothing exotic here on purpose.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from common import MODELS_DIR, ensure_dirs, load_config
from environment.nav_env import NavEnv


def train(timesteps: int | None = None, config: dict | None = None):
    cfg = config or load_config()
    ensure_dirs()
    timesteps = timesteps or int(cfg["training"]["baseline_timesteps"])
    n_envs = int(cfg["training"]["baseline_n_envs"])

    vec = make_vec_env(lambda: NavEnv(cfg), n_envs=n_envs)
    model = PPO("MlpPolicy", vec, verbose=1, seed=0,
                n_steps=1024, batch_size=256, learning_rate=3e-4,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=timesteps, progress_bar=False)
    path = os.path.join(MODELS_DIR, "ppo_baseline.zip")
    model.save(path)
    vec.close()

    eval_env = Monitor(NavEnv(cfg))
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
    eval_env.close()
    print(f"[baseline] saved {path}")
    print(f"[baseline] eval over 20 episodes: {mean_r:.2f} +/- {std_r:.2f}")
    return mean_r


if __name__ == "__main__":
    train()
