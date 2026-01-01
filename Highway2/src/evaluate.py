#!/usr/bin/env python3
"""
Highway2 評価スクリプト

使い方:
    python evaluate.py                      # デフォルトモデルを評価
    python evaluate.py --model model.zip    # 指定モデルを評価
    python evaluate.py --seeds 20           # 20シードで評価
"""

import argparse
from pathlib import Path

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from env import CoopMergeEnv, CoopMergeConfig


NUM_AGENTS = 12  # 1インスタンス内のエージェント数


def make_vec_env(num_envs: int = 4, seed: int = 0):
    """評価用VecEnv作成"""
    base = CoopMergeEnv(num_agents=NUM_AGENTS, config=CoopMergeConfig(), seed=seed)
    venv = ss.pettingzoo_env_to_vec_env_v1(base)
    venv = ss.concat_vec_envs_v1(
        venv,
        num_vec_envs=num_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    venv = VecMonitor(venv)
    return venv


def step_vecenv_compat(venv, action):
    """VecEnvのstep互換性ラッパー"""
    out = venv.step(action)
    if len(out) == 4:
        obs, rew, done, infos = out
        return obs, rew, done, infos
    if len(out) == 5:
        obs, rew, term, trunc, infos = out
        done = np.logical_or(term, trunc)
        return obs, rew, done, infos
    raise RuntimeError("unexpected step format")


def evaluate_model(
    model_path: str,
    num_seeds: int = 5,
    num_envs: int = 4,
    max_steps: int = 500,
    device: str = "cpu",
) -> dict:
    """
    モデルを複数シードで評価

    Returns:
        dict: 評価結果（mean, std, min, max, all_returns）
    """
    print(f"Evaluating model: {model_path}")
    print(f"  Seeds: {num_seeds}, Envs per seed: {num_envs}")

    all_team_returns = []

    for seed_idx in range(num_seeds):
        seed = 1000 + seed_idx
        venv = make_vec_env(num_envs=num_envs, seed=seed)

        model = PPO.load(model_path, env=venv, device=device)

        obs = venv.reset()
        ep_team = np.zeros(num_envs, dtype=np.float64)

        for _ in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = step_vecenv_compat(venv, act)

            rew = np.asarray(rew, dtype=np.float64).reshape(num_envs, NUM_AGENTS)
            ep_team += rew[:, 0]

            done2 = np.asarray(done).reshape(num_envs, NUM_AGENTS)
            if np.all(done2[:, 0]):
                break

        all_team_returns.extend(ep_team.tolist())
        venv.close()

        print(f"  Seed {seed}: mean={np.mean(ep_team):.2f}")

    rets = np.array(all_team_returns, dtype=np.float64)

    results = {
        "mean": float(rets.mean()),
        "std": float(rets.std()),
        "min": float(rets.min()),
        "max": float(rets.max()),
        "n_episodes": len(rets),
        "all_returns": rets.tolist(),
    }

    print("\n=== Evaluation Results ===")
    print(f"  Episodes : {results['n_episodes']}")
    print(f"  Mean     : {results['mean']:.2f}")
    print(f"  Std      : {results['std']:.2f}")
    print(f"  Min/Max  : {results['min']:.2f} / {results['max']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Highway2 Evaluation Script")
    parser.add_argument("--model", type=str, default="./ppo_trained.zip", help="モデルのパス")
    parser.add_argument("--seeds", type=int, default=5, help="評価するシード数")
    parser.add_argument("--envs", type=int, default=4, help="シードあたりの環境数")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return

    evaluate_model(args.model, num_seeds=args.seeds, num_envs=args.envs)


if __name__ == "__main__":
    main()
