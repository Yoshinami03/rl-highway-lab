#!/usr/bin/env python3
"""
Highway2 学習スクリプト

使い方:
    python train.py                    # 新規学習（デフォルト10万ステップ）
    python train.py --timesteps 500000 # 50万ステップ学習
    python train.py --resume model.zip # 既存モデルから追加学習
    python train.py --eval-only        # 評価のみ
    python train.py --render-only      # 描画のみ
"""

import argparse
from pathlib import Path

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from env import CoopMergeEnv, CoopMergeConfig


def make_vec_env(num_agents: int = 12, num_envs: int = 1, seed: int = 0):
    """PettingZoo環境をSB3用VecEnvに変換"""
    base = CoopMergeEnv(num_agents=num_agents, config=CoopMergeConfig(), seed=seed)
    venv = ss.pettingzoo_env_to_vec_env_v1(base)
    venv = ss.concat_vec_envs_v1(
        venv,
        num_vec_envs=num_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    venv = VecMonitor(venv)
    return venv


def train(
    timesteps: int = 100_000,
    resume_path: str | None = None,
    save_path: str = "./ppo_trained.zip",
    checkpoint_freq: int = 50_000,
    seed: int = 0,
):
    """学習を実行"""
    print("=" * 50)
    print(" Highway2 Training")
    print("=" * 50)
    print(f"  Timesteps    : {timesteps:,}")
    print(f"  Resume from  : {resume_path or 'None (new model)'}")
    print(f"  Save to      : {save_path}")
    print("=" * 50)

    venv = make_vec_env(seed=seed)

    if resume_path and Path(resume_path).exists():
        print(f"\nLoading model from {resume_path}...")
        model = PPO.load(resume_path, env=venv, device="cpu")
        model.ent_coef = 0.001
        model.learning_rate = 1e-4
    else:
        print("\nCreating new model...")
        model = PPO("MlpPolicy", venv, verbose=1, device="cpu")

    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints",
        name_prefix="ppo_ckpt",
    )

    print(f"\nStarting training for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=(resume_path is None),
        callback=ckpt_cb,
        progress_bar=False,
    )

    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return save_path


def evaluate(model_path: str, num_seeds: int = 5, num_envs: int = 4):
    """モデルを評価"""
    from evaluate import evaluate_model
    return evaluate_model(model_path, num_seeds=num_seeds, num_envs=num_envs)


def visualize(model_path: str, output_path: str = "./demo.mp4"):
    """動画を生成"""
    from visualize import render_video
    return render_video(model_path, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(description="Highway2 Training Script")
    parser.add_argument("--timesteps", type=int, default=100_000, help="学習ステップ数")
    parser.add_argument("--resume", type=str, default=None, help="追加学習するモデルのパス")
    parser.add_argument("--save", type=str, default="./ppo_trained.zip", help="保存先パス")
    parser.add_argument("--seed", type=int, default=0, help="シード値")
    parser.add_argument("--eval-only", action="store_true", help="評価のみ実行")
    parser.add_argument("--render-only", action="store_true", help="描画のみ実行")
    parser.add_argument("--no-eval", action="store_true", help="学習後の評価をスキップ")
    parser.add_argument("--no-render", action="store_true", help="学習後の描画をスキップ")

    args = parser.parse_args()

    model_path = args.save

    if args.eval_only:
        if not Path(model_path).exists():
            print(f"Error: Model not found at {model_path}")
            return
        evaluate(model_path)
        return

    if args.render_only:
        if not Path(model_path).exists():
            print(f"Error: Model not found at {model_path}")
            return
        visualize(model_path)
        return

    # 学習
    model_path = train(
        timesteps=args.timesteps,
        resume_path=args.resume,
        save_path=args.save,
        seed=args.seed,
    )

    # 評価
    if not args.no_eval:
        print("\n" + "=" * 50)
        print(" Evaluation")
        print("=" * 50)
        try:
            evaluate(model_path)
        except Exception as e:
            print(f"Evaluation failed: {e}")

    # 描画
    if not args.no_render:
        print("\n" + "=" * 50)
        print(" Rendering Video")
        print("=" * 50)
        try:
            visualize(model_path)
        except Exception as e:
            print(f"Rendering failed: {e}")

    print("\n" + "=" * 50)
    print(" Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
