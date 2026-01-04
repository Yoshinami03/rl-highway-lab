#!/usr/bin/env python3
"""
Highway2 学習スクリプト

使い方:
    python train.py                    # 既存モデルがあれば継続、なければ新規（デフォルト10万ステップ）
    python train.py --timesteps 500000 # 50万ステップ学習
    python train.py --new              # 強制的に新規モデルを作成
    python train.py --eval-only        # 評価のみ
    python train.py --render-only      # 描画のみ
"""

import argparse
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import CoopMergeConfig
from env_utils import make_vec_env


def train(
    timesteps: int = 100_000,
    save_path: str = "./ppo_trained.zip",
    checkpoint_freq: int = 50_000,
    seed: int = 0,
    force_new: bool = False,
    config: Optional[CoopMergeConfig] = None,
):
    """学習を実行"""
    cfg = config or CoopMergeConfig()

    print("=" * 50)
    print(" Highway2 Training")
    print("=" * 50)
    print(f"  Timesteps    : {timesteps:,}")
    print(f"  Save to      : {save_path}")

    # 既存モデルの自動検出（force_newがFalseの場合）
    resume_path = None
    if not force_new and Path(save_path).exists():
        resume_path = save_path
        print(f"  Resume from  : {resume_path} (auto-detected)")
    elif force_new:
        print(f"  Resume from  : None (force new model)")
    else:
        print(f"  Resume from  : None (no existing model)")

    print("=" * 50)

    venv = make_vec_env(seed=seed, config=cfg)

    if resume_path and Path(resume_path).exists():
        print(f"\nLoading model from {resume_path}...")
        model = PPO.load(resume_path, env=venv, device="cpu")
        # Update hyperparameters from config
        model.learning_rate = cfg.ppo_learning_rate
        model.ent_coef = cfg.ppo_entropy_coef
        model.vf_coef = cfg.ppo_vf_coef
    else:
        print("\nCreating new model...")
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            device="cpu",
            learning_rate=cfg.ppo_learning_rate,
            ent_coef=cfg.ppo_entropy_coef,
            vf_coef=cfg.ppo_vf_coef,
            clip_range=cfg.ppo_clip_range,
            gamma=cfg.ppo_gamma,
            gae_lambda=cfg.ppo_gae_lambda,
            batch_size=cfg.ppo_batch_size,
            n_epochs=cfg.ppo_n_epochs,
        )

    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints",
        name_prefix="ppo_ckpt",
    )

    print(f"\nStarting training for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=(resume_path is None or force_new),
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
    parser.add_argument("--new", action="store_true", help="既存モデルを無視して新規作成")
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
        save_path=args.save,
        seed=args.seed,
        force_new=args.new,
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
