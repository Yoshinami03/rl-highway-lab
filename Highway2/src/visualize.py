#!/usr/bin/env python3
"""
Highway2 描画スクリプト

使い方:
    python visualize.py                        # デフォルトモデルで動画生成
    python visualize.py --model model.zip      # 指定モデルで動画生成
    python visualize.py --output demo.mp4      # 出力ファイル名指定
    python visualize.py --seconds 20           # 20秒の動画を生成
"""

import argparse
from pathlib import Path

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

try:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_RENDER_DEPS = True
except ImportError:
    HAS_RENDER_DEPS = False

from env import CoopMergeEnv, CoopMergeConfig


def render_frame(env, cfg, xlim, width=960, height=540, dpi=100, y_scale=4.0):
    """1フレームをレンダリング"""
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(80, -20)
    ax.axis("off")

    # 本線レーン (0, 1, 2)
    xs = np.linspace(xlim[0], xlim[1], 200)
    for lane in range(3):
        y = lane * cfg.lane_width * y_scale
        ax.plot(xs, np.full_like(xs, y), lw=10, color="gray", zorder=1)

    # 合流レーン (3)
    ramp_x1 = min(xlim[1], cfg.merge_end)
    xs_ramp = np.linspace(xlim[0], ramp_x1, 200)
    ys_ramp = [env._lane_center_y(3, x) * y_scale for x in xs_ramp]
    ax.plot(xs_ramp, ys_ramp, lw=12, color="gray", zorder=1)

    # 区間線
    ax.axvline(cfg.merge_start, ls="--", color="black", alpha=0.5, zorder=2)
    ax.axvline(cfg.merge_end, ls="--", color="black", alpha=0.5, zorder=2)

    # 車両
    veh_w = cfg.lane_width * 4
    veh_h = cfg.lane_width * 0.3 * y_scale
    for i in range(env._num_agents):
        if not env.active[i]:
            continue
        x = float(env.x[i])
        y = float(env.y[i]) * y_scale

        # 警告状態の判定
        is_warning = False
        if int(env.lane[i]) == 3:
            dist_to_end = cfg.merge_end - float(env.x[i])
            if 0 < dist_to_end < 40:
                is_warning = True

        color = "orange" if is_warning else "white"
        rect = plt.Rectangle(
            (x - veh_w / 2, y - veh_h / 2),
            veh_w,
            veh_h,
            fc=color,
            ec="black",
            lw=2,
            alpha=0.9,
            zorder=10,
        )
        ax.add_patch(rect)

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def render_video(
    model_path: str,
    output_path: str = "./demo.mp4",
    fps: int = 15,
    seconds: int = 12,
    seed: int = 0,
) -> str:
    """動画を生成"""
    if not HAS_RENDER_DEPS:
        raise ImportError("imageio and matplotlib are required for rendering. "
                         "Install with: pip install imageio[ffmpeg] matplotlib")

    print(f"Rendering video: {output_path}")
    print(f"  Model: {model_path}")
    print(f"  Duration: {seconds}s at {fps}fps")

    cfg = CoopMergeConfig()
    env = CoopMergeEnv(num_agents=12, config=cfg, seed=seed)
    agents = env.possible_agents

    # モデル読み込み用VecEnv
    base = CoopMergeEnv(num_agents=12, config=cfg, seed=seed)
    venv = ss.pettingzoo_env_to_vec_env_v1(base)
    venv = ss.concat_vec_envs_v1(
        venv,
        num_vec_envs=1,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    venv = VecMonitor(venv)

    model = PPO.load(model_path, env=venv, device="cpu")

    def obs_dict_to_batch(obs_dict):
        return np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in agents], axis=0)

    def act_batch_to_dict(act_batch):
        return {a: np.asarray(act_batch)[i].astype(np.int64) for i, a in enumerate(agents)}

    # カメラ範囲
    xlim = (-25, cfg.merge_end + 200)

    # ロールアウト
    obs_dict, _ = env.reset(seed=seed)
    frames = []
    n_frames = fps * seconds

    print(f"  Generating {n_frames} frames...")
    for frame_idx in range(n_frames):
        obs_batch = obs_dict_to_batch(obs_dict)
        act_batch, _ = model.predict(obs_batch, deterministic=True)
        actions = act_batch_to_dict(act_batch)
        obs_dict, _, _, trunc_dict, _ = env.step(actions)

        frame = render_frame(env, cfg, xlim)
        frames.append(frame)

        if any(trunc_dict.values()):
            obs_dict, _ = env.reset(seed=seed)

        if (frame_idx + 1) % 30 == 0:
            print(f"    {frame_idx + 1}/{n_frames} frames")

    # 動画書き出し
    print(f"  Writing video...")
    try:
        imageio.mimwrite(
            output_path,
            frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
    except Exception as e:
        print(f"  Warning: libx264 failed ({e}), using mpeg4...")
        fallback_path = output_path.replace(".mp4", "_fallback.mp4")
        imageio.mimwrite(fallback_path, frames, fps=fps, codec="mpeg4", macro_block_size=1)
        output_path = fallback_path

    print(f"  Video saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Highway2 Visualization Script")
    parser.add_argument("--model", type=str, default="./ppo_trained.zip", help="モデルのパス")
    parser.add_argument("--output", type=str, default="./demo.mp4", help="出力ファイル名")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    parser.add_argument("--seconds", type=int, default=12, help="動画の長さ（秒）")
    parser.add_argument("--seed", type=int, default=0, help="シード値")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return

    render_video(
        args.model,
        output_path=args.output,
        fps=args.fps,
        seconds=args.seconds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
