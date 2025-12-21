"""
ランダム行動で環境を可視化するスクリプト
学習済みモデルなしで、エピソードの長さを確認できます
"""

import os
import sys
import time

# src 配下のローカル highway_env を優先して import する
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_highway import HighwayMultiEnv
from env_config import env_config


def visualize_random_episode(max_steps: int = None):
    """
    ランダム行動でエピソードを実行し、可視化する
    
    Args:
        max_steps: 最大ステップ数（Noneの場合は環境設定のデフォルト値を使用）
    """
    # 環境を作成（render_mode='human'で可視化）
    env = HighwayMultiEnv(config=env_config, render_mode='human')
    
    print(f"Environment created with {len(env.agents)} agents")
    print(f"Max episode steps: {env_config.max_episode_steps if max_steps is None else max_steps}")
    print(f"Action space size: {env_config.action_space_size}")
    print("\nStarting random episode visualization...")
    print("Press Ctrl+C to stop\n")
    
    # エピソードをリセット
    observations, infos = env.reset()
    
    step_count = 0
    total_reward = 0.0
    episode_active = True
    
    try:
        while episode_active:
            # ランダムな行動を選択
            actions = {
                agent: env.action_space(agent).sample() 
                for agent in env.agents
            }
            
            # ステップを実行
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 報酬を集計
            step_reward = sum(rewards.values())
            total_reward += step_reward
            
            step_count += 1
            
            # 10ステップごとに進捗を表示
            if step_count % 10 == 0:
                print(f"Step {step_count}: Reward={step_reward:.2f}, Total={total_reward:.2f}, Active agents={len(env.agents)}")
            
            # 終了判定
            if any(terminations.values()) or any(truncations.values()):
                episode_active = False
                
                # 終了理由を表示
                if any(terminations.values()):
                    print(f"\n✓ Episode terminated at step {step_count}")
                    # 衝突チェック
                    if hasattr(env.core_env, 'vehicle') and env.core_env.vehicle.crashed:
                        print("  Reason: Collision")
                    else:
                        print("  Reason: Max steps reached")
                else:
                    print(f"\n✓ Episode truncated at step {step_count}")
            
            # 描画のための短い待機
            time.sleep(0.01)
            
            # 最大ステップ数の上書きチェック
            if max_steps and step_count >= max_steps:
                print(f"\n✓ Reached specified max steps: {max_steps}")
                episode_active = False
                
    except KeyboardInterrupt:
        print(f"\n\nInterrupted at step {step_count}")
    
    finally:
        print(f"\nFinal Statistics:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward per step: {total_reward/step_count:.2f}")
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize random episode")
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=None,
        help="Maximum steps (default: use env config)"
    )
    
    args = parser.parse_args()
    visualize_random_episode(max_steps=args.max_steps)

