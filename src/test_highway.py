"""
モデル評価スクリプト
学習済みモデルのパフォーマンスを複数エピソードで評価し、統計を出力します。
"""

import os
import sys
import argparse
import json
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import numpy as np
from stable_baselines3 import PPO

# src 配下のローカル highway_env を優先して import する
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_highway import HighwayMultiEnv
from env_config import env_config, HighwayEnvConfig, MODEL_NAME, MAX_STEPS


@dataclass
class EpisodeMetrics:
    """1エピソードの指標を記録するクラス"""
    steps: int = 0
    total_reward: float = 0.0
    speeds: List[float] = None
    headways: List[float] = None
    ttc_violations: int = 0
    lane_changes: int = 0
    crashed: bool = False
    success: bool = False
    
    def __post_init__(self):
        if self.speeds is None:
            self.speeds = []
        if self.headways is None:
            self.headways = []
    
    def record_step(self, speed: float, headway: float, ttc_violation: bool, lane_changed: bool):
        """1ステップの指標を記録"""
        self.steps += 1
        self.speeds.append(speed)
        self.headways.append(headway)
        if ttc_violation:
            self.ttc_violations += 1
        if lane_changed:
            self.lane_changes += 1
    
    def get_avg_speed(self) -> float:
        """平均速度を取得"""
        return np.mean(self.speeds) if self.speeds else 0.0
    
    def get_min_headway(self) -> float:
        """最小車間距離を取得"""
        return np.min(self.headways) if self.headways else float('inf')


def calculate_ttc(ego_pos, ego_speed, other_pos, other_speed) -> Optional[float]:
    """Time To Collision を計算"""
    dx = other_pos[0] - ego_pos[0]
    dy = other_pos[1] - ego_pos[1]
    
    # 前方かつy方向が近い
    if dx > 0 and abs(dy) < 5.0:
        distance = np.linalg.norm([dx, dy])
        rel_speed = ego_speed - other_speed
        if rel_speed > 0.1:  # 前方車の方が遅い
            return distance / rel_speed
    return None


def run_episode(env: HighwayMultiEnv, model: PPO, max_steps: int, render: bool = False) -> EpisodeMetrics:
    """1エピソードを実行し、指標を記録"""
    metrics = EpisodeMetrics()
    obs_dict, info = env.reset()
    
    prev_lane_indices = {}  # 前ステップのレーンインデックス（レーン変更検出用）
    
    for step in range(max_steps):
        if not env.agents:
            break
        
        # 各エージェントの行動を予測
        actions = {}
        for agent in env.agents:
            obs_vec = obs_dict[agent]
            if not isinstance(obs_vec, np.ndarray):
                obs_vec = np.array(obs_vec, dtype=np.float32)
            action, _ = model.predict(obs_vec, deterministic=True)
            actions[agent] = int(action)
        
        # ステップ実行
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        # 報酬を累積
        metrics.total_reward += sum(rewards.values())
        
        # 各車両の指標を記録
        controlled = env._get_controlled_vehicles()
        for i, v in enumerate(controlled):
            if i >= len(env.agents):
                break
            
            speed = float(v.speed)
            
            # 前方車との距離（headway）を計算
            v_pos = np.array(v.position)
            min_headway = 200.0
            ttc_violation = False
            
            for other in env.core_env.road.vehicles:
                if other is v:
                    continue
                
                other_pos = np.array(other.position)
                dx = other_pos[0] - v_pos[0]
                dy = other_pos[1] - v_pos[1]
                
                if dx > 0 and abs(dy) < 5.0:
                    distance = np.linalg.norm([dx, dy])
                    if distance < min_headway:
                        min_headway = distance
                    
                    # TTC計算
                    ttc = calculate_ttc(v_pos, v.speed, other_pos, other.speed)
                    if ttc is not None and ttc < 2.0:
                        ttc_violation = True
            
            # レーン変更検出
            lane_changed = False
            current_lane = getattr(v, "lane_index", None)
            agent_key = env.agents[i] if i < len(env.agents) else None
            if agent_key:
                if agent_key in prev_lane_indices:
                    if current_lane != prev_lane_indices[agent_key]:
                        lane_changed = True
                prev_lane_indices[agent_key] = current_lane
            
            # ステップ指標を記録
            metrics.record_step(speed, min_headway, ttc_violation, lane_changed)
            
            # 衝突検出
            if getattr(v, "crashed", False):
                metrics.crashed = True
        
        # レンダリング
        if render:
            env.render()
        
        # 終了条件チェック
        if any(terminations.values()):
            # 衝突なしで終了なら成功
            if not metrics.crashed:
                metrics.success = True
            break
        
        if any(truncations.values()):
            break
    
    return metrics


def run_test(
    model_path: str,
    config: Optional[HighwayEnvConfig] = None,
    num_episodes: int = 100,
    max_steps: int = 1000,
    render: bool = False,
    save_path: Optional[str] = None
) -> Dict:
    """
    複数エピソードで評価し、統計を計算
    
    Args:
        model_path: 学習済みモデルのパス
        config: 環境設定（Noneの場合はenv_configを使用）
        num_episodes: 評価エピソード数
        max_steps: 1エピソードの最大ステップ数
        render: 描画するかどうか
        save_path: 結果を保存するJSONファイルパス（Noneなら保存しない）
    
    Returns:
        dict: 評価結果の統計
    """
    if config is None:
        config = env_config
    
    # 環境とモデルを初期化
    env = HighwayMultiEnv(config=config, render_mode="human" if render else None)
    model = PPO.load(model_path)
    
    print(f"=== Starting Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print()
    
    # 全エピソードの指標を記録
    all_metrics: List[EpisodeMetrics] = []
    
    for episode in range(num_episodes):
        metrics = run_episode(env, model, max_steps, render=render and (episode % 10 == 0))
        all_metrics.append(metrics)
        
        # 進捗表示
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")
    
    env.close()
    
    # 統計を計算
    success_count = sum(1 for m in all_metrics if m.success)
    collision_count = sum(1 for m in all_metrics if m.crashed)
    episode_rewards = [m.total_reward for m in all_metrics]
    episode_lengths = [m.steps for m in all_metrics]
    all_speeds = [speed for m in all_metrics for speed in m.speeds]
    all_headways = [hw for m in all_metrics for hw in m.headways if hw < 200.0]
    total_ttc_violations = sum(m.ttc_violations for m in all_metrics)
    total_lane_changes = sum(m.lane_changes for m in all_metrics)
    
    results = {
        "num_episodes": num_episodes,
        "success_rate": success_count / num_episodes * 100,
        "collision_rate": collision_count / num_episodes * 100,
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "avg_speed": float(np.mean(all_speeds)) if all_speeds else 0.0,
        "std_speed": float(np.std(all_speeds)) if all_speeds else 0.0,
        "min_headway": float(np.min(all_headways)) if all_headways else float('inf'),
        "avg_headway": float(np.mean(all_headways)) if all_headways else float('inf'),
        "total_danger_count": total_ttc_violations,
        "avg_danger_per_episode": total_ttc_violations / num_episodes,
        "total_lane_changes": total_lane_changes,
        "avg_lane_changes_per_episode": total_lane_changes / num_episodes,
    }
    
    # 結果を表示
    print()
    print("=" * 50)
    print("=== Model Evaluation Results ===")
    print("=" * 50)
    print(f"Episodes: {results['num_episodes']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Collision Rate: {results['collision_rate']:.1f}%")
    print(f"Avg Episode Reward: {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"Avg Episode Length: {results['avg_episode_length']:.1f} ± {results['std_episode_length']:.1f} steps")
    print(f"Avg Speed: {results['avg_speed']:.1f} ± {results['std_speed']:.1f} m/s")
    print(f"Min Headway: {results['min_headway']:.1f} m")
    print(f"Avg Headway: {results['avg_headway']:.1f} m")
    print(f"Total Danger Count (TTC<2s): {results['total_danger_count']}")
    print(f"Avg Danger per Episode: {results['avg_danger_per_episode']:.1f}")
    print(f"Total Lane Changes: {results['total_lane_changes']}")
    print(f"Avg Lane Changes per Episode: {results['avg_lane_changes_per_episode']:.1f}")
    print("=" * 50)
    
    # JSON保存
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {save_path}")
    
    return results


def main():
    """CLI エントリーポイント"""
    parser = argparse.ArgumentParser(description="学習済みモデルの評価")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="学習済みモデルのパス"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="評価エピソード数"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="1エピソードの最大ステップ数"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="一部エピソードを描画（10エピソードごと）"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="結果を保存するJSONファイルパス"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="エージェント数（環境変数より優先）"
    )
    
    args = parser.parse_args()
    
    # 設定を準備
    import copy
    cfg = copy.deepcopy(env_config)
    if args.num_agents is not None:
        cfg.num_agents = args.num_agents
        cfg.controlled_vehicles = args.num_agents
        cfg.vehicles_count = args.num_agents
    
    # 評価実行
    run_test(
        model_path=args.model,
        config=cfg,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_path=args.save_results
    )


if __name__ == "__main__":
    main()

