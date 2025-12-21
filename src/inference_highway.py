import os
import sys
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
import copy

# src 配下のローカル highway_env を優先して import する
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_config import env_config, HighwayEnvConfig, MODEL_NAME, MAX_STEPS, INFER_NUM_AGENTS, INFER_CONTROLLED_VEHICLES, INFER_VEHICLES_COUNT
from run_highway import HighwayMultiEnv


def run_inference(
    model_path: str, 
    config: Optional[HighwayEnvConfig] = None, 
    max_steps: int = 1000,
    render_mode: Optional[str] = "human",
    num_agents: Optional[int] = None,
) -> None:
    """
    学習済みモデルで推論を実行
    
    Args:
        model_path: 学習済みモデルのパス
        config: HighwayEnvConfigインスタンス（Noneの場合はenv_configを使用）
        max_steps: 最大ステップ数
        render_mode: レンダーモード（推論時は"human"を推奨）
    """
    # 設定は推論用にコピーして上書きする（学習設定を汚さない）
    cfg = copy.deepcopy(config if config is not None else env_config)

    # 優先順位: 引数 num_agents > 環境変数 INFER_NUM_AGENTS > 元設定
    if num_agents is not None:
        cfg.num_agents = num_agents
    elif INFER_NUM_AGENTS is not None:
        cfg.num_agents = INFER_NUM_AGENTS

    # controlled_vehicles / vehicles_count も推論専用上書きがあれば適用
    if INFER_CONTROLLED_VEHICLES is not None:
        cfg.controlled_vehicles = INFER_CONTROLLED_VEHICLES
    if INFER_VEHICLES_COUNT is not None:
        cfg.vehicles_count = INFER_VEHICLES_COUNT

    # 推論時はレンダリングを有効にする（学習時と同じ環境設定を使用）
    env = HighwayMultiEnv(config=cfg, num_agents=cfg.num_agents, render_mode=render_mode)
    model = PPO.load(model_path)

    obs_dict, info = env.reset()

    for _ in range(max_steps):
        if not env.agents:
            break

        actions = {}
        for agent in env.agents:
            obs_vec = obs_dict[agent]
            if isinstance(obs_vec, np.ndarray):
                pass
            else:
                obs_vec = np.array(obs_vec, dtype=np.float32)
            action, _ = model.predict(obs_vec, deterministic=True)
            actions[agent] = int(action)

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()


if __name__ == "__main__":
    # 設定ファイルから環境設定を読み込み（学習時と同じ設定を使用）
    run_inference(MODEL_NAME, config=env_config, max_steps=MAX_STEPS, render_mode="human")
