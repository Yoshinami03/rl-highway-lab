from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from run_highway import HighwayMultiEnv
from env_config import env_config, HighwayEnvConfig, MODEL_NAME, MAX_STEPS


def run_inference(
    model_path: str, 
    config: Optional[HighwayEnvConfig] = None, 
    max_steps: int = 1000,
    render_mode: Optional[str] = "human"
) -> None:
    """
    学習済みモデルで推論を実行
    
    Args:
        model_path: 学習済みモデルのパス
        config: HighwayEnvConfigインスタンス（Noneの場合はenv_configを使用）
        max_steps: 最大ステップ数
        render_mode: レンダーモード（推論時は"human"を推奨）
    """
    if config is None:
        config = env_config
    
    # 推論時はレンダリングを有効にする（学習時と同じ環境設定を使用）
    env = HighwayMultiEnv(config=config, render_mode=render_mode)
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
