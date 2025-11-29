from typing import Optional
import supersuit as ss
from stable_baselines3 import PPO
from run_highway import HighwayMultiEnv
from env_config import env_config, HighwayEnvConfig


def make_env(
    config: Optional[HighwayEnvConfig] = None, 
    num_vec_envs: int = 4, 
    num_cpus: int = 4
):
    """
    ベクトル化された環境を作成
    
    Args:
        config: HighwayEnvConfigインスタンス（Noneの場合はenv_configを使用）
        num_vec_envs: 並列環境数
        num_cpus: 使用するCPU数
    """
    if config is None:
        config = env_config
    
    env = HighwayMultiEnv(config=config)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # concat_vec_envs_v1は環境から自動的に観測空間と行動空間を取得するため、明示的に渡す必要はない
    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs, 
        num_cpus=num_cpus, 
        base_class="stable_baselines3"
    )
    return env


if __name__ == "__main__":
    # 設定ファイルから環境設定を読み込み（学習と推論で同じ設定を使用）
    vec_env = make_env(config=env_config, num_vec_envs=4, num_cpus=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=2000000)
    model.save("highway-merge-ppo")

    vec_env.close()
