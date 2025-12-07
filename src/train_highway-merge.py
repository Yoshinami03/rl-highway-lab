from typing import Optional
import supersuit as ss
from stable_baselines3 import PPO
from run_highway import HighwayMultiEnv
from env_config import (
    env_config, 
    HighwayEnvConfig, 
    TOTAL_TIMESTEPS,
    MODEL_NAME,
    PPO_POLICY,
    PPO_VERBOSE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_LEARNING_RATE,
    NUM_VEC_ENVS,
    NUM_CPUS,
)


def make_env(
    config: Optional[HighwayEnvConfig] = None, 
    num_vec_envs: Optional[int] = None, 
    num_cpus: Optional[int] = None
):
    """
    ベクトル化された環境を作成
    
    Args:
        config: HighwayEnvConfigインスタンス（Noneの場合はenv_configを使用）
        num_vec_envs: 並列環境数（Noneの場合はNUM_VEC_ENVSを使用）
        num_cpus: 使用するCPU数（Noneの場合はNUM_CPUSを使用）
    """
    if config is None:
        config = env_config
    if num_vec_envs is None:
        num_vec_envs = NUM_VEC_ENVS
    if num_cpus is None:
        num_cpus = NUM_CPUS
    
    env = HighwayMultiEnv(config=config)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs, 
        num_cpus=num_cpus, 
        base_class="stable_baselines3"
    )
    return env


if __name__ == "__main__":
    # 設定ファイルから環境設定を読み込み（学習と推論で同じ設定を使用）
    vec_env = make_env(config=env_config)

    model = PPO(
        PPO_POLICY,
        vec_env,
        verbose=PPO_VERBOSE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        learning_rate=PPO_LEARNING_RATE,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_NAME)

    vec_env.close()
