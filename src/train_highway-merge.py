import os
import sys
from typing import Optional
import supersuit as ss
from stable_baselines3 import PPO

# src 配下のローカル highway_env を優先して import する
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    
    この関数は、マルチエージェント環境を各エージェントが独立に学習できる形に変換します。
    pettingzoo_env_to_vec_env_v1() は各エージェントを個別の環境インスタンスとして扱い、
    同じモデル（PPO）を各エージェントに適用することで、協調行動を学習させます。
    
    Args:
        config: HighwayEnvConfigインスタンス（Noneの場合はenv_configを使用）
        num_vec_envs: 並列環境数（Noneの場合はNUM_VEC_ENVSを使用）
        num_cpus: 使用するCPU数（Noneの場合はNUM_CPUSを使用）
    
    注意:
        - 学習モデルは1台分のもの（単一のPPOモデル）
        - 各エージェントは同じモデルを使用して独立に行動を決定
        - 各エージェントは自分の観測のみを見て、自分の行動を決定
        - これにより、同じモデルを複製して使う形になり、協調行動が自然に生まれる
    """
    if config is None:
        config = env_config
    if num_vec_envs is None:
        num_vec_envs = NUM_VEC_ENVS
    if num_cpus is None:
        num_cpus = NUM_CPUS
    
    # マルチエージェント環境を作成
    env = HighwayMultiEnv(config=config)
    
    # PettingZooのParallelEnvをVecEnvに変換
    # この変換により、各エージェントが個別の環境インスタンスとして扱われる
    # 観測shape: (num_agents, obs_dim) - 各エージェントの観測が並ぶ
    # 行動shape: (num_agents,) - 各エージェントの行動が並ぶ
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 複数の環境インスタンスを並列実行（データ収集の高速化）
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
