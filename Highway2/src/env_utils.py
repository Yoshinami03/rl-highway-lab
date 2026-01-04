"""環境作成のユーティリティ関数"""
from typing import Optional

import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor

from config import CoopMergeConfig
from Env import CoopMergeEnv


def make_vec_env(num_agents: int = 12, num_envs: int = 1, seed: int = 0,
                 config: Optional[CoopMergeConfig] = None):
    """PettingZoo環境をSB3用VecEnvに変換"""
    cfg = config or CoopMergeConfig()
    base = CoopMergeEnv(num_agents=num_agents, config=cfg, seed=seed)
    venv = ss.pettingzoo_env_to_vec_env_v1(base)
    venv = ss.concat_vec_envs_v1(
        venv,
        num_vec_envs=num_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    venv = VecMonitor(venv)
    return venv
