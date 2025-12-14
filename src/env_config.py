"""
高速道路合流環境の設定ファイル
このファイルで環境のパラメータを管理します
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple

# .envファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenvがインストールされていない場合は環境変数のみを使用
    pass


def get_env_str(key: str, default: str = "") -> str:
    """環境変数を文字列として取得"""
    return os.getenv(key, default).strip()


def get_env_int(key: str, default: int = 0) -> int:
    """環境変数を整数として取得"""
    value = get_env_str(key)
    return int(value) if value else default


def get_env_float(key: str, default: float = 0.0) -> float:
    """環境変数を浮動小数点数として取得"""
    value = get_env_str(key)
    return float(value) if value else default


def get_env_optional_int(key: str) -> Optional[int]:
    """環境変数をオプショナル整数として取得（空欄の場合はNone）"""
    value = get_env_str(key)
    return int(value) if value else None


def get_env_tuple(key: str, default: Tuple[int, ...] = (3,)) -> Tuple[int, ...]:
    """環境変数をタプルとして取得（カンマ区切り）"""
    value = get_env_str(key)
    if not value:
        return default
    try:
        # カンマ区切りの場合は分割、そうでない場合は単一値
        if ',' in value:
            return tuple(int(x.strip()) for x in value.split(','))
        else:
            return (int(value),)
    except ValueError:
        return default


# 環境変数から設定を読み込む
ENV_NAME = get_env_str("ENV_NAME", "merge-v0")
NUM_AGENTS = get_env_int("NUM_AGENTS", 5)
RENDER_MODE = get_env_str("RENDER_MODE") or None
VEHICLES_COUNT = get_env_optional_int("VEHICLES_COUNT")
CONTROLLED_VEHICLES = get_env_optional_int("CONTROLLED_VEHICLES")
OBSERVATION_TYPE = get_env_str("OBSERVATION_TYPE", "Kinematics")
DURATION = get_env_int("DURATION", 40)
OBS_SHAPE = get_env_tuple("OBS_SHAPE", (3,))
OBS_DTYPE = get_env_str("OBS_DTYPE", "float32")
ACTION_SPACE_SIZE = get_env_int("ACTION_SPACE_SIZE", 5)
SPEED_NORMALIZATION = get_env_float("SPEED_NORMALIZATION", 30.0)
CRASH_PENALTY = get_env_float("CRASH_PENALTY", -100.0)

# カメラズーム
CAMERA_ZOOM = get_env_float("CAMERA_ZOOM", 5.0)

# シミュレーション設定
SIMULATION_FREQUENCY = get_env_int("SIMULATION_FREQUENCY", 5)  # 低くすると進行がゆっくりになる

# 学習設定
TOTAL_TIMESTEPS = get_env_int("TOTAL_TIMESTEPS", 2000000)

# モデル設定
MODEL_NAME = get_env_str("MODEL_NAME", "highway-merge-ppo")

# 学習ハイパーパラメータ
PPO_POLICY = get_env_str("PPO_POLICY", "MlpPolicy")
PPO_VERBOSE = get_env_int("PPO_VERBOSE", 1)
PPO_N_STEPS = get_env_int("PPO_N_STEPS", 1024)
PPO_BATCH_SIZE = get_env_int("PPO_BATCH_SIZE", 256)
PPO_LEARNING_RATE = get_env_float("PPO_LEARNING_RATE", 0.0003)

# ベクトル化環境設定
NUM_VEC_ENVS = get_env_int("NUM_VEC_ENVS", 4)
NUM_CPUS = get_env_int("NUM_CPUS", 4)

# 推論設定
MAX_STEPS = get_env_int("MAX_STEPS", 1000)


@dataclass
class HighwayEnvConfig:
    """高速道路環境の設定クラス"""
    
    # 環境の基本設定
    env_name: str = ENV_NAME
    num_agents: int = NUM_AGENTS
    render_mode: Optional[str] = RENDER_MODE  # "human" or None
    
    # 環境パラメータ
    vehicles_count: Optional[int] = VEHICLES_COUNT  # Noneの場合はnum_agentsを使用
    controlled_vehicles: Optional[int] = CONTROLLED_VEHICLES  # Noneの場合はnum_agentsを使用
    observation_type: str = OBSERVATION_TYPE
    duration: int = DURATION
    
    # 観測空間の設定
    obs_shape: tuple = OBS_SHAPE  # [x位置, y位置, 速度]
    obs_dtype: str = OBS_DTYPE
    
    # 行動空間の設定
    action_space_size: int = ACTION_SPACE_SIZE  # 0:減速, 1:維持, 2:加速, 3:左レーン変更, 4:右レーン変更
    
    # 報酬設定
    speed_normalization: float = SPEED_NORMALIZATION  # 速度の正規化係数
    crash_penalty: float = CRASH_PENALTY  # 衝突時のペナルティ

    # 時間設定
    simulation_frequency: int = SIMULATION_FREQUENCY

    # カメラ設定（ズームのみ）
    camera_zoom: float = CAMERA_ZOOM
    
    def get_gym_config(self) -> Dict[str, Any]:
        """gym.make()に渡すconfig辞書を生成"""
        return {
            "vehicles_count": self.vehicles_count or self.num_agents,
            "controlled_vehicles": self.controlled_vehicles or self.num_agents,
            "observation": {"type": self.observation_type},
            "duration": self.duration,
            "simulation_frequency": self.simulation_frequency,
            "real_time_rendering": True,  # viewerのtickを有効化して実時間描画
            "scaling": self.camera_zoom,  # ズーム倍率（数値を下げると広く見える）
        }
    
    def get_render_mode(self, override: Optional[str] = None) -> Optional[str]:
        """レンダーモードを取得（overrideがあれば優先）"""
        return override if override is not None else self.render_mode


# 共通環境設定インスタンス（学習と推論で同じ設定を使用）
# 推論時にレンダリングが必要な場合は、HighwayMultiEnvのrender_mode引数で上書き可能
env_config = HighwayEnvConfig()

# デフォルト設定（後方互換性のため）
default_config = env_config

