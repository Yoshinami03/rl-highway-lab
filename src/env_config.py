"""
高速道路合流環境の設定ファイル
このファイルで環境のパラメータを管理します
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from settings import (
    ENV_NAME,
    NUM_AGENTS,
    RENDER_MODE,
    VEHICLES_COUNT,
    CONTROLLED_VEHICLES,
    OBSERVATION_TYPE,
    DURATION,
    OBS_SHAPE,
    OBS_DTYPE,
    ACTION_SPACE_SIZE,
    SPEED_NORMALIZATION,
    CRASH_PENALTY,
)


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
    
    def get_gym_config(self) -> Dict[str, Any]:
        """gym.make()に渡すconfig辞書を生成"""
        return {
            "vehicles_count": self.vehicles_count or self.num_agents,
            "controlled_vehicles": self.controlled_vehicles or self.num_agents,
            "observation": {"type": self.observation_type},
            "duration": self.duration,
        }
    
    def get_render_mode(self, override: Optional[str] = None) -> Optional[str]:
        """レンダーモードを取得（overrideがあれば優先）"""
        return override if override is not None else self.render_mode


# 共通環境設定インスタンス（学習と推論で同じ設定を使用）
# 推論時にレンダリングが必要な場合は、HighwayMultiEnvのrender_mode引数で上書き可能
env_config = HighwayEnvConfig()

# デフォルト設定（後方互換性のため）
default_config = env_config

