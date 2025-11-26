"""
高速道路合流環境の設定ファイル
このファイルで環境のパラメータを管理します
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HighwayEnvConfig:
    """高速道路環境の設定クラス"""
    
    # 環境の基本設定
    env_name: str = "highway-v0"
    num_agents: int = 5
    render_mode: str = None  # "human" or None
    
    # 環境パラメータ
    vehicles_count: int = None  # Noneの場合はnum_agentsを使用
    controlled_vehicles: int = None  # Noneの場合はnum_agentsを使用
    observation_type: str = "Kinematics"
    duration: int = 40
    
    # 観測空間の設定
    obs_shape: tuple = (3,)  # [x位置, y位置, 速度]
    obs_dtype: str = "float32"
    
    # 行動空間の設定
    action_space_size: int = 5  # 0:減速, 1:維持, 2:加速, 3:左レーン変更, 4:右レーン変更
    
    # 報酬設定
    speed_normalization: float = 30.0  # 速度の正規化係数
    crash_penalty: float = -100.0  # 衝突時のペナルティ
    
    def get_gym_config(self) -> Dict[str, Any]:
        """gym.make()に渡すconfig辞書を生成"""
        return {
            "vehicles_count": self.vehicles_count or self.num_agents,
            "controlled_vehicles": self.controlled_vehicles or self.num_agents,
            "observation": {"type": self.observation_type},
            "duration": self.duration,
        }
    
    def get_render_mode(self, override: str = None) -> str:
        """レンダーモードを取得（overrideがあれば優先）"""
        return override if override is not None else self.render_mode


# デフォルト設定インスタンス
default_config = HighwayEnvConfig()

# 学習用設定（レンダリングなし）
train_config = HighwayEnvConfig(
    num_agents=5,
    render_mode=None,
    duration=40,
)

# 推論用設定（レンダリングあり）
inference_config = HighwayEnvConfig(
    num_agents=5,
    render_mode="human",
    duration=40,
)

