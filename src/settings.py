"""
環境設定ファイル
学習と推論で使用する環境パラメータを定義します
"""

# 環境の基本設定
ENV_NAME = "merge-v0"
NUM_AGENTS = 5
RENDER_MODE = None  # 学習時はNone、推論時は"human"を指定

# 環境パラメータ
VEHICLES_COUNT = None  # Noneの場合はNUM_AGENTSを使用
CONTROLLED_VEHICLES = None  # Noneの場合はNUM_AGENTSを使用
OBSERVATION_TYPE = "Kinematics"
DURATION = 40

# 観測空間の設定
OBS_SHAPE = (3,)  # [x位置, y位置, 速度]
OBS_DTYPE = "float32"

# 行動空間の設定
ACTION_SPACE_SIZE = 5  # 0:減速, 1:維持, 2:加速, 3:左レーン変更, 4:右レーン変更

# 報酬設定
SPEED_NORMALIZATION = 30.0  # 速度の正規化係数
CRASH_PENALTY = -100.0  # 衝突時のペナルティ

