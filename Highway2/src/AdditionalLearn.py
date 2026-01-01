import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from Env import CoopMergeEnv, CoopMergeConfig

# -----------------------
# VecEnv (num_envs=1) : 学習時と同じ作り方
# -----------------------
base = CoopMergeEnv(num_agents=12, config=CoopMergeConfig(), seed=0)

# PettingZoo ParallelEnv -> (gym-like) vector env
venv = ss.pettingzoo_env_to_vec_env_v1(base)

venv = ss.concat_vec_envs_v1(
    venv,
    num_vec_envs=1,          # まずは 1 でOK。並列したければ増やす
    num_cpus=0,              # Colabなら 0 でOK（multiprocessing無し）
    base_class="stable_baselines3",
)
venv = VecMonitor(venv)

# -----------------------
# Load -> (optional tweak) -> Learn more -> Save
# -----------------------
save_path = "./ppo_trained.zip"
model = PPO.load(save_path, env=venv, device="cpu")

# --- 「小さめのランダム性」：探索（entropy）を下げる ---
# PPO の exploration は主に entropy 正則化係数 ent_coef で調整します。
# 例: 既存が ent_coef=0.01 くらいなら 0.001 に下げる。
model.ent_coef = 0.001

# 学習率も落とす
model.learning_rate = 1e-4

# --- checkpoint: 途中保存（任意だが推奨） ---
ckpt_cb = CheckpointCallback(
    save_freq=500_000,                 # 50万stepごとに保存
    save_path="./checkpoints_coopmerge",
    name_prefix="ppo_coopmerge_finetune"
)

# --- 追加学習 ---
model.learn(
    total_timesteps= 100_000,           # 適当に変える
    reset_num_timesteps=False,          # 既存の続きとして学習曲線を繋ぐ
    callback=ckpt_cb,
    progress_bar=True
)

# --- 保存 ---
save_path2 = "./ppo_trained2" + ".zip"
model.save(save_path2)
print("saved:", save_path2)