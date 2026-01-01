import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from Env import CoopMergeEnv, CoopMergeConfig

base = CoopMergeEnv(num_agents=12, config=CoopMergeConfig(), seed=0)

# PettingZoo ParallelEnv -> (gym-like) vector env
venv = ss.pettingzoo_env_to_vec_env_v1(base)

# ★ここが必須：SB3 がそのまま食える VecEnv にする（num_vec_envs=1 でもやる）
venv = ss.concat_vec_envs_v1(
    venv,
    num_vec_envs=1,          # まずは 1 でOK。並列したければ増やす
    num_cpus=0,              # Colabなら 0 でOK（multiprocessing無し）
    base_class="stable_baselines3",
)
venv = VecMonitor(venv)

model = PPO("MlpPolicy", venv, verbose=1)


# --- checkpoint: 途中保存（任意だが推奨） ---
ckpt_cb = CheckpointCallback(
    save_freq=500_000,                 # 50万stepごとに保存
    save_path="./checkpoints_coopmerge",
    name_prefix="ppo_coopmerge_finetune"
)

# 学習
model.learn(
    total_timesteps=100_000,           #★実際には 5 000 000 くらい必要
    reset_num_timesteps=False,         # 既存の続きとして学習曲線を繋ぐ
    callback=ckpt_cb,
    progress_bar=True
)

# --- 保存 ---
save_path = "./ppo_trained" + ".zip"
model.save(save_path)
print("saved:", save_path)