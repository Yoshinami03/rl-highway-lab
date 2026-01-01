import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from Env import CoopMergeEnv, CoopMergeConfig

NUM_ENVS   = 12   # ← concat の複製数（= インスタンス数）
NUM_AGENTS = 12   # ← 1インスタンス内のエージェント数

def make_vec_env(num_envs=NUM_ENVS, seed=0):
    base = CoopMergeEnv(num_agents=NUM_AGENTS, config=CoopMergeConfig(), seed=seed)
    venv = ss.pettingzoo_env_to_vec_env_v1(base)
    venv = ss.concat_vec_envs_v1(
        venv, num_vec_envs=num_envs, num_cpus=0, base_class="stable_baselines3"
    )
    venv = VecMonitor(venv)
    return venv

def step_vecenv_compat(venv, action):
    out = venv.step(action)
    if len(out) == 4:
        obs, rew, done, infos = out
        return obs, rew, done, infos
    if len(out) == 5:
        obs, rew, term, trunc, infos = out
        done = np.logical_or(term, trunc)
        return obs, rew, done, infos
    raise RuntimeError("unexpected step format")

def eval_team_over_seeds(model_path, seeds, max_steps=500, device="cpu", check_dup=True):
    all_team_returns = []

    for s in seeds:
        venv = make_vec_env(NUM_ENVS, seed=int(s))
        assert venv.num_envs == NUM_ENVS * NUM_AGENTS, venv.num_envs  # 144 のはず

        model = PPO.load(model_path, env=venv, device=device)

        obs = venv.reset()
        ep_team = np.zeros(NUM_ENVS, dtype=np.float64)  # ★ 12本（インスタンス数）だけ積算

        for _ in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = step_vecenv_compat(venv, act)

            rew = np.asarray(rew, dtype=np.float64).reshape(NUM_ENVS, NUM_AGENTS)

            if check_dup:
                # インスタンス内で12エージェント分の報酬が同一か確認
                max_err = np.max(np.abs(rew - rew[:, [0]]))
                if max_err > 1e-9:
                    print(f"[seed {s}] WARNING: reward not duplicated within instance. max_err={max_err:e}")

            ep_team += rew[:, 0]  # ★ 代表1本だけ足す（=チーム報酬）

            # done も同様にインスタンス代表で判定（どれも同じなら [:,0] でOK）
            done2 = np.asarray(done).reshape(NUM_ENVS, NUM_AGENTS)
            if np.all(done2[:, 0]):
                break

        all_team_returns.extend(ep_team.tolist())
        venv.close()

    return np.array(all_team_returns, dtype=np.float64)

# 実行
# 保存済みモデルを選択
# model_path = "./ppo_coopmerge_finetuned_more.zip" # pretrained model
save_path2 = "./ppo_trained2.zip"
model_path = save_path2

seeds = list(range(1000, 1020))

rets = eval_team_over_seeds(model_path, seeds)

print("team eval over seeds:")
print("  n      =", len(rets))  # ★ 240 のはず
print("  mean   =", float(rets.mean()))
print("  std    =", float(rets.std()))
print("  min/max=", float(rets.min()), float(rets.max()))
