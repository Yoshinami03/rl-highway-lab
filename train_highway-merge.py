import supersuit as ss
from stable_baselines3 import PPO
from run_highway import HighwayMultiEnv


def make_env(num_agents: int, num_vec_envs: int = 4, num_cpus: int = 4):
    env = HighwayMultiEnv(num_agents=num_agents, render_mode=None)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    return env


if __name__ == "__main__":
    vec_env = make_env(num_agents=5, num_vec_envs=4, num_cpus=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=200000)
    model.save("highway-merge-ppo")

    vec_env.close()
