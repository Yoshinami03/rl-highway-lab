import supersuit as ss
from pettingzoo.utils import parallel_to_aec
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from run_highway import HighwayMultiEnv

def make_env(num_agents):
    env = HighwayMultiEnv(num_agents=num_agents)
    env = parallel_to_aec(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class="stable_baselines3")
    return env

if __name__ == "__main__":
    vec_env = make_env(num_agents=5)

    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=1024, batch_size=256, learning_rate=3e-4)

    model.learn(total_timesteps=200000)

    model.save("highway-merge-ppo")

    vec_env.close()