import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

# model = PPO.load("ppo_lander", env=env, device="cpu")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_lander")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")