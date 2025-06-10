import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt

# Utwórz środowisko wieloagentowe
env = simple_spread_v3.parallel_env()
env.reset()

# Przekształć środowisko na pojedynczego agenta (dla PPO)
from supersuit import pad_observations_v0, pad_action_space_v0, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

env = pad_observations_v0(env)
env = pad_action_space_v0(env)
vec_env = pettingzoo_env_to_vec_env_v1(env)
vec_env = concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class='stable_baselines3')

# Trening PPO
model = PPO("MlpPolicy", vec_env, verbose=1)
rewards = []

for i in range(100):  # 100 epok treningowych
    model.learn(total_timesteps=10000)
    obs = vec_env.reset()
    episode_rewards = []
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        episode_rewards.append(np.mean(reward))
        if done.any():
            break
    rewards.append(np.sum(episode_rewards))

# Zapisz wytrenowany model
model.save("ppo_simple_spread")

# Krzywa uczenia
plt.plot(rewards)
plt.xlabel('Epoka')
plt.ylabel('Suma nagród')
plt.title('Krzywa uczenia PPO w środowisku wieloagentowym')
plt.savefig('learning_curve.png')
plt.show()

env = simple_spread_v3.parallel_env(render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    actions = {}
    for agent in env.agents:
        action, _ = model.predict(np.array([obs[agent]]), deterministic=True)
        actions[agent] = action[0]
    obs, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    if all(terminations.values()) or all(truncations.values()):
        break

env.close()