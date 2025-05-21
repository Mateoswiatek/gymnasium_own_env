


import gymnasium as gym
from city_env import CityEnv
from RLAgent import RLAgent

env = CityEnv()

obs, info = env.reset()
done = False

agent = RLAgent(env)


agent.train(num_episodes=10000)
agent.evaluate(num_episodes=1000)
agent.plot_learning_curve()


# while not done:
#     env.render()
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Action taken: {action}, Reward: {reward}")
