# train_agent.py - Training an RL agent
from MicroCivEnv import MicroCivEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

def train_agent():
    # Create a vectorized environment
    env = make_vec_env(lambda: MicroCivEnv(), n_envs=4)

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the agent
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the trained agent
    model.save("microciv_ppo")

    # Close the environment
    env.close()

def evaluate_agent():
    # Load the trained agent
    model = PPO.load("microciv_ppo")

    # Create environment
    env = MicroCivEnv(render_mode="human")

    # Evaluate the agent
    obs, _ = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    train_agent()
    evaluate_agent()