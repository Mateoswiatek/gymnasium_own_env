from __future__ import annotations

import glob
import os
import time
import tqdm

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

from pettingzoo.butterfly import pistonball_v6


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    num_envs = 8

    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(render_mode="rgb_array", **env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
    env = ss.color_reduction_v0(env)
    env = ss.resize_v1(env, x_size=64, y_size=64)
    env = ss.frame_stack_v1(env, 4)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")
    print(f"Training for {steps} steps with {num_envs} parallel environments")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy,
        env,
        verbose=1,  # Changed from 3 to 1 for cleaner output
        batch_size=32,  # Reduced batch size for more frequent updates
        learning_rate=3e-4,
        n_steps=64,  # Add explicit n_steps
        ent_coef=0.1,  # Add some exploration
        vf_coef=0.1,  # Value function coefficient
        gamma=0.99,  # Discount factor
    )

    print("Starting model training...")
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=steps)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Training failed with error: {e}")
        env.close()
        return

    # Save the training curve
    env_name = env.unwrapped.metadata.get('name', 'unknown')
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    model_name = f"{env_name}_{timestamp}"
    model.save(model_name)
    print(f"Model saved as: {model_name}")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()
    

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
    env = ss.color_reduction_v0(env)
    env = ss.resize_v1(env, x_size=64, y_size=64)
    env = ss.frame_stack_v1(env, 4)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
        print(f"Loading model: {latest_policy}")
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    print(f"Running {num_games} evaluation games...")
    for i in tqdm.tqdm(range(num_games)):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    env_fn = pistonball_v6  # Use Pistonball

    env_kwargs = dict(continuous=False, max_cycles=125)  # Adjust max_cycles if needed

    print("=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)
    callback = train(env_fn, steps=50000, seed=0, **env_kwargs)

    print("\n" + "=" * 50)
    print("EVALUATION PHASE")
    print("=" * 50)
    # Evaluate 10 games
    eval(env_fn, num_games=10, **env_kwargs, render_mode="rgb_array")

    print("\n" + "=" * 50)
    print("VISUAL EVALUATION")
    print("=" * 50)
    # Watch 2 games
    eval(env_fn, num_games=10, render_mode="human", **env_kwargs)