from __future__ import annotations

import glob
import os
import time
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from pettingzoo.butterfly import pistonball_v6


class LearningCurveCallback(BaseCallback):
    """
    Custom callback for tracking learning metrics during training
    """
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.timesteps = []
        self.mean_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout - this is when we have fresh data"""
        
        # Record timestep
        self.timesteps.append(self.num_timesteps)
        
        # Get episode rewards from the environment's episode buffer
        recent_rewards = []
        
        # Method 1: Try to get from VecMonitor
        if hasattr(self.training_env, 'episode_returns') and len(self.training_env.episode_returns) > 0:
            recent_rewards = list(self.training_env.episode_returns)
            if self.verbose > 1:
                print(f"Got {len(recent_rewards)} episodes from VecMonitor")
        
        # Method 2: Try to get from model's episode info buffer
        elif hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_rewards = []
            for ep_info in self.model.ep_info_buffer:
                if isinstance(ep_info, dict) and 'r' in ep_info:
                    ep_rewards.append(ep_info['r'])
            recent_rewards = ep_rewards
            if self.verbose > 1:
                print(f"Got {len(recent_rewards)} episodes from ep_info_buffer")
        
        # Store rewards and calculate mean
        if recent_rewards:
            self.episode_rewards.extend(recent_rewards)
            # Use recent episodes for mean (last 100 or all if fewer)
            recent_subset = self.episode_rewards[-100:]
            mean_reward = np.mean(recent_subset)
        else:
            # Fallback: use a simple running average of environment rewards
            mean_reward = getattr(self, '_last_mean_reward', 0)
        
        self.mean_rewards.append(mean_reward)
        
        # Get training metrics from the model's logger
        # These are available after each rollout
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            logs = self.model.logger.name_to_value
            
            # Print available keys for debugging (only first few times)
            if len(self.timesteps) <= 3 and self.verbose > 0:
                print(f"Available logger keys: {sorted(list(logs.keys()))}")
            
            # Get losses with various possible key names
            policy_loss = logs.get('train/policy_gradient_loss', 
                                 logs.get('train/policy_loss', 
                                         logs.get('policy_gradient_loss', 0)))
            
            value_loss = logs.get('train/value_loss', 
                                logs.get('value_loss', 0))
            
            entropy_loss = logs.get('train/entropy_loss', 
                                  logs.get('entropy_loss', 0))
        
        self.policy_losses.append(abs(policy_loss))
        self.value_losses.append(abs(value_loss))
        self.entropy_losses.append(abs(entropy_loss))
        
        if self.verbose > 0:
            print(f"Rollout {len(self.timesteps)}: Steps={self.num_timesteps}, "
                  f"Mean Reward={mean_reward:.2f}, "
                  f"Policy Loss={policy_loss:.6f}, "
                  f"Value Loss={value_loss:.6f}, "
                  f"Episodes this rollout={len(recent_rewards)}")
    
    def plot_learning_curve(self, save_path: str = None):
        """Plot the learning curves"""
        if not self.timesteps:
            print("No data to plot - callback may not have been called")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Learning Curves', fontsize=16)
        
        # Plot mean rewards
        axes[0, 0].plot(self.timesteps, self.mean_rewards, 'b-', alpha=0.8, linewidth=2)
        axes[0, 0].set_title('Mean Episode Reward')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        if self.mean_rewards:
            max_reward = max(self.mean_rewards)
            axes[0, 0].axhline(y=max_reward, color='r', linestyle='--', alpha=0.5, 
                              label=f'Max: {max_reward:.2f}')
            axes[0, 0].legend()
        
        # Plot policy loss
        if any(l > 0 for l in self.policy_losses):
            axes[0, 1].plot(self.timesteps, self.policy_losses, 'r-', alpha=0.8, linewidth=2)
            axes[0, 1].set_title('Policy Loss (absolute)')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')  # Log scale for better visualization
        else:
            axes[0, 1].text(0.5, 0.5, 'No policy loss data', 
                           transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title('Policy Loss')
        
        # Plot value loss
        if any(l > 0 for l in self.value_losses):
            axes[1, 0].plot(self.timesteps, self.value_losses, 'g-', alpha=0.8, linewidth=2)
            axes[1, 0].set_title('Value Loss')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')  # Log scale for better visualization
        else:
            axes[1, 0].text(0.5, 0.5, 'No value loss data', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Value Loss')
        
        # Plot entropy loss
        if any(l > 0 for l in self.entropy_losses):
            axes[1, 1].plot(self.timesteps, self.entropy_losses, 'm-', alpha=0.8, linewidth=2)
            axes[1, 1].set_title('Entropy Loss (absolute)')
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')  # Log scale for better visualization
        else:
            axes[1, 1].text(0.5, 0.5, 'No entropy loss data', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Entropy Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        if self.mean_rewards:
            print(f"\nTraining Summary:")
            print(f"- Data points collected: {len(self.timesteps)}")
            print(f"- Total episodes recorded: {len(self.episode_rewards)}")
            print(f"- Final mean reward: {self.mean_rewards[-1]:.2f}")
            print(f"- Max mean reward: {max(self.mean_rewards):.2f}")
            print(f"- Total training steps: {self.timesteps[-1] if self.timesteps else 0}")


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

    # Add VecMonitor to track episode statistics
    env = VecMonitor(env)

    # Create learning curve callback
    eval_freq = min(1000, steps // 20)
    learning_curve_callback = LearningCurveCallback(eval_freq=eval_freq, verbose=1)

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
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        tensorboard_log="./ppo_pistonball_tensorboard/",
    )

    print("Starting model training...")
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=steps, callback=learning_curve_callback)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Training failed with error: {e}")
        env.close()
        return None

    # Save the training curve
    env_name = env.unwrapped.metadata.get('name', 'unknown')
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    model_name = f"{env_name}_{timestamp}"
    model.save(model_name)
    print(f"Model saved as: {model_name}")

    # Plot and save learning curve
    curve_path = f"{model_name}_learning_curve.png"
    learning_curve_callback.plot_learning_curve(save_path=curve_path)

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()
    return learning_curve_callback
    

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
    learning_callback = train(env_fn, steps=50000, seed=0, **env_kwargs)

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