import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import pickle
import time
from typing import Dict, List, Tuple
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingCallback(BaseCallback):
    """
    Callback for tracking training progress and collecting metrics
    """
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super(TrainingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Record episode data when episode ends
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    self.timesteps.append(self.num_timesteps)
        return True

class CustomNetwork(BaseFeaturesExtractor):
    """
    Custom neural network architecture for testing different network structures
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, architecture: str = "deep"):
        super(CustomNetwork, self).__init__(observation_space, features_dim)

        if architecture == "deep":
            # Deep network with 4 hidden layers
            self.net = nn.Sequential(
                nn.Linear(observation_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, features_dim),
                nn.ReLU()
            )
        elif architecture == "wide":
            # Wide network with fewer but larger layers
            self.net = nn.Sequential(
                nn.Linear(observation_space.shape[0], 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, features_dim),
                nn.ReLU()
            )
        else:  # simple
            # Simple network with 2 hidden layers
            self.net = nn.Sequential(
                nn.Linear(observation_space.shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, features_dim),
                nn.ReLU()
            )

    def forward(self, observations):
        return self.net(observations)

class RLExperiment:
    """
    Main class for conducting reinforcement learning experiments
    """

    def __init__(self, env_name: str = "LunarLanderContinuous-v3", algorithm: str = "PPO"):
        self.env_name = env_name
        self.algorithm_name = algorithm
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf

        # Create environment for testing
        self.env = gym.make(env_name)
        print(f"Environment: {env_name}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")

    def run_single_experiment(self, hyperparams: Dict, run_id: int, total_timesteps: int = 50000) -> Tuple[List, List, float]:
        """
        Run a single training experiment with given hyperparameters
        """
        print(f"Running experiment {run_id} with hyperparams: {hyperparams}")

        # Create monitored environment
        env = Monitor(gym.make(self.env_name))

        # Create model based on algorithm
        if self.algorithm_name == "PPO":
            model = PPO("MlpPolicy", env, **hyperparams, verbose=0)
        elif self.algorithm_name == "A2C":
            model = A2C("MlpPolicy", env, **hyperparams, verbose=0)
        elif self.algorithm_name == "SAC":
            model = SAC("MlpPolicy", env, **hyperparams, verbose=0)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

        # Setup callback for tracking
        callback = TrainingCallback()

        # Measure training time
        start_time = time.time()

        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=callback)

        training_time = time.time() - start_time

        # Evaluate final performance
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        # Clean up
        env.close()

        return callback.episode_rewards, callback.timesteps, mean_reward, training_time, model

    def run_hyperparameter_study(self, hyperparams_sets: List[Dict], n_runs: int = 10, total_timesteps: int = 50000):
        """
        Run multiple experiments with different hyperparameters
        """
        self.results = {}

        for i, hyperparams in enumerate(hyperparams_sets):
            param_name = f"Config_{i+1}"
            print(f"\n--- Testing {param_name} ---")

            episode_rewards_all = []
            timesteps_all = []
            final_scores = []
            training_times = []
            models = []

            for run in range(n_runs):
                rewards, timesteps, final_score, train_time, model = self.run_single_experiment(
                    hyperparams, run, total_timesteps
                )

                episode_rewards_all.append(rewards)
                timesteps_all.append(timesteps)
                final_scores.append(final_score)
                training_times.append(train_time)
                models.append(model)

                # Track best model
                if final_score > self.best_score:
                    self.best_score = final_score
                    self.best_model = model

                print(f"Run {run+1}: Final reward = {final_score:.2f}, Time = {train_time:.1f}s")

            self.results[param_name] = {
                'hyperparams': hyperparams,
                'episode_rewards': episode_rewards_all,
                'timesteps': timesteps_all,
                'final_scores': final_scores,
                'training_times': training_times,
                'models': models,
                'avg_final_score': np.mean(final_scores),
                'std_final_score': np.std(final_scores),
                'avg_training_time': np.mean(training_times)
            }

            print(f"Average final reward: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
            print(f"Average training time: {np.mean(training_times):.1f}s")

    def plot_learning_curves(self, save_path: str = None):
        """
        Plot learning curves for all hyperparameter configurations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Curves - {self.algorithm_name} on {self.env_name}', fontsize=16)

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        # Plot 1: Individual runs
        ax1 = axes[0, 0]
        for i, (config_name, data) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            for j, (rewards, timesteps) in enumerate(zip(data['episode_rewards'], data['timesteps'])):
                if len(rewards) > 0 and len(timesteps) > 0:
                    # Smooth the rewards for better visualization
                    if len(rewards) > 10:
                        smoothed_rewards = self._smooth_data(rewards, window=min(10, len(rewards)//5))
                        ax1.plot(timesteps[:len(smoothed_rewards)], smoothed_rewards,
                                 color=color, alpha=0.3, linewidth=0.5)

        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Individual Training Runs')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Average curves with confidence intervals
        ax2 = axes[0, 1]
        for i, (config_name, data) in enumerate(self.results.items()):
            color = colors[i % len(colors)]

            # Aggregate data across runs
            all_timesteps, mean_rewards, std_rewards = self._aggregate_curves(
                data['episode_rewards'], data['timesteps']
            )

            if len(all_timesteps) > 0:
                ax2.plot(all_timesteps, mean_rewards, color=color, linewidth=2, label=config_name)
                ax2.fill_between(all_timesteps,
                                 np.array(mean_rewards) - np.array(std_rewards),
                                 np.array(mean_rewards) + np.array(std_rewards),
                                 color=color, alpha=0.2)

        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Average Episode Reward')
        ax2.set_title('Average Learning Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Final performance comparison
        ax3 = axes[1, 0]
        config_names = list(self.results.keys())
        final_means = [self.results[name]['avg_final_score'] for name in config_names]
        final_stds = [self.results[name]['std_final_score'] for name in config_names]

        bars = ax3.bar(config_names, final_means, yerr=final_stds, capsize=5,
                       color=colors[:len(config_names)], alpha=0.7)
        ax3.set_ylabel('Final Average Reward')
        ax3.set_title('Final Performance Comparison')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean, std in zip(bars, final_means, final_stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                     f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')

        # Plot 4: Training time comparison
        ax4 = axes[1, 1]
        training_times = [self.results[name]['avg_training_time'] for name in config_names]
        bars = ax4.bar(config_names, training_times, color=colors[:len(config_names)], alpha=0.7)
        ax4.set_ylabel('Average Training Time (seconds)')
        ax4.set_title('Training Time Comparison')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{time_val:.1f}s', ha='center', va='bottom')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _smooth_data(self, data: List, window: int = 10) -> List:
        """Smooth data using moving average"""
        if len(data) < window:
            return data

        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        return smoothed

    def _aggregate_curves(self, all_rewards: List[List], all_timesteps: List[List],
                          max_timesteps: int = 50000, step_size: int = 1000) -> Tuple[List, List, List]:
        """Aggregate learning curves across multiple runs"""
        timestep_points = list(range(0, max_timesteps + step_size, step_size))
        aggregated_rewards = []

        for target_timestep in timestep_points:
            rewards_at_timestep = []

            for rewards, timesteps in zip(all_rewards, all_timesteps):
                if len(rewards) == 0 or len(timesteps) == 0:
                    continue

                # Find closest timestep
                closest_idx = min(range(len(timesteps)),
                                  key=lambda i: abs(timesteps[i] - target_timestep))

                if abs(timesteps[closest_idx] - target_timestep) <= step_size:
                    rewards_at_timestep.append(rewards[closest_idx])

            if rewards_at_timestep:
                aggregated_rewards.append(rewards_at_timestep)
            else:
                aggregated_rewards.append([])

        # Calculate mean and std
        mean_rewards = []
        std_rewards = []
        valid_timesteps = []

        for i, rewards in enumerate(aggregated_rewards):
            if len(rewards) > 0:
                mean_rewards.append(np.mean(rewards))
                std_rewards.append(np.std(rewards))
                valid_timesteps.append(timestep_points[i])

        return valid_timesteps, mean_rewards, std_rewards

    def test_network_architectures(self, hyperparams: Dict, architectures: List[str],
                                   n_runs: int = 5, total_timesteps: int = 30000):
        """
        Test different network architectures
        """
        architecture_results = {}

        for arch in architectures:
            print(f"\n--- Testing {arch} architecture ---")

            scores = []
            times = []

            for run in range(n_runs):
                env = Monitor(gym.make(self.env_name))

                # Create policy kwargs with custom network
                policy_kwargs = {
                    "features_extractor_class": CustomNetwork,
                    "features_extractor_kwargs": {"architecture": arch}
                }

                # Create model with custom network
                if self.algorithm_name == "PPO":
                    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                                **hyperparams, verbose=0)
                elif self.algorithm_name == "A2C":
                    model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs,
                                **hyperparams, verbose=0)

                start_time = time.time()
                model.learn(total_timesteps=total_timesteps)
                train_time = time.time() - start_time

                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

                scores.append(mean_reward)
                times.append(train_time)

                env.close()
                print(f"Run {run+1}: {mean_reward:.2f}")

            architecture_results[arch] = {
                'scores': scores,
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'avg_time': np.mean(times)
            }

            print(f"Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

        return architecture_results

    def evaluate_best_model(self, n_episodes: int = 20):
        """
        Evaluate the best trained model with deterministic actions
        """
        if self.best_model is None:
            print("No trained model available!")
            return None

        print(f"\n--- Evaluating Best Model (Score: {self.best_score:.2f}) ---")

        env = gym.make(self.env_name)

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                # Use deterministic actions (no exploration)
                action, _ = self.best_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {steps}")

        env.close()

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"\nDeterministic Performance:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Best Episode: {max(episode_rewards):.2f}")
        print(f"Worst Episode: {min(episode_rewards):.2f}")

        return {
            'episode_rewards': episode_rewards,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': np.mean(episode_lengths)
        }

    def save_best_model(self, filepath: str = "best_model"):
        """Save the best model to disk"""
        if self.best_model is not None:
            self.best_model.save(filepath)
            print(f"Best model saved to {filepath}")
        else:
            print("No model to save!")

    def generate_report(self):
        """
        Generate a comprehensive report of the experiments
        """
        print("\n" + "="*80)
        print("EXPERIMENT REPORT")
        print("="*80)

        print(f"\nEnvironment: {self.env_name}")
        print(f"Algorithm: {self.algorithm_name}")

        print(f"\nEnvironment Characteristics:")
        print(f"- Observation Space: {self.env.observation_space}")
        print(f"- Action Space: {self.env.action_space}")
        print(f"- Continuous Actions: {isinstance(self.env.action_space, gym.spaces.Box)}")

        print(f"\nHyperparameter Configurations Tested:")
        for config_name, data in self.results.items():
            print(f"\n{config_name}:")
            for param, value in data['hyperparams'].items():
                print(f"  - {param}: {value}")
            print(f"  - Average Final Score: {data['avg_final_score']:.2f} ± {data['std_final_score']:.2f}")
            print(f"  - Average Training Time: {data['avg_training_time']:.1f} seconds")

        print(f"\nBest Configuration:")
        best_config = max(self.results.items(), key=lambda x: x[1]['avg_final_score'])
        print(f"- Configuration: {best_config[0]}")
        print(f"- Average Score: {best_config[1]['avg_final_score']:.2f}")
        print(f"- Best Individual Score: {self.best_score:.2f}")

def main():
    """
    Main function to run the complete experiment
    """
    print("Starting Reinforcement Learning Experiment")
    print("="*50)

    # Initialize experiment
    experiment = RLExperiment(env_name="LunarLanderContinuous-v3", algorithm="PPO")

    # Define hyperparameter sets to test
    hyperparams_sets = [
        {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95
        },
        {
            'learning_rate': 1e-4,
            'n_steps': 1024,
            'batch_size': 32,
            'gamma': 0.995,
            'gae_lambda': 0.9
        },
        {
            'learning_rate': 5e-4,
            'n_steps': 4096,
            'batch_size': 128,
            'gamma': 0.98,
            'gae_lambda': 0.98
        }
    ]

    # Run hyperparameter study (4 points requirement)
    print("Phase 1: Hyperparameter Study")
    experiment.run_hyperparameter_study(hyperparams_sets, n_runs=10, total_timesteps=50_000)

    # Plot learning curves
    print("\nPhase 2: Generating Learning Curves")
    experiment.plot_learning_curves(save_path="learning_curves.png")

    # Test different network architectures (6 points requirement)
    print("\nPhase 3: Testing Network Architectures")
    best_hyperparams = hyperparams_sets[0]  # Use first config for architecture testing
    architectures = ["simple", "deep", "wide"]
    arch_results = experiment.test_network_architectures(best_hyperparams, architectures)

    # Print architecture results
    print("\nArchitecture Comparison:")
    for arch, results in arch_results.items():
        print(f"{arch}: {results['avg_score']:.2f} ± {results['std_score']:.2f}")

    # Evaluate best model deterministically (8 points requirement)
    print("\nPhase 4: Deterministic Evaluation of Best Model")
    det_results = experiment.evaluate_best_model(n_episodes=20)

    # Save the best model
    experiment.save_best_model("best_ppo_model")

    # Generate comprehensive report
    experiment.generate_report()

    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("Files generated:")
    print("- learning_curves.png: Learning curves visualization")
    print("- best_ppo_model.zip: Best trained model")

if __name__ == "__main__":
    main()

# pip install "gymnasium[box2d]"