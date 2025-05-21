import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm

class RLAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        """
        Agent Q-learning do zarządzania jednostkami w środowisku miastowym.
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table: stan (serializowany np. tuple) -> dict(akcja -> wartość)
        self.q_table = defaultdict(lambda: defaultdict(float))

        self.episode_rewards = []
        self.average_rewards = []


        self.seed = 100

        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, state):
        """Wybiera akcję zgodnie z epsilon-greedy."""
        available_actions = self.env.get_available_actions(state)

        # Eksploracja
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)

        # Eksploatacja
        q_values = self.q_table[state]
        if not q_values:
            return self.rng.choice(available_actions)

        # Wybierz najlepszą znaną akcję
        best_action = max(q_values, key=q_values.get)
        if best_action in available_actions:
            return best_action
        else:
            return self.rng.choice(available_actions)

    def update_q_table(self, state, action, reward, next_state, done):
        s = self.serialize_state(state)
        s_next = self.serialize_state(next_state)

        max_future_q = max(self.q_table[s_next].values()) if not done and self.q_table[s_next] else 0.0
        current_q = self.q_table[s][action]

        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[s][action] = new_q

    def train(self, num_episodes=10000):
        """Trenuje agenta w środowisku."""
        for episode in tqdm.tqdm(range(num_episodes), desc="Trenowanie agenta"):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.update_q_table(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)

            # Statystyka co 100 epizodów
            if episode % 100 == 0 and episode > 0:
                avg = np.mean(self.episode_rewards[-100:])
                self.average_rewards.append(avg)
                self.epsilon = max(0.01, self.epsilon * 0.995)

        return self.q_table

    def evaluate(self, num_episodes=1000):
        """Ewaluacja bez eksploracji."""
        total_rewards = 0

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Eksploatuj (bez epsilon)
                actions = self.env.get_available_actions(state)
                q_values = self.q_table[state]

                if q_values:
                    best_action = max(q_values, key=q_values.get)
                    action = best_action if best_action in actions else self.rng.choice(actions)
                else:
                    action = self.rng.choice(actions)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            total_rewards += episode_reward

        avg_reward = total_rewards / num_episodes
        print(f"Średnia nagroda po {num_episodes} epizodach ewaluacji: {avg_reward:.3f}")
        return avg_reward

    def plot_learning_curve(self):
        """Rysuje krzywą uczenia."""
        plt.figure(figsize=(10, 5))
        plt.plot([i * 100 for i in range(len(self.average_rewards))], self.average_rewards)
        plt.title("Krzywa uczenia (średnia nagroda co 100 epizodów)")
        plt.xlabel("Epizody")
        plt.ylabel("Średnia nagroda")
        plt.grid()
        plt.show()
