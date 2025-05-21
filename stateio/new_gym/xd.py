import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Any
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm


class CityEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 grid_size: int = 10,
                 world_size: int = 10,
                 num_cities: int = 5,
                 seed: int = 42,
                 max_units: int = 10):
        super(CityEnv, self).__init__()

        self.grid_size = grid_size
        self.world_size = world_size
        self.num_cities = num_cities
        self.max_units = max_units
        self.seed = seed
        self.np_random = np.random.default_rng(seed)

        self.graph = nx.Graph()
        self.cities = []
        self._generate_cities()

        # Akcje: (miasto startowe, miasto docelowe) + akcja "pass"
        self.edges = list(self.graph.edges)
        self.action_space = spaces.Discrete(len(self.edges) + 1)  # +1 dla pass (nic nie robi)

        self.edge_to_action = {edge: idx for idx, edge in enumerate(self.edges)}
        self.action_to_edge = {v: k for k, v in self.edge_to_action.items()}
        self.pass_action = len(self.edges)  # indeks akcji "pass"

        # Obserwacja: liczba jednostek i właściciele
        self.observation_space = spaces.Dict({
            "units": spaces.Box(low=0, high=max_units, shape=(num_cities,), dtype=np.int32),
            "owners": spaces.Box(low=-1, high=1, shape=(num_cities,), dtype=np.int8),  # -1 brak właściciela, 0 agent
        })

        self.units = np.zeros(num_cities, dtype=np.int32)
        self.owners = np.full(num_cities, -1, dtype=np.int8)

        self.reset()

    def _generate_cities(self):
        self.graph.clear()
        self.cities = []

        positions = set()
        for i in range(self.num_cities):
            for _ in range(100):
                x = self.np_random.integers(0, self.world_size)
                y = self.np_random.integers(0, self.world_size)
                if (x, y) in positions:
                    continue
                positions.add((x, y))
                self.graph.add_node(i, pos=(x, y))
                self.cities.append((x, y))
                break

        # Utworzenie krawędzi
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                city_a = self.cities[i]
                city_b = self.cities[j]
                dist = np.linalg.norm(np.array(city_a) - np.array(city_b))
                self.graph.add_edge(i, j, weight=dist)

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        super().reset(seed=seed)
        self.units = self.np_random.integers(1, 5, size=self.num_cities, dtype=np.int32)
        self.owners = np.full(self.num_cities, -1, dtype=np.int8)
        self.owners[0] = 0  # Agent posiada pierwsze miasto
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Zamiana na tuple, by mieć hashowalny klucz
        units_t = tuple(self.units.tolist())
        owners_t = tuple(self.owners.tolist())
        return {"units": units_t, "owners": owners_t}

    def get_available_actions(self, state):
        # Akcja pass jest zawsze dostępna
        available = [self.pass_action]

        units = np.array(state["units"])
        owners = np.array(state["owners"])

        # Możemy ruszać jednostkami tylko z miast, które należą do agenta i mają >1 jednostki
        for edge_idx, (from_city, to_city) in enumerate(self.edges):
            if owners[from_city] == 0 and units[from_city] > 1:
                available.append(edge_idx)

        return available

    def step(self, action: int):
        self.steps += 1

        reward = 0
        done = False

        if action == self.pass_action:
            # pass: nic nie robimy, ale jednostki rosną
            pass
        elif action >= len(self.edges):
            raise ValueError("Invalid action.")

        else:
            from_city, to_city = self.action_to_edge[action]

            # Możliwa akcja tylko jeśli agent posiada from_city i ma >1 jednostek
            if self.owners[from_city] == 0 and self.units[from_city] > 1:
                moved_units = self.units[from_city] // 2
                self.units[from_city] -= moved_units

                # Walka lub przejęcie
                if self.owners[to_city] != 0:
                    if moved_units > self.units[to_city]:
                        self.units[to_city] = moved_units - self.units[to_city]
                        self.owners[to_city] = 0
                        reward += 1  # bonus za przejęcie
                    else:
                        self.units[to_city] -= moved_units
                else:
                    self.units[to_city] += moved_units

        # Produkcja jednostek w miastach (co 5 kroków generujemy nową jednostkę)
        if self.steps % 5 == 0:
            for i in range(self.num_cities):
                if self.units[i] < self.max_units:
                    self.units[i] += 1

        obs = self._get_obs()
        if (self.owners == 0).all():
            done = True
            reward += 10

        if self.steps >= 100:
            done = True

        return obs, reward, done, False, {}

    def render(self):
        print("Miasta:")
        for i in range(self.num_cities):
            owner = "Agent" if self.owners[i] == 0 else "Neutral" if self.owners[i] == -1 else "Enemy"
            print(f"City {i}: Units = {self.units[i]}, Owner = {owner}")

    def close(self):
        pass


class RLAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        """
        Agent Q-learning do zarządzania jednostkami w środowisku miastowym.
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table: stan (serializowany tuple) -> dict(akcja -> wartość)
        self.q_table = defaultdict(lambda: defaultdict(float))

        self.episode_rewards = []
        self.average_rewards = []
        self.rng = np.random.default_rng()

    def serialize_state(self, state):
        units = tuple(state["units"])
        owners = tuple(state["owners"])
        return (units, owners)


    def choose_action(self, state):
        serialized_state = self.serialize_state(state)
        available_actions = self.env.get_available_actions(state)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)

        q_values = self.q_table[serialized_state]
        if not q_values:
            return self.rng.choice(available_actions)

        best_action = max(q_values, key=q_values.get)
        return best_action if best_action in available_actions else self.rng.choice(available_actions)


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
        total_rewards = 0

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                actions = self.env.get_available_actions(state)
                s = self.serialize_state(state)
                q_values = self.q_table[s]

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


if __name__ == "__main__":
    env = CityEnv()
    agent = RLAgent(env)

    agent.train(num_episodes=10000)
    agent.evaluate(num_episodes=1000)
    agent.plot_learning_curve()

# Średnia nagroda po 1000 epizodach ewaluacji: 0.225