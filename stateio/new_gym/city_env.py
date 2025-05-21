import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Any
import random

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

        # Akcje: (miasto startowe, miasto docelowe)
        self.action_space = spaces.Discrete(len(self.edges))
        self.edge_to_action = {edge: idx for idx, edge in enumerate(self.edges)}
        self.action_to_edge = {v: k for k, v in self.edge_to_action.items()}

        # Obserwacja: liczba jednostek w miastach + właściciel
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

        self.edges = list(self.graph.edges)

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        super().reset(seed=seed)
        self.units = self.np_random.integers(1, 5, size=self.num_cities, dtype=np.int32)
        self.owners = np.full(self.num_cities, -1, dtype=np.int8)
        self.owners[0] = 0  # Agent posiada pierwsze miasto
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "units": self.units.copy(),
            "owners": self.owners.copy()
        }

    def step(self, action: int):
        self.steps += 1

        if action >= len(self.edges):
            raise ValueError("Invalid action.")

        from_city, to_city = self.action_to_edge[action]

        reward = 0
        done = False

        # Możliwa akcja tylko jeśli agent posiada from_city
        if self.owners[from_city] == 0 and self.units[from_city] > 1:
            moved_units = self.units[from_city] // 2
            self.units[from_city] -= moved_units

            # Jeśli to_city jest wrogie lub neutralne — walka lub przejęcie
            if self.owners[to_city] != 0:
                if moved_units > self.units[to_city]:
                    self.units[to_city] = moved_units - self.units[to_city]
                    self.owners[to_city] = 0
                    reward += 1  # Bonus za przejęcie miasta
                else:
                    self.units[to_city] -= moved_units
            else:
                self.units[to_city] += moved_units

        # Produkcja jednostek w miastach
        for i in range(self.num_cities):
            if self.units[i] < self.max_units:
                self.units[i] += 1

        obs = self._get_obs()
        if (self.owners == 0).all():  # Agent zdobył wszystkie miasta
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
