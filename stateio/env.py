import sys
import gymnasium as gym
from gymnasium import spaces
import random
import math
from typing import Tuple, List

import networkx as nx
import numpy as np
import pygame

SIZE_OF_CITY = 20

# Kolory
BACKGROUND_COLOR = (240, 240, 240)
GRID_COLOR = (200, 200, 200)


PLAYERS_COLORS = [
    (255, 0, 0),    # Czerwony
    (0, 0, 255),    # Niebieski
    (0, 128, 0),    # Zielony
    (255, 165, 0),  # Pomarańczowy
    (128, 0, 128),  # Fioletowy
    (165, 42, 42),  # Brązowy
    (0, 128, 128),  # Morski
    (255, 105, 180) # Różowy
]

class City:
    def __init__(self, x: int, y: int, id: int, max_units: int = 50, max_neutral_units: int = 10):
        """
        Inicjalizacja miasta.

        Args:
            x: Współrzędna X na planszy
            y: Współrzędna Y na planszy
        """
        self.x = x
        self.y = y
        self.warriors = 5
        self.max_units = max_units
        self.max_neutral_units = max_neutral_units
        self.player: Player = None
        self.id = id

    def step(self):
        limit = self.max_neutral_units if self.player is None else self.max_units
        if self.warriors < limit:
            self.warriors+=1

    # Interface
    def draw(self, screen, font, world_to_screen_fn):
        screen_x, screen_y = world_to_screen_fn(self.x, self.y)

        color = (255, 255, 255)
        if self.player is not None:
            color = PLAYERS_COLORS[self.player.env.players.index(self.player) % len(PLAYERS_COLORS)]

        pygame.draw.circle(screen, color, (screen_x, screen_y), SIZE_OF_CITY)
        pygame.draw.circle(screen, (0, 0, 0), (screen_x, screen_y), SIZE_OF_CITY, 2)

        if font:
            text = font.render(f"City {self.id}", True, (0, 0, 0))
            rect = text.get_rect(center=(screen_x, screen_y - SIZE_OF_CITY - 10))
            screen.blit(text, rect)

            # Liczba wojowników wewnątrz kółka
            warrior_text = font.render(str(self.warriors), True, (0, 0, 0))
            warrior_rect = warrior_text.get_rect(center=(screen_x, screen_y))
            screen.blit(warrior_text, warrior_rect)


    def is_clicked(self, mouse_pos: Tuple[int, int], world_to_screen_fn) -> bool:
        """
        Sprawdza, czy miasto zostało kliknięte.

        Args:
            mouse_pos: Pozycja kliknięcia myszy (x, y)

        Returns:
            bool: True, jeśli kliknięcie było wewnątrz miasta
        """
        screen_x, screen_y = world_to_screen_fn(self.x, self.y)
        distance = math.sqrt((mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2)
        return distance <= SIZE_OF_CITY


class GridGame(gym.Env):
    """Główna klasa gry z planszą NxN i miastami."""
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 grid_size: int = 10,
                 world_size: int = 10,
                 screen_size: int = 1000,
                 num_cities: int = 5,
                 num_players: int = 2,
                 max_neutral_warrior: int = 10,
                 seed: int = 100,
                 max_units: int = 50,
                 render_mode: str = None
                 ):

        self.players: List[Player] = [Player(env=self, is_bot=True, id=id+1) for id in range(num_players)]
        self.players[0].is_bot=False

        self.seed = seed # For restart Game.

        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)

        self.world_size = world_size

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._init_for_pygame(grid_size, screen_size, num_cities, max_neutral_warrior)

        # Initialize graph
        self.graph = nx.Graph()
        # Generowanie miast
        self._generate_cities()

        self.edges = list(self.graph.edges)

        self.edge_to_action = {}
        self.action_to_edge = {}
        action_idx = 0
        for edge in self.edges:
            a, b = edge[0].id, edge[1].id
            self.edge_to_action[(a, b)] = action_idx
            self.action_to_edge[action_idx] = (a, b)
            action_idx += 1
            self.edge_to_action[(b, a)] = action_idx
            self.action_to_edge[action_idx] = (b, a)
            action_idx += 1
        self.pass_action = action_idx
        self.action_space = spaces.Discrete(self.pass_action + 1)

        # Obserwacja: liczba jednostek i właściciele
        self.observation_space = spaces.Dict({
            "units": spaces.Box(low=0, high=max_units, shape=(num_cities,), dtype=np.int32),
            "owners": spaces.Box(low=-1, high=1, shape=(num_cities,), dtype=np.int8),  # -1 brak właściciela, 0 agent
        })

        self.current_player_idx = 0
        self.steps = 0

        # Wybrane miasto
        self.selected_city = None

    def _init_for_pygame(self,
                         grid_size: int = 10,
                         screen_size: int = 200,
                         num_cities: int = 5,
                         max_neutral_warrior: int = 10,
                         max_units: int = 50):
        # Inicjalizacja PyGame
        pygame.init()

        # Ustawienia gry
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.cell_size = screen_size // grid_size
        self.num_cities = min(num_cities, grid_size * grid_size)
        self.max_neutral_warrior = max_neutral_warrior
        self.max_units = max_units

        # Tworzenie okna gry
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption(f"Plansza {grid_size}x{grid_size} z miastami")

            # Czcionki
            self.font = pygame.font.SysFont('Arial', 14)
            self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

    def _generate_cities(self):
        """Generuje losowe miasta na planszy."""

        positions = set()
        for i in range(self.num_cities):
            for _ in range(100):
                x = self.py_rng.randint(0, self.world_size)
                y = self.py_rng.randint(0, self.world_size)
                if (x, y) in positions:
                    continue
                positions.add((x, y))
                city = City(x, y, i+1, max_units=self.max_units, max_neutral_units=self.max_neutral_warrior)
                self.graph.add_node(city)
                break

        # Tworzymy pełny graf – krawędzie między wszystkimi miastami
        cities = list(self.graph.nodes)
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                city_a = cities[i]
                city_b = cities[j]
                dx = city_a.x - city_b.x
                dy = city_a.y - city_b.y
                distance = round((dx ** 2 + dy ** 2) ** 0.5, 1)  # Euklidesowy dystans
                self.graph.add_edge(city_a, city_b, weight=distance)
        
        for player in self.players:
            # Przydzielanie graczy do miast
            if self.graph.nodes:
                city = self.py_rng.choice(list(self.graph.nodes))
                while city.player is not None:
                    city = self.py_rng.choice(list(self.graph.nodes))
                city.player = player
                city.warriors = 10 
                player.env = self



##################################################################
##################################################################
#              RYSOWANIE DLA CELÓW WIZUALIZACYJNYCH              #
##################################################################
##################################################################
    def world_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Skaluje pozycję świata do rozmiaru ekranu."""
        padding = SIZE_OF_CITY + 5  # padding w pikselach
        effective_screen = self.screen_size - 2 * padding
        sx = int(padding + (x / self.world_size) * effective_screen)
        sy = int(padding + (y / self.world_size) * effective_screen)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> Tuple[int, int]:
        """Przekształca pozycję ekranu na współrzędne świata z uwzględnieniem paddingu."""
        padding = SIZE_OF_CITY + 5  # taki sam jak w world_to_screen
        effective_screen = self.screen_size - 2 * padding
        wx = int(((sx - padding) / effective_screen) * self.world_size)
        wy = int(((sy - padding) / effective_screen) * self.world_size)
        return wx, wy

    def _draw_edges(self):

        for city_a, city_b, data in self.graph.edges(data=True):
            ax, ay = self.world_to_screen(city_a.x, city_a.y)
            bx, by = self.world_to_screen(city_b.x, city_b.y)

            pygame.draw.line(self.screen, (100, 100, 100), (ax, ay), (bx, by), 1)

            mx, my = (ax + bx) // 2, (ay + by) // 2
            #text = self.font.render(f"{data['weight']}", True, (0, 0, 0))
            #rect = text.get_rect(center=(mx, my))
            #self.screen.blit(text, rect)

    def _draw_game(self):
        """Rysuje siatkę planszy."""

        # Czyszczenie ekranu
        self.screen.fill(BACKGROUND_COLOR)

        for i in range(self.grid_size + 1):
            # Poziome linie
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (0, i * self.cell_size),
                (self.screen_size, i * self.cell_size),
                1
            )

            # Pionowe linie
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.screen_size),
                1
            )

        self._draw_edges()

        # Rysowanie miast
        for city in self.graph.nodes:
            # Podświetl wybrane miasto
            if city == self.selected_city:
                screen_x, screen_y = self.world_to_screen(city.x, city.y)
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 0),
                    (screen_x, screen_y),
                    SIZE_OF_CITY + 5,
                    3
                )

            city.draw(self.screen, self.font, self.world_to_screen)

        # Wyświetlanie informacji o wybranym mieście
        self._draw_city_info()

        # Aktualizacja ekranu
        pygame.display.flip()

    def _draw_city_info(self):
        """Wyświetla informacje o wybranym mieście."""
        if self.selected_city:
            # Przygotuj informacje o mieście
            info_text = [
                f"Pozycja: ({self.selected_city.x}, {self.selected_city.y})",
                f"Player: {self.selected_city.player.id if self.selected_city.player else 'Neutral'}",
                f"Warriors: {self.selected_city.warriors}",
            ]

            # Rysuj panel informacyjny
            info_surface = pygame.Surface((250, 100))
            info_surface.fill((255, 255, 255))
            pygame.draw.rect(info_surface, (0, 0, 0), info_surface.get_rect(), 2)

            # Tytuł
            title = self.title_font.render("Informacje o mieście", True, (0, 0, 0))
            info_surface.blit(title, (10, 10))

            # Informacje
            for i, text in enumerate(info_text):
                info_line = self.font.render(text, True, (0, 0, 0))
                info_surface.blit(info_line, (20, 40 + i * 20))

            # Wyświetl panel w prawym górnym rogu
            self.screen.blit(info_surface, (self.screen_size - 260, 10))

    def _get_obs(self):
        # Zamiana na tuple, by mieć hashowalny klucz
        units_t = tuple(city.warriors for city in self.graph.nodes)
        owners_t = tuple(
            (self.players.index(city.player) if city.player else -1) for city in self.graph.nodes
        )
        return {"units": units_t, "owners": owners_t}

    def step(self, action: int):
        self.steps += 1

        reward = -0.1
        done = False
        cities = list(self.graph.nodes)

        if action == self.pass_action:
            # pass: nic nie robimy, ale jednostki rosną
            # print("pass")
            pass
        elif action > self.pass_action:
            raise ValueError("Invalid action.")

        else:
            from_id, to_id = self.action_to_edge[action]
            from_city = next(city for city in cities if city.id == from_id)
            to_city = next(city for city in cities if city.id == to_id)
            current_player = self.players[self.current_player_idx]
            # print(f"Ruch z {from_id} do {to_id}, gracz {current_player.id}")

            # Możliwa akcja tylko jeśli agent posiada from_city i ma >1 jednostek
            if from_city.player != current_player and from_city.warriors > 1:
                return self._get_obs(), reward, False, False, {"reason": "Invalid action"}
            moved_units = from_city.warriors
            from_city.warriors -= moved_units

            # Walka lub przejęcie
            if moved_units > to_city.warriors:
                to_city.warriors = moved_units - to_city.warriors
                to_city.player = current_player
                reward += 1  # bonus za przejęcie
            else:
                to_city.warriors = to_city.warriors - moved_units
                reward -= 0.5  # kara za przegraną walkę

        obs = self._get_obs()
        # Sprawdź, czy wszystkie miasta są bez właściciela lub należą do bieżącego gracza
        if all(city.player is None or city.player == self.players[self.current_player_idx] for city in cities):
            done = True
            reward += 100

        if self.steps >= 100:
            done = True

        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

        if self.current_player_idx == 0:
            for city in cities:
                city.step()
            
        if self.render_mode is not None:
            self.render()

        # Jeśli gracz jest botem, wykonaj jego ruch
        while not done and self.players[self.current_player_idx].is_bot:
            bot_player = self.players[self.current_player_idx]
            bot_action = bot_player.bot_move(obs)
            obs, bot_reward, done, _, info = self.step(bot_action)
            reward -= bot_reward + 0.1

        return obs, reward, done, False, {}

    def get_available_actions(self, state):
        # Akcja pass jest zawsze dostępna
        available = [self.pass_action]
        current_player = self.players[self.current_player_idx]

        # Możemy ruszać jednostkami tylko z miast, które należą do agenta i mają >1 jednostki
        for action_idx, (from_city_id, to_city_id) in self.action_to_edge.items():
            from_city = next(city for city in self.graph.nodes if city.id == from_city_id)
            if from_city.player == current_player and from_city.warriors > 1:
                available.append(action_idx)

        return available

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)

        # Initialize graph
        self.graph = nx.Graph()
        # Generowanie miast
        self._generate_cities()
        # Wybrane miasto
        self.selected_city = None

        self.current_player_idx = 0
        self.steps = 0

        return self._get_obs(), {}
    
    def render(self):
        """Renderuje grę."""
        if self.render_mode == "human":
            self._draw_game()



class Player:
    def __init__(self, env: GridGame, is_bot: bool = False, is_ai: bool = False, id: int = None):
        self.id = id if id is not None else random.randint(0, 1000)
        self.is_bot = is_bot
        self.env = env

    def bot_move(self, state):
        for action in self.env.get_available_actions(state):
            if action == self.env.pass_action:
                continue
            from_city_id, to_city_id = self.env.action_to_edge[action]
            from_city = next(city for city in self.env.graph.nodes if city.id == from_city_id)
            to_city = next(city for city in self.env.graph.nodes if city.id == to_city_id)
            if from_city.warriors > to_city.warriors and to_city.player != self:
                return action
        return self.env.pass_action

    def choose_action(self, state):
        # if(self.is_bot):
        #     # print("Bot")
        #     return self.bot_move(state)

        waiting = True
        clock = pygame.time.Clock()

        while waiting:
            if self.env.render_mode == "human":
                self.env._draw_game()
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()

                    for city in self.env.graph.nodes:
                        if city.is_clicked(mouse_pos, self.env.world_to_screen):
                            if self.env.selected_city == city:
                                self.env.selected_city = None
                            elif self.env.selected_city is not None:
                                return self.env.edge_to_action[(self.env.selected_city.id, city.id)]
                            else:
                                self.env.selected_city = city
                            break
                    else:
                        self.env.selected_city = None

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                        return self.env.pass_action
                    elif event.key == pygame.K_BACKSPACE:
                        self.env.reset()

            clock.tick(60)