import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Tuple, Any
import pygame
import random
from enum import Enum
import sys

from enums import ResourceType, ActionType, TileType, BuildingType, UnitType
from main import Player, City, Unit


class MicroCivEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, map_size=(15, 15), num_ai_players=3):
        self.map_size = map_size
        self.num_ai_players = num_ai_players
        self.map = None
        self.players = []
        self.current_player_idx = 0
        self.turn_number = 1
        self.max_turns = 100
        self.game_over = False
        self.selected_unit_idx = None
        self.selected_city_idx = None

        # Koszty zasobów dla różnych akcji
        self.costs = {
            ActionType.BUILD_CITY: {ResourceType.WOOD: 30, ResourceType.STONE: 20},
            ActionType.BUILD_FARM: {ResourceType.WOOD: 10},
            ActionType.BUILD_MINE: {ResourceType.WOOD: 15, ResourceType.STONE: 5},
            ActionType.BUILD_LUMBERMILL: {ResourceType.WOOD: 10},
            ActionType.CREATE_SETTLER: {ResourceType.FOOD: 30, ResourceType.GOLD: 10},
            ActionType.CREATE_WARRIOR: {ResourceType.FOOD: 20, ResourceType.WOOD: 10, ResourceType.GOLD: 5},
            ActionType.CREATE_WORKER: {ResourceType.FOOD: 20, ResourceType.GOLD: 5},
            ActionType.BUILD_HOUSE: {ResourceType.WOOD: 10, ResourceType.STONE: 5},
            ActionType.BUILD_GRANARY: {ResourceType.WOOD: 15, ResourceType.STONE: 10},
            ActionType.BUILD_BARRACKS: {ResourceType.WOOD: 20, ResourceType.STONE: 15},
            ActionType.BUILD_MARKET: {ResourceType.WOOD: 15, ResourceType.STONE: 10, ResourceType.GOLD: 10},
        }

        # Produkcja zasobów z różnych typów kafelków
        self.tile_production = {
            TileType.GRASS: {ResourceType.FOOD: 1},
            TileType.FOREST: {ResourceType.FOOD: 1, ResourceType.WOOD: 2},
            TileType.MOUNTAIN: {ResourceType.STONE: 2},
            TileType.WATER: {},
            TileType.CITY: {ResourceType.GOLD: 1},
            TileType.FARM: {ResourceType.FOOD: 3},
            TileType.MINE: {ResourceType.STONE: 3, ResourceType.GOLD: 1},
            TileType.LUMBERMILL: {ResourceType.WOOD: 3},
        }

        # Produkcja zasobów z budynków
        self.building_production = {
            BuildingType.HOUSE: {ResourceType.FOOD: 1},
            BuildingType.GRANARY: {ResourceType.FOOD: 3},
            BuildingType.BARRACKS: {}, # Barracks don't produce resources directly
            BuildingType.MARKET: {ResourceType.GOLD: 3},
        }

        # Przestrzeń obserwacji
        # Obserwacja będzie składać się z:
        # 1. Mapy (kafelki)
        # 2. Pozycji jednostek
        # 3. Pozycji miast
        # 4. Zasobów gracza
        # 5. Innych informacji o stanie gry

        # Mapa będzie reprezentowana jako tablica 3D: (wysokość, szerokość, kanały)
        # Kanały: [typ kafelka, jednostka gracz_1, jednostka gracz_2, ..., miasto gracz_1, miasto gracz_2, ...]
        map_channels = 1 + (num_ai_players + 1) * 2  # 1 dla mapy + 2 na gracza (jednostki i miasta)

        # Przestrzeń obserwacji obejmuje mapę oraz wektor cech dla aktualnego gracza
        # Wektor cech: [zasoby (4), liczba jednostek (1), liczba miast (1), poziom technologii (1), numer tury (1)]
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=255, shape=(map_size[0], map_size[1], map_channels), dtype=np.uint8),
            "player_features": spaces.Box(low=0, high=1000, shape=(8,), dtype=np.int32),
            "valid_actions": spaces.Box(low=0, high=1, shape=(len(ActionType),), dtype=np.int8)
        })

        # Przestrzeń akcji
        # Akcje: ruch w 4 kierunki, budowa 4 typów struktur, tworzenie 3 typów jednostek,
        # budowa 4 typów budynków, koniec tury, nic nie rób
        self.action_space = spaces.Discrete(len(ActionType))

        # Ustawienie trybu renderowania
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Pygame initialization
        self.window = None
        self.clock = None
        self.cell_size = 40
        self.font = None
        self.selected_unit = None
        self.selected_city = None

        self.colors = {
            TileType.GRASS: (124, 252, 0),
            TileType.FOREST: (34, 139, 34),
            TileType.MOUNTAIN: (139, 137, 137),
            TileType.WATER: (0, 191, 255),
            TileType.CITY: (255, 215, 0),
            TileType.FARM: (255, 255, 0),
            TileType.MINE: (169, 169, 169),
            TileType.LUMBERMILL: (160, 82, 45),
            TileType.UNEXPLORED: (0, 0, 0),
        }

        self.player_colors = [
            (255, 0, 0),    # Czerwony
            (0, 0, 255),    # Niebieski
            (0, 255, 0),    # Zielony
            (255, 165, 0),  # Pomarańczowy
        ]

    def _generate_map(self):
        # Inicjalizacja mapy z losowymi kafelkami
        self.map = np.full(self.map_size, TileType.GRASS.value, dtype=np.int8)

        for y in range(self.map_size[0]):
            for x in range(self.map_size[1]):
                if random.random() < 0.25:
                    self.map[y, x] = TileType.FOREST.value
                elif random.random() < 0.15:
                    self.map[y, x] = TileType.MOUNTAIN.value
                elif random.random() < 0.10:
                    self.map[y, x] = TileType.WATER.value

        self._create_clusters(TileType.FOREST)
        self._create_clusters(TileType.MOUNTAIN)
        self._create_clusters(TileType.WATER)
        # # Początkowo wszystkie kafelki są niewidoczne dla gracza
        # self.fog_of_war = np.full(self.map_size, True, dtype=bool)
        #
        # return self.map



    def _create_clusters(self, tile_type, num_clusters=5, cluster_size=3):
        height, width = self.map_size

        for _ in range(num_clusters):
            # Losowy punkt początkowy klastra
            center_y = random.randint(0, height - 1)
            center_x = random.randint(0, width - 1)

            # Tworzenie klastra
            for _ in range(cluster_size):
                # Losowy punkt w pobliżu centrum
                offset_y = random.randint(-2, 2)
                offset_x = random.randint(-2, 2)

                y = max(0, min(height - 1, center_y + offset_y))
                x = max(0, min(width - 1, center_x + offset_x))

                if random.random() < 0.7:  # 70% szans na umieszczenie kafelka
                    self.map[y, x] = tile_type.value

    def _initialize_players(self):
        """
        Tworzenie głównego gracza oraz przeciwników AI
        :return:
        """
        self.players = [Player(player_id=0, is_ai=False)]
        for i in range(1, self.num_ai_players + 1):
            self.players.append(Player(player_id=i, is_ai=True))

        # Umieszczanie początkowych jednostek i miast
        self._place_starting_units()


    def _place_starting_units(self):
        """
        Umieszczanie początkowych jednostek i miast dla każdego gracza.
        Jeśli jest więcej graczy, to losujemy dla nich pozycję.
        Nie może być w wodzie, i czy jest wystarczająco daleko od innych graczy.
        :return:
        """
        height, width = self.map_size

        start_positions = [
            (2, 2),
            (width - 3, height - 3),
            (2, height - 3),
            (width - 3, 2)
        ]

        for player_idx, player in enumerate(self.players):
            # Pobieramy pozycję dla tego gracza
            if player_idx < len(start_positions):
                pos_x, pos_y = start_positions[player_idx]
            else:
                # Jeśli mamy więcej graczy niż pozycji, losujemy pozycję
                while True:
                    pos_x = random.randint(2, width - 3)
                    pos_y = random.randint(2, height - 3)

                    # Sprawdzamy, czy pozycja jest odpowiednia (np. nie w wodzie)
                    if self.map[pos_y, pos_x] != TileType.WATER.value:
                        # Sprawdzamy, czy jest wystarczająco daleko od innych graczy
                        is_valid = True
                        for other_player in self.players[:player_idx]:
                            for city in other_player.cities:
                                dist = abs(city.x - pos_x) + abs(city.y - pos_y)
                                if dist < 5: # Minimalna odległość między miastami
                                    is_valid = False
                                    break
                        if is_valid:
                            break

            # Unikamy umieszczania miast na wodzie
            if self.map[pos_y, pos_x] == TileType.WATER.value:
                # Szukamy najbliższego kafelka, który nie jest wodą
                for search_dist in range(1, 5):
                    found = False
                    for dy in range(-search_dist, search_dist + 1):
                        for dx in range(-search_dist, search_dist + 1):
                            check_y = pos_y + dy
                            check_x = pos_x + dx
                            if (0 <= check_y < height and 0 <= check_x < width and
                                    self.map[check_y, check_x] != TileType.WATER.value):
                                pos_y, pos_x = check_y, check_x
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            # Tworzymy miasto
            city = City(x=pos_x, y=pos_y, player_id=player_idx, name=f"City {player_idx}")
            player.add_city(city)
            self.map[pos_y, pos_x] = TileType.CITY.value

            # Tworzymy początkowe jednostki
            settler = Unit(unit_type=UnitType.SETTLER, x=pos_x, y=pos_y-1, player_id=player_idx)
            warrior = Unit(unit_type=UnitType.WARRIOR, x=pos_x+1, y=pos_y, player_id=player_idx)

            # Upewniamy się, że jednostki nie wylądują na wodzie
            if pos_y-1 >= 0 and self.map[pos_y-1, pos_x] == TileType.WATER.value:
                self.map[pos_y-1, pos_x] = TileType.GRASS.value
            if pos_x+1 < width and self.map[pos_y, pos_x+1] == TileType.WATER.value:
                self.map[pos_y, pos_x+1] = TileType.GRASS.value

            player.add_unit(settler)
            player.add_unit(warrior)

            # Odkrywamy obszar wokół startowej pozycji gracza
            self._update_visibility(player, pos_x, pos_y, 3)

    def _update_visibility(self, player, center_x, center_y, radius=2):
        height, width = self.map_size
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                # Obliczamy odległość Manhattan
                dist = abs(x - center_x) + abs(y - center_y)
                if dist <= radius:
                    # Dodajemy kafelek do odkrytych
                    player.explored_tiles.add((x, y))
                    if dist <= radius - 1:
                        # Kafelki bliżej niż radius są widoczne
                        player.visible_tiles.add((x, y))


    def _get_observation(self):
        # Pobieramy aktualnego gracza
        current_player = self.players[self.current_player_idx]

        # Przygotowujemy obserwację mapy
        height, width = self.map_size
        map_channels = 1 + (len(self.players)) * 2 # 1 dla mapy + 2 na gracza (jednostki i miasta)
        obs_map = np.zeros((height, width, map_channels), dtype=np.uint8)

        # Pierwszy kanał to mapa (widoczna dla aktualnego gracza)
        # Nieodkryte kafelki mają wartość TileType.UNEXPLORED.value
        base_map = np.full(self.map_size, TileType.UNEXPLORED.value, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                if (x, y) in current_player.explored_tiles:
                    base_map[y, x] = self.map[y, x]

        obs_map[:, :, 0] = base_map

        # Kolejne kanały to jednostki i miasta każdego gracza
        for player_idx, player in enumerate(self.players):
            # Kanał dla jednostek gracza
            unit_channel = 1 + player_idx * 2
            # Kanał dla miast gracza
            city_channel = 2 + player_idx * 2

            # Umieszczamy jednostki i miasta na mapie, jeśli są widoczne dla aktualnego gracza
            for unit in player.units:
                if (unit.x, unit.y) in current_player.visible_tiles:
                    obs_map[unit.y, unit.x, unit_channel] = unit.unit_type.value + 1
            for city in player.cities:
                if (city.x, city.y) in current_player.visible_tiles:
                    obs_map[city.y, city.x, city_channel] = 1

        # Przygotowujemy wektor cech gracza
        player_features = np.zeros(8, dtype=np.int32)

        # Zasoby (4)
        player_features[0] = current_player.resources[ResourceType.FOOD]
        player_features[1] = current_player.resources[ResourceType.WOOD]
        player_features[2] = current_player.resources[ResourceType.STONE]
        player_features[3] = current_player.resources[ResourceType.GOLD]
        #TODO być może dodawać liczbę określonych jednostek a nie ogólną ilość
        # Liczba jednostek (1)
        player_features[4] = len(current_player.units)
        # Liczba miast (1)
        player_features[5] = len(current_player.cities)
        # Poziom technologii (1)
        player_features[6] = current_player.tech_level
        # Numer tury (1)
        player_features[7] = self.turn_number

        # Maska dozwolonych akcji
        valid_actions = self._get_valid_actions()

        return {
            "map": obs_map,
            "player_features": player_features,
            "valid_actions": valid_actions
        }

    def _get_valid_actions(self):
        # Przygotowujemy wektor dozwolonych akcji
        valid_actions = np.zeros(len(ActionType), dtype=np.int8)

        # Końca tury i bezczynności są zawsze dozwolone
        valid_actions[ActionType.END_TURN.value] = 1
        valid_actions[ActionType.DO_NOTHING.value] = 1

        # Pobieramy aktualnego gracza
        current_player = self.players[self.current_player_idx]

        # Sprawdzamy, czy mamy wybraną jednostkę
        if self.selected_unit_idx is not None and 0 <= self.selected_unit_idx < len(current_player.units):
            unit = current_player.units[self.selected_unit_idx]

            # Jeśli jednostka może się poruszać, dodajemy akcje ruchu
            if unit.movement_left > 0:
                # Sprawdzamy możliwe kierunki ruchu
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # góra, dół, lewo, prawo
                action_types = [ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]

                for (dx, dy), action_type in zip(directions, action_types):
                    new_x, new_y = unit.x + dx, unit.y + dy

                    # Sprawdzamy, czy nowa pozycja jest na mapie
                    if 0 <= new_x < self.map_size[1] and 0 <= new_y < self.map_size[0]:
                        # Sprawdzamy, czy kafelek nie jest wodą
                        if self.map[new_y, new_x] != TileType.WATER.value:
                            # Sprawdzamy, czy nie ma tam innej jednostki lub miasta innego gracza
                            is_valid = True


                            for player in self.players:
                                # Sprawdzamy jednostki
                                for other_unit in player.units:
                                    if other_unit.x == new_x and other_unit.y == new_y:
                                        # Jeśli to jednostka przeciwnika, można przejść tylko wojownikiem
                                        if player.player_id != current_player.player_id:
                                            if unit.unit_type != UnitType.WARRIOR:
                                                is_valid = False
                                        else:
                                            # Jeśli to własna jednostka, nie można się tam ruszyć
                                            is_valid = False

                                # Sprawdzamy miasta
                                for city in player.cities:
                                    if city.x == new_x and city.y == new_y and player.player_id != current_player.player_id:
                                        is_valid = False

                            if is_valid:
                                valid_actions[action_type.value] = 1

            # Jeśli jednostka nie wykonała jeszcze akcji, dodajemy odpowiednie akcje
            if not unit.has_acted:
                # Dla osadnika - budowa miasta
                if unit.unit_type == UnitType.SETTLER:
                    # Sprawdzamy, czy możemy zbudować miasto na aktualnej pozycji
                    can_build_city = True

                    # Nie możemy budować na wodzie #TODO to można byłoby zdjąć np dla określonych klas / ras
                    if self.map[unit.y, unit.x] == TileType.WATER.value:
                        can_build_city = False


                    # Sprawdzamy, czy nie ma już miasta w pobliżu
                    for player in self.players:
                        for city in player.cities:
                            dist = abs(city.x - unit.x) + abs(city.y - unit.y)
                            if dist < 3: # Minimalna odległość między miastami
                                can_build_city = False
                                break

                    # Sprawdzamy, czy mamy wystarczające zasoby
                    if not current_player.can_afford(self.costs[ActionType.BUILD_CITY]):
                        can_build_city = False
                    if can_build_city:
                        valid_actions[ActionType.BUILD_CITY.value] = 1

                # Dla robotnika - budowa struktur
                elif unit.unit_type == UnitType.WORKER:
                    # Sprawdzamy, czy aktualny kafelek to trawa lub las (można budować farmy, kopalnie, tartaki)
                    tile_type = self.map[unit.y, unit.x]

                    # Na trawie można budować farmy
                    if tile_type == TileType.GRASS.value and current_player.can_afford(self.costs[ActionType.BUILD_FARM]):
                        valid_actions[ActionType.BUILD_FARM.value] = 1
                    # W lesie można budować tartaki
                    elif tile_type == TileType.FOREST.value and current_player.can_afford(self.costs[ActionType.BUILD_LUMBERMILL]):
                        valid_actions[ActionType.BUILD_LUMBERMILL.value] = 1
                    # W górach można budować kopalnie
                    elif tile_type == TileType.MOUNTAIN.value and current_player.can_afford(self.costs[ActionType.BUILD_MINE]):
                        valid_actions[ActionType.BUILD_MINE.value] = 1

        # Sprawdzamy, czy mamy wybrane miasto
        if self.selected_city_idx is not None and 0 <= self.selected_city_idx < len(current_player.cities):
            city = current_player.cities[self.selected_city_idx]

            # Jeśli miasto nie wykonało jeszcze akcji, dodajemy akcje tworzenia jednostek i budowy budynków
            if not city.has_acted:
                # Tworzenie jednostek
                if current_player.can_afford(self.costs[ActionType.CREATE_SETTLER]):
                    valid_actions[ActionType.CREATE_SETTLER.value] = 1
                if current_player.can_afford(self.costs[ActionType.CREATE_WARRIOR]):
                    valid_actions[ActionType.CREATE_WARRIOR.value] = 1
                if current_player.can_afford(self.costs[ActionType.CREATE_WORKER]):
                    valid_actions[ActionType.CREATE_WORKER.value] = 1

                # Budowa budynków
                if (BuildingType.HOUSE not in city.buildings and
                        current_player.can_afford(self.costs[ActionType.BUILD_HOUSE])):
                    valid_actions[ActionType.BUILD_HOUSE.value] = 1
                if (BuildingType.GRANARY not in city.buildings and
                        current_player.can_afford(self.costs[ActionType.BUILD_GRANARY])):
                    valid_actions[ActionType.BUILD_GRANARY.value] = 1
                if (BuildingType.BARRACKS not in city.buildings and
                        current_player.can_afford(self.costs[ActionType.BUILD_BARRACKS])):
                    valid_actions[ActionType.BUILD_BARRACKS.value] = 1
                if (BuildingType.MARKET not in city.buildings and
                        current_player.can_afford(self.costs[ActionType.BUILD_MARKET])):
                    valid_actions[ActionType.BUILD_MARKET.value] = 1

        return valid_actions

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new game."""
        # Inicjalizacja generatora liczb losowych
        super().reset(seed=seed)
        self._generate_map()
        self._initialize_players()

        # Resetowanie zmiennych gry
        self.current_player_idx = 0
        self.turn_number = 1
        self.game_over = False
        self.selected_unit_idx = None
        self.selected_city_idx = None

        # Aktualizacja widoczności dla gracza
        current_player = self.players[self.current_player_idx]
        for unit in current_player.units:
            self._update_visibility(current_player, unit.x, unit.y)
        for city in current_player.cities:
            self._update_visibility(current_player, city.x, city.y)

        # Inicjalizacja pygame do renderowania
        if self.render_mode is not None:
            self._render_init()

        return self._get_observation(), {}


    def step(self, action):
        """Execute one time step in the environment."""
        if self.game_over:
            return self._get_observation(), 0, True, False, {"info": "Game over"}

        # Pobieramy aktualnego gracza
        current_player = self.players[self.current_player_idx]

        # Wykonujemy akcję dla aktualnego gracza (jeśli jest to gracz AI, akcja jest ignorowana)
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Jeśli to człowiek, wykonujemy wybraną akcję
        if not current_player.is_ai:
            # Sprawdzamy, czy akcja jest dozwolona
            valid_actions = self._get_valid_actions()

            if valid_actions[action] == 1:
                # Wykonujemy akcję
                reward = self._execute_action(action)
                info["action_executed"] = True
            else:
                # Akcja nie jest dozwolona
                reward = -1 # Kara za próbę wykonania niedozwolonej akcji
                info["action_executed"] = False
                info["message"] = f"Action {ActionType(action).name} is not valid"
        else:
            # Dla AI wykonujemy prostą strategię
            self._execute_ai_turn(current_player)
            info["action_executed"] = True

        # Sprawdzamy, czy gra się skończyła
        if self.turn_number > self.max_turns:
            terminated = True
            self.game_over = True
            info["reason"] = "Max turns reached"

        # Sprawdzamy warunki zwycięstwa
        winner = self._check_victory()
        if winner is not None:
            terminated = True
            self.game_over = True
            info["winner"] = winner
            info["reason"] = "Victory condition met"
            reward += 100 if winner == 0 else -50 # Przyznajemy nagrodę za zwycięstwo lub karę za przegraną

        # Renderujemy grę, jeśli trzeba
        if self.render_mode is not None:
            self.render()

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_action(self, action):
        """Wykonuje akcję w grze i zwraca nagrodę."""
        action_type = ActionType(action)
        current_player = self.players[self.current_player_idx]
        reward = 0

        # Obsługa różnych typów akcji
        if action_type == ActionType.END_TURN:
            # Kończymy turę aktualnego gracza
            self._end_turn()
            reward = 0.1 # Mała nagroda za progres w grze
        elif action_type == ActionType.DO_NOTHING:
            reward = 0 # Nic nie robimy
        elif action_type in [ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]:
            # Ruch jednostki
            if self.selected_unit_idx is not None:
                unit = current_player.units[self.selected_unit_idx]
                # Kierunek ruchu
                dx, dy = 0, 0
                if action_type == ActionType.MOVE_UP:
                    dy = -1
                elif action_type == ActionType.MOVE_DOWN:
                    dy = 1
                elif action_type == ActionType.MOVE_LEFT:
                    dx = -1
                elif action_type == ActionType.MOVE_RIGHT:
                    dx = 1

                # Nowa pozycja
                new_x, new_y = unit.x + dx, unit.y + dy
                # Sprawdzamy, czy na nowej pozycji jest jednostka przeciwnika
                for player in self.players:
                    if player.player_id != current_player.player_id:
                        for i, enemy_unit in enumerate(player.units):
                            if enemy_unit.x == new_x and enemy_unit.y == new_y:
                                # Walka! (tylko wojownik może atakować)
                                if unit.unit_type == UnitType.WARRIOR:
                                    #TODO zmienić, aby nie było tak prosto, jakieś hp itp itd?
                                    # Usuwamy jednostkę przeciwnika
                                    player.units.pop(i)
                                    reward = 5 # Nagroda za pokonanie przeciwnika
                                break

                # Aktualizujemy pozycję jednostki
                unit.x, unit.y = new_x, new_y
                unit.movement_left -= 1

                # Aktualizujemy widoczność
                self._update_visibility(current_player, new_x, new_y)
                reward += 0.1 # Odkrywamy nowe obszary (mała nagroda) #TODO tutaj zmienić, aby uzależnić nagrodę od faktycznie poszerzonej widoczności.

        # Budowa miasta
        elif action_type == ActionType.BUILD_CITY:
            if self.selected_unit_idx is not None:
                unit = current_player.units[self.selected_unit_idx]
                if unit.unit_type == UnitType.SETTLER:
                    # Pobieramy koszty
                    costs = self.costs[ActionType.BUILD_CITY]
                    # Odejmujemy zasoby
                    for resource_type, amount in costs.items():
                        current_player.resources[resource_type] -= amount

                    # Tworzymy nowe miasto
                    city = City(x=unit.x, y=unit.y, player_id=current_player.player_id,
                                name=f"City {len(current_player.cities) + 1}")
                    current_player.add_city(city)

                    # Aktualizujemy mapę
                    self.map[unit.y, unit.x] = TileType.CITY.value

                    # Usuwamy osadnika
                    current_player.units.pop(self.selected_unit_idx)
                    self.selected_unit_idx = None
                    reward = 10 # Duża nagroda za zbudowanie miasta

        # Budowa farmy
        elif action_type == ActionType.BUILD_FARM:
            if self.selected_unit_idx is not None:
                unit = current_player.units[self.selected_unit_idx]
                if unit.unit_type == UnitType.WORKER:
                    costs = self.costs[ActionType.BUILD_FARM]
                    for resource_type, amount in costs.items():
                        current_player.resources[resource_type] -= amount
                    self.map[unit.y, unit.x] = TileType.FARM.value
                    unit.has_acted = True # Zaznaczamy, że jednostka wykonała akcję
                    reward = 2 # Nagroda za budowę

        # Budowa kopalni
        elif action_type == ActionType.BUILD_MINE:
            if self.selected_unit_idx is not None:
                unit = current_player.units[self.selected_unit_idx]
                if unit.unit_type == UnitType.WORKER:
                    costs = self.costs[ActionType.BUILD_MINE]
                    for resource_type, amount in costs.items():
                        current_player.resources[resource_type] -= amount
                    self.map[unit.y, unit.x] = TileType.MINE.value
                    unit.has_acted = True
                    reward = 2

        # Budowa tartaku
        elif action_type == ActionType.BUILD_LUMBERMILL:
            if self.selected_unit_idx is not None:
                unit = current_player.units[self.selected_unit_idx]
                if unit.unit_type == UnitType.WORKER:
                    costs = self.costs[ActionType.BUILD_LUMBERMILL]
                    for resource_type, amount in costs.items():
                        current_player.resources[resource_type] -= amount
                    self.map[unit.y, unit.x] = TileType.LUMBERMILL.value
                    unit.has_acted = True
                    reward = 2

        # Akcje związane z miastem
        elif action_type in [ActionType.CREATE_SETTLER, ActionType.CREATE_WARRIOR, ActionType.CREATE_WORKER]:
            # Tworzenie jednostki
            if self.selected_city_idx is not None:
                city = current_player.cities[self.selected_city_idx]

                # Określamy typ jednostki
                unit_type = None
                if action_type == ActionType.CREATE_SETTLER:
                    unit_type = UnitType.SETTLER
                elif action_type == ActionType.CREATE_WARRIOR:
                    unit_type = UnitType.WARRIOR
                elif action_type == ActionType.CREATE_WORKER:
                    unit_type = UnitType.WORKER

                # Pobieramy koszty
                costs = self.costs[action_type]
                # Odejmujemy zasoby
                for resource_type, amount in costs.items():
                    current_player.resources[resource_type] -= amount

                # Szukamy wolnego pola wokół miasta
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                found = False

                for dx, dy in directions:
                    new_x, new_y = city.x + dx, city.y + dy

                    # Sprawdzamy, czy pole jest na mapie i nie jest wodą
                    if (0 <= new_x < self.map_size[1] and 0 <= new_y < self.map_size[0] and
                            self.map[new_y, new_x] != TileType.WATER.value):

                        # Sprawdzamy, czy pole jest wolne (nie ma na nim innej jednostki)
                        is_occupied = False
                        for player in self.players:
                            for other_unit in player.units:
                                if other_unit.x == new_x and other_unit.y == new_y:
                                    is_occupied = True
                                    break
                            if is_occupied:
                                break

                        if not is_occupied:
                            # Tworzymy nową jednostkę
                            unit = Unit(unit_type=unit_type, x=new_x, y=new_y, player_id=current_player.player_id)
                            current_player.add_unit(unit)
                            city.has_acted = True # Zaznaczamy, że miasto wykonało akcję
                            reward = 3 # Nagroda za stworzenie jednostki
                            found = True
                            break

                if not found:
                    # Nie udało się stworzyć jednostki - zwracamy zasoby
                    for resource_type, amount in costs.items():
                        current_player.resources[resource_type] += amount
                    reward = -1 # Kara za nieudaną akcję

        # Budowa budynku w mieście
        elif action_type in [ActionType.BUILD_HOUSE, ActionType.BUILD_GRANARY,
                             ActionType.BUILD_BARRACKS, ActionType.BUILD_MARKET]:
            if self.selected_city_idx is not None:
                city = current_player.cities[self.selected_city_idx]

                # Określamy typ budynku
                building_type = None
                if action_type == ActionType.BUILD_HOUSE:
                    building_type = BuildingType.HOUSE
                elif action_type == ActionType.BUILD_GRANARY:
                    building_type = BuildingType.GRANARY
                elif action_type == ActionType.BUILD_BARRACKS:
                    building_type = BuildingType.BARRACKS
                elif action_type == ActionType.BUILD_MARKET:
                    building_type = BuildingType.MARKET

                # Pobieramy koszty
                costs = self.costs[action_type]

                # Odejmujemy zasoby
                for resource_type, amount in costs.items():
                    current_player.resources[resource_type] -= amount

                # Dodajemy budynek do miasta
                city.buildings.add(building_type)
                city.has_acted = True # Zaznaczamy, że miasto wykonało akcję
                reward = 4 # Nagroda za budowę budynku

        return reward


    #TODO zamienić to na agentów z Gymnasium.
    def _execute_ai_turn(self, player):

        # Prosta strategia AI:
        # 1. Jeśli mamy robotników, budujemy struktury
        # 2. Jeśli mamy osadników, budujemy miasta
        # 3. Poruszamy wojowników w kierunku przeciwnika
        # 4. W miastach tworzymy jednostki

        # Najpierw obsługujemy jednostki
        for unit in player.units:
            if unit.unit_type == UnitType.WORKER:

                # Dla robotnika - próbujemy budować struktury
                tile_type = self.map[unit.y, unit.x]

                # Budujemy farmę
                if tile_type == TileType.GRASS.value and player.can_afford(self.costs[ActionType.BUILD_FARM]):
                    for resource_type, amount in self.costs[ActionType.BUILD_FARM].items():
                        player.resources[resource_type] -= amount
                    self.map[unit.y, unit.x] = TileType.FARM.value
                    unit.has_acted = True
                # Budujemy tartak
                elif tile_type == TileType.FOREST.value and player.can_afford(self.costs[ActionType.BUILD_LUMBERMILL]):
                    for resource_type, amount in self.costs[ActionType.BUILD_LUMBERMILL].items():
                        player.resources[resource_type] -= amount
                    self.map[unit.y, unit.x] = TileType.LUMBERMILL.value
                    unit.has_acted = True
                # Budujemy kopalnię
                elif tile_type == TileType.MOUNTAIN.value and player.can_afford(self.costs[ActionType.BUILD_MINE]):
                    for resource_type, amount in self.costs[ActionType.BUILD_MINE].items():
                        player.resources[resource_type] -= amount
                    self.map[unit.y, unit.x] = TileType.MINE.value
                    unit.has_acted = True
                else:
                    # Poruszamy się losowo
                    self._move_unit_randomly(unit, player)

            # Dla osadnika - próbujemy budować miasto
            elif unit.unit_type == UnitType.SETTLER:
                # Sprawdzamy, czy możemy zbudować miasto na aktualnej pozycji
                can_build_city = True

                # Nie możemy budować na wodzie
                if self.map[unit.y, unit.x] == TileType.WATER.value:
                    can_build_city = False

                # Sprawdzamy, czy nie ma już miasta w pobliżu
                for other_player in self.players:
                    for city in other_player.cities:
                        dist = abs(city.x - unit.x) + abs(city.y - unit.y)
                        if dist < 3: # Minimalna odległość między miastami
                            can_build_city = False
                            break

                # Sprawdzamy, czy mamy wystarczające zasoby
                if not player.can_afford(self.costs[ActionType.BUILD_CITY]):
                    can_build_city = False

                # Budujemy miasto
                if can_build_city:
                    for resource_type, amount in self.costs[ActionType.BUILD_CITY].items():
                        player.resources[resource_type] -= amount

                    # Tworzymy nowe miasto
                    city = City(x=unit.x, y=unit.y, player_id=player.player_id,
                                name=f"City {len(player.cities) + 1}")
                    player.add_city(city)
                    # Aktualizujemy mapę
                    self.map[unit.y, unit.x] = TileType.CITY.value
                    # Usuwamy osadnika
                    player.units.remove(unit)
                else:
                    # Poruszamy się losowo
                    self._move_unit_randomly(unit, player)

            # Dla wojownika - próbujemy atakować przeciwnika lub poruszać się w jego kierunku
            elif unit.unit_type == UnitType.WARRIOR:
                has_moved = False # Najpierw sprawdzamy, czy w pobliżu jest wróg
                for other_player in self.players:
                    if other_player.player_id != player.player_id:
                        for enemy_unit in other_player.units:
                            dist = abs(enemy_unit.x - unit.x) + abs(enemy_unit.y - unit.y)
                            if dist <= 1: # Jednostka przeciwnika jest obok
                                # Atakujemy
                                other_player.units.remove(enemy_unit)
                                unit.has_acted = True
                                has_moved = True
                                break
                            elif dist <= 5 and unit.movement_left > 0: # Jednostka przeciwnika jest niedaleko
                                # Poruszamy się w kierunku wroga
                                if enemy_unit.x > unit.x and self._can_move_to(unit.x + 1, unit.y):
                                    unit.x += 1
                                    unit.movement_left -= 1
                                    has_moved = True
                                elif enemy_unit.x < unit.x and self._can_move_to(unit.x - 1, unit.y):
                                    unit.x -= 1
                                    unit.movement_left -= 1
                                    has_moved = True
                                elif enemy_unit.y > unit.y and self._can_move_to(unit.x, unit.y + 1):
                                    unit.y += 1
                                    unit.movement_left -= 1
                                    has_moved = True
                                elif enemy_unit.y < unit.y and self._can_move_to(unit.x, unit.y - 1):
                                    unit.y -= 1
                                    unit.movement_left -= 1
                                    has_moved = True

                                # Aktualizujemy widoczność
                                self._update_visibility(player, unit.x, unit.y)
                                break
                    if has_moved:
                        break

                if not has_moved and unit.movement_left > 0:
                    # Poruszamy się losowo
                    self._move_unit_randomly(unit, player)

        # Obsługujemy miasta
        for city in player.cities:
            if not city.has_acted:
                # Priorytetowo tworzymy jednostki
                if len(player.units) < 10: # Limit jednostek
                    # Najpierw próbujemy tworzyć osadników, by rozwijać cywilizację
                    if player.can_afford(self.costs[ActionType.CREATE_SETTLER]) and random.random() < 0.3:
                        self._create_unit_around_city(city, player, UnitType.SETTLER, ActionType.CREATE_SETTLER)
                    # Następnie wojowników
                    elif player.can_afford(self.costs[ActionType.CREATE_WARRIOR]) and random.random() < 0.4:
                        self._create_unit_around_city(city, player, UnitType.WARRIOR, ActionType.CREATE_WARRIOR)
                    # Na końcu robotników
                    elif player.can_afford(self.costs[ActionType.CREATE_WORKER]) and random.random() < 0.5:
                        self._create_unit_around_city(city, player, UnitType.WORKER, ActionType.CREATE_WORKER)

                # Jeśli nie stworzyliśmy jednostki, budujemy budynki
                if not city.has_acted:
                    # Budujemy dom
                    if (BuildingType.HOUSE not in city.buildings and
                            player.can_afford(self.costs[ActionType.BUILD_HOUSE])):
                        for resource_type, amount in self.costs[ActionType.BUILD_HOUSE].items():
                            player.resources[resource_type] -= amount
                        city.buildings.add(BuildingType.HOUSE)
                        city.has_acted = True

                    # Budujemy spichlerz
                    elif (BuildingType.GRANARY not in city.buildings and
                          player.can_afford(self.costs[ActionType.BUILD_GRANARY])):
                        for resource_type, amount in self.costs[ActionType.BUILD_GRANARY].items():
                            player.resources[resource_type] -= amount
                        city.buildings.add(BuildingType.GRANARY)
                        city.has_acted = True

                    # Budujemy koszary
                    elif (BuildingType.BARRACKS not in city.buildings and
                          player.can_afford(self.costs[ActionType.BUILD_BARRACKS])):
                        for resource_type, amount in self.costs[ActionType.BUILD_BARRACKS].items():
                            player.resources[resource_type] -= amount
                        city.buildings.add(BuildingType.BARRACKS)
                        city.has_acted = True

                    # Budujemy targ
                    elif (BuildingType.MARKET not in city.buildings and
                          player.can_afford(self.costs[ActionType.BUILD_MARKET])):
                        for resource_type, amount in self.costs[ActionType.BUILD_MARKET].items():
                            player.resources[resource_type] -= amount
                        city.buildings.add(BuildingType.MARKET)
                        city.has_acted = True

    def _can_move_to(self, x, y):
        """Sprawdza, czy jednostka może się poruszyć na dane pole."""
        if not (0 <= x < self.map_size[1] and 0 <= y < self.map_size[0]):
            return False
        # Sprawdzamy, czy pole nie jest wodą
        if self.map[y, x] == TileType.WATER.value:
            return False

        # Sprawdzamy, czy nie ma już jednostki na tym polu
        for player in self.players:
            for unit in player.units:
                if unit.x == x and unit.y == y:
                    return False
        return True

    def _move_unit_randomly(self, unit, player):
        """Porusza jednostkę w losowym kierunku."""
        if unit.movement_left <= 0:
            return

        # Kierunki ruchu
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            new_x, new_y = unit.x + dx, unit.y + dy
            if self._can_move_to(new_x, new_y):
                unit.x, unit.y = new_x, new_y
                unit.movement_left -= 1

                # Aktualizujemy widoczność
                self._update_visibility(player, new_x, new_y)
                break

    def _create_unit_around_city(self, city, player, unit_type, action_type):
        """Próbuje stworzyć jednostkę wokół miasta."""
        # Kierunki
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        random.shuffle(directions)

        for dx, dy in directions:
            new_x, new_y = city.x + dx, city.y + dy

            # Sprawdzamy, czy pole jest na mapie i nie jest wodą
            if (0 <= new_x < self.map_size[1] and 0 <= new_y < self.map_size[0] and
                    self.map[new_y, new_x] != TileType.WATER.value):
                is_occupied = False # Sprawdzamy, czy pole jest wolne
                for other_player in self.players:
                    for other_unit in other_player.units:
                        if other_unit.x == new_x and other_unit.y == new_y:
                            is_occupied = True
                            break
                    if is_occupied:
                        break

                if not is_occupied:
                    costs = self.costs[action_type]
                    for resource_type, amount in costs.items():
                        player.resources[resource_type] -= amount
                    unit = Unit(unit_type=unit_type, x=new_x, y=new_y, player_id=player.player_id)
                    player.add_unit(unit)
                    city.has_acted = True # Zaznaczamy, że miasto wykonało akcję
                    return True
        return False


    def _end_turn(self):
        """Kończy turę aktualnego gracza i przechodzi do następnego."""
        # Resetujemy akcje jednostek i miast
        current_player = self.players[self.current_player_idx]
        for unit in current_player.units:
            unit.reset_turn()
        for city in current_player.cities:
            city.reset_turn()

        # Przechodzimy do następnego gracza
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

        # Jeśli wróciliśmy do gracza 0,
        if self.current_player_idx == 0:
            # Zwiększamy numer tury
            self.turn_number += 1

            # Aktualizujemy zasoby dla wszystkich graczy
            self._update_resources()

    def _update_resources(self):
        """Aktualizuje zasoby dla wszystkich graczy na początku nowej tury."""
        for player in self.players:
            # Pobieramy zasoby z kafelków wokół miast
            for city in player.cities:
                # Zbieramy zasoby z kafelków w zasięgu miasta (promień 2)
                #TODO zmienić, aby zasięg się zwiększał w zależności od rozwoju miasta / wykrywanie konflików.
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        x, y = city.x + dx, city.y + dy

                        # Sprawdzamy, czy kafelek jest na mapie
                        if 0 <= x < self.map_size[1] and 0 <= y < self.map_size[0]:
                            tile_type = TileType(self.map[y, x]) # Pobieramy typ kafelka

                            # Dodajemy produkcję z tego kafelka
                            if tile_type in self.tile_production:
                                for resource_type, amount in self.tile_production[tile_type].items():
                                    player.resources[resource_type] += amount

                # Dodajemy produkcję z budynków w mieście
                for building in city.buildings:
                    if building in self.building_production:
                        for resource_type, amount in self.building_production[building].items():
                            player.resources[resource_type] += amount

            # Uwzględniamy koszt utrzymania jednostek (odejmujemy 1 złota za każdą jednostkę)
            #TODO zmienić, tak aby różne jedostki miały różne koszty utrzymania.
            maintenance_cost = len(player.units)
            player.resources[ResourceType.GOLD] = max(0, player.resources[ResourceType.GOLD] - maintenance_cost)

            # Zapewniamy minimalną ilość zasobów (aby gracz nie utknął bez możliwości działania)
            # TODO to można byłoby zmienić jakoś, np aby jednostki umierały czy coś takiego.
            player.resources[ResourceType.FOOD] = max(5, player.resources[ResourceType.FOOD])
            player.resources[ResourceType.WOOD] = max(5, player.resources[ResourceType.WOOD])
            player.resources[ResourceType.STONE] = max(3, player.resources[ResourceType.STONE])
            player.resources[ResourceType.GOLD] = max(2, player.resources[ResourceType.GOLD])

    def _check_victory(self):
        """Check victory conditions - control 70% of map or defeat all opponents."""
        total_tiles = self.map_size[0] * self.map_size[1]

        # Check if any player controls 70% of the map
        for player in self.players:
            controlled = player.get_controlled_tiles(self.map_size)
            if len(controlled) / total_tiles >= 0.7:
                return player.player_id

        # Check if only one player remains with cities
        players_with_cities = [p for p in self.players if len(p.cities) > 0]
        if len(players_with_cities) == 1:
            return players_with_cities[0].player_id

        return None

    def _render_init(self):
        """Initialize pygame rendering."""
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(
            (self.map_size[1] * self.cell_size,
             self.map_size[0] * self.cell_size + 100))  # Extra space for info
        pygame.display.set_caption("MicroCiv")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def render(self):
        """Render the game state using pygame."""
        if self.window is None and self.render_mode == "human":
            self._render_init()

        canvas = pygame.Surface(
            (self.map_size[1] * self.cell_size,
             self.map_size[0] * self.cell_size + 100))
        canvas.fill((255, 255, 255))

        self._draw_map(canvas)
        self._draw_units(canvas)
        self._draw_player_info(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_map(self, canvas):
        """Draw the game map."""
        current_player = self.players[self.current_player_idx]

        for y in range(self.map_size[0]):
            for x in range(self.map_size[1]):
                if (x, y) in current_player.explored_tiles:
                    tile_type = TileType(self.map[y, x])
                    color = self.colors[tile_type]
                else:
                    color = self.colors[TileType.UNEXPLORED]

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size - 1,
                        self.cell_size - 1,
                        ),
                )

    def _draw_units(self, canvas):
        """Draw all visible units."""
        current_player = self.players[self.current_player_idx]

        for player in self.players:
            for unit in player.units:
                if (unit.x, unit.y) in current_player.visible_tiles:
                    color = self.player_colors[player.player_id % len(self.player_colors)]
                    pygame.draw.circle(
                        canvas,
                        color,
                        (unit.x * self.cell_size + self.cell_size // 2,
                         unit.y * self.cell_size + self.cell_size // 2),
                        self.cell_size // 3,
                        )

                    # Draw unit type indicator
                    unit_symbol = ""
                    if unit.unit_type == UnitType.SETTLER:
                        unit_symbol = "S"
                    elif unit.unit_type == UnitType.WARRIOR:
                        unit_symbol = "W"
                    elif unit.unit_type == UnitType.WORKER:
                        unit_symbol = "K"

                    text = self.font.render(unit_symbol, True, (255, 255, 255))
                    canvas.blit(
                        text,
                        (unit.x * self.cell_size + self.cell_size // 2 - 5,
                         unit.y * self.cell_size + self.cell_size // 2 - 8),
                    )


    def _draw_player_info(self, canvas):
        """Draw player information at the bottom of the screen."""
        current_player = self.players[self.current_player_idx]
        y_offset = self.map_size[0] * self.cell_size

        # Draw player resources
        resources_text = f"Food: {current_player.resources[ResourceType.FOOD]}  " \
                         f"Wood: {current_player.resources[ResourceType.WOOD]}  " \
                         f"Stone: {current_player.resources[ResourceType.STONE]}  " \
                         f"Gold: {current_player.resources[ResourceType.GOLD]}"
        text = self.font.render(resources_text, True, (0, 0, 0))
        canvas.blit(text, (10, y_offset + 10))

        # Draw turn and player info
        turn_text = f"Turn: {self.turn_number}/{self.max_turns} | Player: {current_player.player_id + 1}"
        text = self.font.render(turn_text, True, (0, 0, 0))
        canvas.blit(text, (10, y_offset + 40))

        # Draw selected unit/city info
        if self.selected_unit_idx is not None and 0 <= self.selected_unit_idx < len(current_player.units):
            unit = current_player.units[self.selected_unit_idx]
            unit_text = f"Selected: {unit.unit_type.name} at ({unit.x}, {unit.y})"
            text = self.font.render(unit_text, True, (0, 0, 0))
            canvas.blit(text, (10, y_offset + 70))
        elif self.selected_city_idx is not None and 0 <= self.selected_city_idx < len(current_player.cities):
            city = current_player.cities[self.selected_city_idx]
            city_text = f"Selected: {city.name} at ({city.x}, {city.y})"
            text = self.font.render(city_text, True, (0, 0, 0))
            canvas.blit(text, (10, y_offset + 70))

    def close(self):
        """Close the environment and pygame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def select_unit(self, x: int, y: int):
        """Select a unit at the given coordinates."""
        current_player = self.players[self.current_player_idx]
        self.selected_unit_idx = None
        self.selected_city_idx = None

        for i, unit in enumerate(current_player.units):
            if unit.x == x and unit.y == y:
                self.selected_unit_idx = i
                return True
        return False

    def select_city(self, x: int, y: int):
        """Select a city at the given coordinates."""
        current_player = self.players[self.current_player_idx]
        self.selected_unit_idx = None
        self.selected_city_idx = None

        for i, city in enumerate(current_player.cities):
            if city.x == x and city.y == y:
                self.selected_city_idx = i
                return True
        return False

    def process_event(self, event):
        """Process pygame events for human interaction."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos[0] // self.cell_size, event.pos[1] // self.cell_size
            if 0 <= x < self.map_size[1] and 0 <= y < self.map_size[0]:
                if event.button == 1:  # Left click
                    if not self.select_unit(x, y):
                        self.select_city(x, y)
                elif event.button == 3:  # Right click
                    self.selected_unit_idx = None
                    self.selected_city_idx = None
            return True
        return False
