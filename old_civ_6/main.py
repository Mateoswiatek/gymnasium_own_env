from enums import UnitType, BuildingType, ResourceType
from typing import Dict, Set, Tuple

class Unit:
    def __init__(self, unit_type: UnitType, x: int, y: int, player_id: int):
        self.unit_type = unit_type
        self.x = x
        self.y = y
        self.player_id = player_id
        self.health = 100
        self.movement_left = self._get_max_movement()
        self.max_movement = self._get_max_movement()
        self.has_acted = False
        self.attack_strength = self._get_attack_strength()

    def _get_max_movement(self) -> int:
        movement_values = {
            UnitType.SETTLER: 2,
            UnitType.WARRIOR: 2,
            UnitType.WORKER: 2
        }
        return movement_values.get(self.unit_type, 1)

    def _get_attack_strength(self) -> int:
        if self.unit_type == UnitType.WARRIOR:
            return 20
        return 0

    def reset_turn(self):
        self.movement_left = self.max_movement
        self.has_acted = False

    def move_to(self, x: int, y: int):
        self.x = x
        self.y = y
        self.movement_left -= 1


class City:
    def __init__(self, x: int, y: int, player_id: int, name: str = "City"):
        self.x = x
        self.y = y
        self.player_id = player_id
        self.name = name
        self.population = 1
        self.buildings: Set[BuildingType] = set()
        self.production_queue = []
        self.health = 100
        self.has_acted = False
        self.defense_strength = 10

    def reset_turn(self):
        self.has_acted = False

    def add_building(self, building_type: BuildingType):
        self.buildings.add(building_type)

    def get_production(self) -> Dict[ResourceType, int]:
        production = {
            ResourceType.FOOD: 2,
            ResourceType.GOLD: 1
        }
        return production


class Player:
    def __init__(self, player_id: int, is_ai: bool = False):
        self.player_id = player_id
        self.resources = {
            ResourceType.FOOD: 50,
            ResourceType.WOOD: 50,
            ResourceType.STONE: 20,
            ResourceType.GOLD: 20
        }
        self.units = []
        self.cities = []
        self.is_ai = is_ai
        self.explored_tiles = set()
        self.visible_tiles = set()
        self.score = 0
        self.tech_level = 1
        self.turn_number = 1

    def can_afford(self, costs: Dict[ResourceType, int]) -> bool:
        return all(self.resources.get(resource, 0) >= amount
                   for resource, amount in costs.items())

    def pay_cost(self, costs: Dict[ResourceType, int]) -> bool:
        if not self.can_afford(costs):
            return False

        for resource, amount in costs.items():
            self.resources[resource] -= amount
        return True

    def add_resources(self, resources: Dict[ResourceType, int]):
        for resource, amount in resources.items():
            self.resources[resource] = self.resources.get(resource, 0) + amount

    def add_unit(self, unit: Unit):
        self.units.append(unit)

    def add_city(self, city: City):
        self.cities.append(city)

    def remove_unit(self, unit: Unit):
        if unit in self.units:
            self.units.remove(unit)

    def calculate_score(self) -> int:
        city_score = len(self.cities) * 100
        unit_score = len(self.units) * 10
        resource_score = sum(self.resources.values()) // 10
        tech_score = self.tech_level * 50
        explored_score = len(self.explored_tiles) // 10

        self.score = city_score + unit_score + resource_score + tech_score + explored_score
        return self.score

    def get_controlled_tiles(self, map_size: Tuple[int, int]) -> Set[Tuple[int, int]]:
        controlled = set()
        for city in self.cities:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    x, y = city.x + dx, city.y + dy
                    if 0 <= x < map_size[0] and 0 <= y < map_size[1]:
                        controlled.add((x, y))
        return controlled
