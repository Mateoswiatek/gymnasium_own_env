from enums import UnitType, BuildingType, ResourceType
from typing import Dict

class Unit:
    def __init__(self, unit_type: UnitType, x: int, y: int, player_id: int):
        self.unit_type = unit_type
        self.x = x
        self.y = y
        self.player_id = player_id
        self.health = 100
        self.movement_left = self._get_max_movement()
        self.has_acted = False

    def _get_max_movement(self) -> int:
        if self.unit_type == UnitType.SETTLER:
            return 2
        elif self.unit_type == UnitType.WARRIOR:
            return 2
        elif self.unit_type == UnitType.WORKER:
            return 2
        return 1

    def reset_turn(self):
        self.movement_left = self._get_max_movement()
        self.has_acted = False


class City:
    def __init__(self, x: int, y: int, player_id: int, name: str = "City"):
        self.x = x
        self.y = y
        self.player_id = player_id
        self.name = name
        self.population = 1
        self.buildings = []
        self.production_queue = []
        self.health = 100
        self.has_acted = False

    def reset_turn(self):
        self.has_acted = False

    def add_building(self, building_type: BuildingType):
        if building_type not in self.buildings:
            self.buildings.append(building_type)

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
        for resource_type, cost in costs.items():
            if self.resources[resource_type] < cost:
                return False
        return True

    def pay_cost(self, costs: Dict[ResourceType, int]) -> bool:
        if not self.can_afford(costs):
            return False

        for resource_type, cost in costs.items():
            self.resources[resource_type] -= cost

        return True

    def add_resources(self, resources: Dict[ResourceType, int]):
        for resource_type, amount in resources.items():
            self.resources[resource_type] += amount

    def add_unit(self, unit: Unit):
        self.units.append(unit)

    def add_city(self, city: City):
        self.cities.append(city)

    def remove_unit(self, unit: Unit):
        if unit in self.units:
            self.units.remove(unit)

    def calculate_score(self):
        city_score = len(self.cities) * 100
        unit_score = len(self.units) * 10
        resource_score = sum(self.resources.values())
        tech_score = self.tech_level * 50
        explored_score = len(self.explored_tiles) * 2

        self.score = city_score + unit_score + resource_score + tech_score + explored_score
        return self.score
