from enum import Enum


class ResourceType(Enum):
    FOOD = 0
    WOOD = 1
    STONE = 2
    GOLD = 3


class TileType(Enum):
    GRASS = 0
    FOREST = 1
    MOUNTAIN = 2
    WATER = 3
    CITY = 4
    FARM = 5
    MINE = 6
    LUMBERMILL = 7
    UNEXPLORED = 8


class UnitType(Enum):
    SETTLER = 0
    WARRIOR = 1
    WORKER = 2


class BuildingType(Enum):
    HOUSE = 0
    GRANARY = 1
    BARRACKS = 2
    MARKET = 3


class ActionType(Enum):
    # Akcje ruchu (0-3)
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3

    # Akcje budowy (4-7)
    BUILD_CITY = 4
    BUILD_FARM = 5
    BUILD_MINE = 6
    BUILD_LUMBERMILL = 7

    # Akcje tworzenia jednostek (8-11)
    CREATE_SETTLER = 8
    CREATE_WARRIOR = 9
    # CREATE_ARCHER = 10
    CREATE_WORKER = 11

    # Akcje budynków (12-15)
    BUILD_HOUSE = 12
    BUILD_GRANARY = 13
    BUILD_BARRACKS = 14
    BUILD_MARKET = 15

    # Akcje specjalne (16-17)
    END_TURN = 16
    DO_NOTHING = 17