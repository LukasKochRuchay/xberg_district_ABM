import random
from collections import defaultdict
from typing import DefaultDict

import mesa
import mesa_geo as mg
from shapely.geometry import Point

from ..agent.building import Building
from ..agent.commuter import Commuter

class District(mg.GeoSpace):
    homes: tuple[Building]
    workplaces: tuple[Building]
    _buildings: dict[int, Building]
    crowding_bin_size_m: float
    _commuters_pos_map: DefaultDict[tuple[int, int], set[Commuter]]
    _commuter_id_map: dict[int, Commuter]
    
    def __init__(self, crs, *, crowding_bin_size_m: float = 25.0) -> None:
        super().__init__(crs=crs)
        self.homes = ()
        self.workplaces = ()
        self._buildings = {}
        self.crowding_bin_size_m = float(crowding_bin_size_m)
        self._commuters_pos_map = defaultdict(set)
        self._commuter_id_map = {}

    def _pos_key(self, pos: mesa.space.FloatCoordinate) -> tuple[int, int]:
        """Quantize a continuous (x,y) into a stable bin key.

        Without quantization, exact float matching makes co-location crowding
        almost always zero.
        """
        bin_size = self.crowding_bin_size_m
        if bin_size <= 0:
            # Fallback to 1m bins.
            bin_size = 1.0
        return (int(pos[0] // bin_size), int(pos[1] // bin_size))

    def get_random_home(self) -> Building:
        return random.choice(self.homes)

    def get_random_workplace(self) -> Building:
        return random.choice(self.workplaces)

    def get_building_by_id(self, unique_id: int) -> Building:
        return self._buildings[unique_id]

    def add_buildings(self, agents) -> None:
        super().add_agents(agents)
        homes, workplaces = [], []
        for agent in agents:
            if isinstance(agent, Building):
                self._buildings[agent.unique_id] = agent
                # Refactor semantics for provided geofiles:
                # 0 => work, 1 => home.
                if agent.function in (1, 1.0):
                    homes.append(agent)
                elif agent.function in (0, 0.0):
                    workplaces.append(agent)

        self.workplaces = self.workplaces + tuple(workplaces)
        self.homes = self.homes + tuple(homes)
        
    def get_commuters_by_pos(
        self, float_pos: mesa.space.FloatCoordinate
    ) -> set[Commuter]:
        return self._commuters_pos_map[self._pos_key(float_pos)]
    
    def get_commuter_by_id(self, unique_id: int) -> Commuter:
        return self._commuter_id_map[unique_id]
    
    def add_commuter(self, agent: Commuter) -> None:
        super().add_agents([agent])
        self._commuters_pos_map[self._pos_key((agent.geometry.x, agent.geometry.y))].add(agent)
        self._commuter_id_map[agent.unique_id] = agent
        
    def move_commuter(
        self, commuter: Commuter, pos: mesa.space.FloatCoordinate
    ) -> None:
        self.__remove_commuter(commuter)
        commuter.geometry = Point(pos)
        self.add_commuter(commuter)

    def __remove_commuter(self, commuter: Commuter) -> None:
        super().remove_agent(commuter)
        del self._commuter_id_map[commuter.unique_id]
        self._commuters_pos_map[self._pos_key((commuter.geometry.x, commuter.geometry.y))].remove(
            commuter
        )
        
        
        

    