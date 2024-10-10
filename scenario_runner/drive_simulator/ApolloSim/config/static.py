from __future__ import annotations

from dataclasses import dataclass
from typing import List

from scenario_runner.drive_simulator.ApolloSim.config import WaypointConfig, BaseConfig, BaseConfigPool

@dataclass
class StaticObstacleConfig(BaseConfig):

    def __init__(
            self,
            idx: int,
            category: str,
            initial_waypoint: WaypointConfig,
            mutable: bool = True,
            trigger_time: float = 0.0,
            role: str = 'static',
    ):
        super(StaticObstacleConfig, self).__init__(
            idx,
            category,
            initial_waypoint,
            mutable,
            trigger_time,
            role
        )

@dataclass
class StaticObstacleConfigPool(BaseConfigPool):

    def __init__(self, configs: List[StaticObstacleConfig]):
        super(StaticObstacleConfigPool, self).__init__(configs)