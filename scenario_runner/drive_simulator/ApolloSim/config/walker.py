from dataclasses import dataclass
from typing import List, Dict

from scenario_runner.drive_simulator.ApolloSim.config import BaseConfig, BaseConfigPool, WaypointConfig

@dataclass
class WalkerConfig(BaseConfig):

    behavior: List[WaypointConfig]

    def __init__(
            self,
            idx: int,
            category: str,
            initial_waypoint: WaypointConfig,
            mutable: bool = True,
            trigger_time: float = 0.0,
            behavior: List[WaypointConfig] = None,
            role: str = 'unknown',
    ):
        super(WalkerConfig, self).__init__(
            idx,
            category,
            initial_waypoint,
            mutable,
            trigger_time,
            role
        )
        self.behavior = behavior

    @classmethod
    def from_json(cls, json_node: Dict) -> 'WalkerConfig':
        json_node['initial_waypoint'] = WaypointConfig.from_json(json_node['initial_waypoint'])
        behavior = list()
        for r_i, r_js in enumerate(json_node['behavior']):
            behavior.append(WaypointConfig.from_json(r_js))
        json_node['behavior'] = behavior
        return cls(**json_node)

class WalkerConfigPool(BaseConfigPool):

    def __init__(self, configs: List[WalkerConfig]):
        super(WalkerConfigPool, self).__init__(configs)