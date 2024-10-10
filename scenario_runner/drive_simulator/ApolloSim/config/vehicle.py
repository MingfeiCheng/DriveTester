from dataclasses import dataclass
from typing import List, Dict
from scenario_runner.drive_simulator.ApolloSim.config import BaseConfig, BaseConfigPool, WaypointConfig, CommandConfig

@dataclass
class VehicleConfig(BaseConfig):

    behavior: List[WaypointConfig or CommandConfig]

    def __init__(
            self,
            idx: int,
            category: str,
            initial_waypoint: WaypointConfig, # start location point
            mutable: bool = True,
            trigger_time: float = 0.0,
            behavior: List[WaypointConfig or CommandConfig] = None,
            role: str = 'vehicle'
    ):
        super(VehicleConfig, self).__init__(
            idx,
            category,
            initial_waypoint,
            mutable,
            trigger_time,
            role
        )
        self.behavior = behavior

    @classmethod
    def from_json(cls, json_node: Dict) -> 'VehicleConfig':
        json_node['initial_waypoint'] = WaypointConfig.from_json(json_node['initial_waypoint'])
        behavior = list()
        for r_i, r_js in enumerate(json_node['behavior']):
            if 'command' in json_node['behavior'][r_i]:
                behavior.append(CommandConfig.from_json(r_js))
            else:
                behavior.append(WaypointConfig.from_json(r_js))
        json_node['behavior'] = behavior
        return cls(**json_node)

class VehicleConfigPool(BaseConfigPool):

    def __init__(self, configs: List[VehicleConfig]):
        super(VehicleConfigPool, self).__init__(configs)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'VehicleConfigPool':
        configs_js = json_node['configs']
        configs = list()
        for a_i, a_js in enumerate(configs_js):
            configs.append(VehicleConfigPool.from_json(a_js))
        json_node['configs'] = configs
        return cls(**json_node)