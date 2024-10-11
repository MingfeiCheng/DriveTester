import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

from common.data_provider import DataProvider
from scenario_runner.drive_simulator.ApolloSim.config import StaticObstacleConfigPool, WalkerConfigPool, VehicleConfigPool, TrafficLightConfig
from scenario_runner.drive_simulator.ApolloSim.config.apollo import ApolloConfigPool

@dataclass
class ApolloSimScenarioConfig:
    """
    Currently, we only support single ego vehicle
    """
    idx: Any # for scenario identifier
    ego_config_pool: ApolloConfigPool
    static_obstacle_config_pool: StaticObstacleConfigPool
    walker_config_pool: WalkerConfigPool
    vehicle_config_pool: VehicleConfigPool
    traffic_light_config: TrafficLightConfig

    def __init__(
            self,
            idx: Any,
            ego_config_pool: ApolloConfigPool,
            static_obstacle_config_pool: StaticObstacleConfigPool,
            walker_config_pool: WalkerConfigPool,
            vehicle_config_pool: VehicleConfigPool,
            traffic_light_config: TrafficLightConfig,
    ):
        self.idx = idx
        self.ego_config_pool = ego_config_pool
        self.static_obstacle_config_pool = static_obstacle_config_pool
        self.walker_config_pool = walker_config_pool
        self.vehicle_config_pool = vehicle_config_pool
        self.traffic_light_config = traffic_light_config

    @classmethod
    def from_json(cls, json_node: Dict):
        json_node['ego_config_pool'] = ApolloConfigPool.from_json(json_node['ego_config_pool'])
        json_node['static_obstacle_config_pool'] = StaticObstacleConfigPool.from_json(json_node['static_obstacle_config_pool'])
        json_node['walker_config_pool'] = WalkerConfigPool.from_json(json_node['walker_config_pool'])
        json_node['vehicle_config_pool'] = VehicleConfigPool.from_json(json_node['vehicle_config_pool'])
        json_node['traffic_light_config'] = TrafficLightConfig.from_json(json_node['traffic_light_config'])
        return cls(**json_node)

    def json_data(self) -> Dict:
        return asdict(self)

    def export(self):
        json_data = self.json_data()
        file_path = os.path.join(DataProvider.scenario_folder(), f"{self.idx}.json")
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
