import json

from loguru import logger
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from apollo_sim.agents.vehicle.apollo_vehicle import ApolloConfig
from apollo_sim.agents.static_obstacle import StaticObstacleConfig
from apollo_sim.agents.walker.waypoint_walker import WaypointWalkerConfig
from apollo_sim.agents.vehicle.waypoint_vehicle import WaypointVehicleConfig
from apollo_sim.agents.traffic_light.rule_light import RuleLightConfig

@dataclass
class ScenarioConfig:
    """
    Currently, we only support single ego vehicle
    """
    idx: Any # for scenario identifier
    apollo: List[ApolloConfig]
    static_obstacle: List[StaticObstacleConfig]
    walker: List[WaypointWalkerConfig]
    vehicle: List[WaypointVehicleConfig]
    traffic_light: RuleLightConfig

    def __init__(
            self,
            idx: Any,
            apollo: List[ApolloConfig],
            static_obstacle: List[StaticObstacleConfig],
            walker: List[WaypointWalkerConfig],
            vehicle: List[WaypointVehicleConfig],
            traffic_light: RuleLightConfig,
    ):
        self.idx = idx
        self.apollo = apollo
        self.static_obstacle = static_obstacle
        self.walker = walker
        self.vehicle = vehicle
        self.traffic_light = traffic_light

    @classmethod
    def from_json(cls, json_node: Dict):
        json_node['apollo'] = [ApolloConfig.from_json(apollo) for apollo in json_node['apollo']]
        json_node['static_obstacle'] = [StaticObstacleConfig.from_json(static_obstacle) for static_obstacle in json_node['static_obstacle']]
        json_node['walker'] = [WaypointWalkerConfig.from_json(walker) for walker in json_node['walker']]
        json_node['vehicle'] = [WaypointVehicleConfig.from_json(vehicle) for vehicle in json_node['vehicle']]
        json_node['traffic_light'] = RuleLightConfig.from_json(json_node['traffic_light'])
        return cls(**json_node)

    def json_data(self) -> Dict:
        return asdict(self)

    def export(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.json_data(), f, indent=4)