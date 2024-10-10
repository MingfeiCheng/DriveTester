from dataclasses import dataclass
from typing import Any, List, Dict

from scenario_runner.drive_simulator.ApolloSim.config import BaseConfig, BaseConfigPool, WaypointConfig

@dataclass
class ApolloConfig(BaseConfig):

    route: List[WaypointConfig]
    terminate_on_failure: bool = True

    def __init__(
            self,
            idx: Any,
            category: str,
            initial_waypoint: WaypointConfig,
            route: List[WaypointConfig],
            trigger_time: float,
            mutable: bool = True,
            termination_on_failure: bool = True,
            role: str = 'apollo'
    ):
        super(ApolloConfig, self).__init__(
            idx,
            category,
            initial_waypoint,
            mutable,
            trigger_time,
            role
        )

        self.route = route
        self.terminate_on_failure = termination_on_failure

    @classmethod
    def from_json(cls, json_node: Any):
        json_node['initial_waypoint'] = WaypointConfig.from_json(json_node['initial_waypoint'])
        json_route = json_node["route"]
        route = list()
        for item in json_route:
            route.append(WaypointConfig.from_json(item))
        json_node["route"] = route
        return cls(**json_node)

class ApolloConfigPool(BaseConfigPool):

    def __init__(self, configs: List[ApolloConfig]):
        super(ApolloConfigPool, self).__init__(configs)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'ApolloConfigPool':
        configs_js = json_node['configs']
        configs = list()
        for a_i, a_js in enumerate(configs_js):
            configs.append(ApolloConfig.from_json(a_js))
        json_node['configs'] = configs
        return cls(**json_node)