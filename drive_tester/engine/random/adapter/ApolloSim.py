from typing import Any, Dict
from loguru import logger
from omegaconf import DictConfig

from apollo_sim.sim_env import SimEnv

from drive_tester.scenario_runner.ApolloSim.config.scenario import ScenarioConfig
from drive_tester.scenario_runner.ApolloSim.tools.actor_tool import ActorTool
from drive_tester.scenario_runner.ApolloSim.tools.map_tool import MapTool

class ApolloSimAdapter:

    def __init__(
            self,
            simulator: SimEnv,
            scenario_config: DictConfig
    ):
        self.map = simulator.map

        self.num_vehicle = scenario_config.get("num_vehicle", 1)
        self.num_walker = scenario_config.get("num_walker", 1)
        self.num_static = scenario_config.get("num_static", 1)

        self.ego_start_lane_id = scenario_config.get("start_lane_id", None)
        self.ego_end_lane_id = scenario_config.get("end_lane_id", None)

        self.ego_route_path, self.map_region_lanes = MapTool.estimate_map_region(
            self.map,
            self.ego_start_lane_id,
            self.ego_end_lane_id
        )

        self.max_tries = 100

    def mutation_adapter(self, scenario_id: Any) -> ScenarioConfig:
        # create new scenario config
        existing_actor_configs = []

        # create ego part
        ego_id = 0
        ego_configs = []
        ego_config = ActorTool.create_apollo(
            self.map,
            ego_id,
            self.ego_start_lane_id,
            self.ego_end_lane_id
        )
        ego_configs.append(ego_config)
        existing_actor_configs.append(ego_config)

        # create vehicle part
        vehicle_configs = []
        try_times = 0
        npc_id = 10000
        while len(vehicle_configs) < self.num_vehicle and try_times < self.max_tries:
            vehicle_config = ActorTool.create_random_waypoint_vehicle(self.map, npc_id, self.map_region_lanes)
            if not ActorTool.conflict_checker(existing_actor_configs, vehicle_config):
                vehicle_configs.append(vehicle_config)
                existing_actor_configs.append(vehicle_config)
                npc_id += 1
            try_times += 1
        logger.info(f'Create {len(vehicle_configs)} vehicles')

        # create walker part
        walker_configs = []
        try_times = 0
        npc_id = 20000
        while len(walker_configs) < self.num_walker and try_times < self.max_tries:
            walker_config = ActorTool.create_random_waypoint_walker(self.map, npc_id, self.ego_route_path)
            if not ActorTool.conflict_checker(existing_actor_configs, walker_config):
                walker_configs.append(walker_config)
                existing_actor_configs.append(walker_config)
                npc_id += 1
            try_times += 1
        logger.info(f'Create {len(walker_configs)} walkers')

        # create static prat
        static_obstacle_configs = []
        try_times = 0
        npc_id = 30000
        while len(static_obstacle_configs) < self.num_static and try_times < self.max_tries:
            static_obstacle_config = ActorTool.create_random_static_obstacle(self.map, npc_id, self.map_region_lanes)
            if not ActorTool.conflict_checker(existing_actor_configs, static_obstacle_config):
                static_obstacle_configs.append(static_obstacle_config)
                existing_actor_configs.append(static_obstacle_config)
                npc_id += 1
            try_times += 1
        logger.info(f'Create {len(static_obstacle_configs)} statics')

        # create traffic light part - random rule light
        traffic_light_config = ActorTool.create_random_rule_light()

        return ScenarioConfig(
            idx=scenario_id,
            apollo=ego_configs,
            static_obstacle=static_obstacle_configs,
            walker=walker_configs,
            vehicle=vehicle_configs,
            traffic_light=traffic_light_config
        )

    def result_adapter(self, scenario_result: Dict):
        print(scenario_result)
        pass