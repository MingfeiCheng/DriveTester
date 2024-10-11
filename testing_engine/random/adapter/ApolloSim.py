import copy
from loguru import logger
from typing import Optional
from common.data_provider import DataProvider
from scenario_runner.drive_simulator.ApolloSim.config import VehicleConfigPool, WalkerConfigPool, \
    StaticObstacleConfigPool, ApolloConfigPool
from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder
from scenario_runner.drive_simulator.ApolloSim.map.apollo_map_parser import MapParser
from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig
from testing_engine.scenario_tool import ApolloSim as ApolloSimTool

class ApolloSimAdaptor:

    def __init__(
            self,
            map_name: str,
            ego_lane_start_id: str,
            ego_lane_end_id: str,
            num_vehicles: int,
            num_walkers: int,
            num_statics: int
    ):
        DataProvider.map_name = map_name
        DataProvider.map_parser = MapParser()
        DataProvider.map_parser.load_from_pkl(map_name)

        self.num_vehicles = num_vehicles
        self.num_walkers = num_walkers
        self.num_statics = num_statics
        self.ego_lane_start_id = ego_lane_start_id
        self.ego_lane_end_id = ego_lane_end_id
        self.ego_lanes, self.potential_lanes = DataProvider.map_parser.get_potential_lanes(
            self.ego_lane_start_id,
            self.ego_lane_end_id
        )
        # logger.debug(self.ego_lanes)
        # logger.debug(self.potential_lanes)
        self.max_tries = 100

    def mutation_adaptor(self, scenario_config: Optional[ApolloSimScenarioConfig]) -> ApolloSimScenarioConfig:
        # generate
        # if scenario_config is None:
        #     scenario_config = self.create_new_scenario_config()
        #     return scenario_config
        # else:
        #     mutated_scenario = copy.deepcopy(scenario_config)
        #     mutated_scenario = self.mutation_adaptor(mutated_scenario)
        #     return mutated_scenario
        scenario_config = self.create_new_scenario_config()
        return scenario_config

    def feedback_adaptor(self, result: ScenarioRecorder):
        return None

    # other tools
    def create_new_scenario_config(self) -> ApolloSimScenarioConfig:
        existing_actor_configs = []

        # create ego part
        ego_id = 0
        ego_configs = []
        ego_config = ApolloSimTool.create_new_apollo(
            ego_id,
            self.ego_lane_start_id,
            self.ego_lane_end_id,
            True,
            False
        )
        ego_configs.append(ego_config)
        existing_actor_configs.append(ego_config)

        # create vehicle part
        vehicle_configs = []
        try_times = 0
        npc_id = 10000
        while len(vehicle_configs) < self.num_vehicles and try_times < self.max_tries:
            vehicle_config = ApolloSimTool.create_new_follow_lane_waypoint_vehicle(npc_id, self.potential_lanes)
            if not ApolloSimTool.conflict_checker(existing_actor_configs, vehicle_config):
                vehicle_configs.append(vehicle_config)
                existing_actor_configs.append(vehicle_config)
                npc_id += 1
            try_times += 1
        logger.info(f'Create {len(vehicle_configs)} vehicles')

        # create walker part
        walker_configs = []
        try_times = 0
        npc_id = 20000
        while len(walker_configs) < self.num_walkers and try_times < self.max_tries:
            walker_config = ApolloSimTool.create_new_waypoint_walker(npc_id, self.ego_lanes)
            if not ApolloSimTool.conflict_checker(existing_actor_configs, walker_config):
                walker_configs.append(walker_config)
                existing_actor_configs.append(walker_config)
                npc_id += 1
            try_times += 1
        logger.info(f'Create {len(walker_configs)} walkers')

        # create static prat
        static_obstacle_configs = []
        try_times = 0
        npc_id = 30000
        while len(static_obstacle_configs) < self.num_statics and try_times < self.max_tries:
            static_obstacle_config = ApolloSimTool.create_new_static_obstacle(npc_id, self.potential_lanes)
            if not ApolloSimTool.conflict_checker(existing_actor_configs, static_obstacle_config):
                static_obstacle_configs.append(static_obstacle_config)
                existing_actor_configs.append(static_obstacle_config)
                npc_id += 1
            try_times += 1
        logger.info(f'Create {len(static_obstacle_configs)} statics')

        # create traffic light part
        traffic_light_config = ApolloSimTool.create_traffic_light()

        return ApolloSimScenarioConfig(
            idx=0,
            ego_config_pool=ApolloConfigPool(ego_configs),
            vehicle_config_pool=VehicleConfigPool(vehicle_configs),
            walker_config_pool=WalkerConfigPool(walker_configs),
            static_obstacle_config_pool=StaticObstacleConfigPool(static_obstacle_configs),
            traffic_light_config=traffic_light_config
        )

    def mutate_existing_scenario_config(self, scenario_config: ApolloSimScenarioConfig) -> ApolloSimScenarioConfig:
        pass