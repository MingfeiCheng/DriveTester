import copy
import pickle

from loguru import logger
from typing import Optional

from omegaconf import DictConfig

from common.data_provider import DataProvider
from scenario_runner.drive_simulator.ApolloSim.config import VehicleConfigPool, WalkerConfigPool, \
    StaticObstacleConfigPool, ApolloConfigPool
from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder
from scenario_runner.drive_simulator.ApolloSim.map.apollo_map_parser import MapParser
from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig
from testing_engine.scenario_tool import ApolloSim as ApolloSimTool
from typing import List

class ScenarioAdaptor:

    def __init__(
            self,
            scenario_config: ApolloSimScenarioConfig,
            recorder: List,
            fitness: float
    ):
        self.id = scenario_config.idx
        self.scenario = scenario_config
        self.fitness = fitness
        self.recorder = recorder

    def update_id(self, idx):
        self.scenario.idx = idx
        self.id = idx

class ApolloSimAdaptor:

    def __init__(
            self,
            map_name: str,
            ego_lane_start_id: str,
            ego_lane_end_id: str,
            num_vehicles: int,
            num_walkers: int,
            num_statics: int,
            oracle_cfg: DictConfig
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

        self.max_tries = 100
        self.oracle_cfg = oracle_cfg

    def mutation_adaptor(
            self,
            scenario_config: Optional[ApolloSimScenarioConfig],
            mutation_stage: str
    ) -> ApolloSimScenarioConfig:
        # generate
        if scenario_config is None or mutation_stage == 'large':
            scenario_config = self.create_new_scenario_config()
            return scenario_config
        else:
            mutated_scenario = copy.deepcopy(scenario_config)
            mutated_scenario = self.mutate_existing_scenario_config(mutated_scenario)
            return mutated_scenario

    def feedback_adaptor(self, result: ScenarioRecorder):
        # calculate fitness
        feedback = result.scenario_feedback[0]
        # min is better
        fitness = []
        for key, value in feedback.items():
            if key == 'oracle.collision':
                fitness.append(min(value, 5.0) / 5.0)
            elif key == 'oracle.destination':
                threshold = self.oracle_cfg.destination.threshold
                fitness.append(float(max((threshold - feedback['oracle.destination']) / threshold, 0.0)))

        fitness = sum(fitness) / len(fitness)

        # calculate ego recordings
        traffic_record_path = result.traffic_record_path
        with open(traffic_record_path, 'rb') as f:
            traffic_record = pickle.load(f)

        ego_recorders = []
        for recording_frame in sorted(traffic_record.keys()):
            data = traffic_record[recording_frame]
            timestamp = data['timestamp']
            apollo_data = data['actor_state'][0]

            ego_recorders.append([
                timestamp,
                apollo_data['location']['x'],
                apollo_data['location']['y'],
                apollo_data['speed'],
                apollo_data['acceleration'],
                apollo_data['location']['yaw'],
                apollo_data['angular_speed']
            ])

        # is complete
        scenario_result = result.scenario_overview[0]
        if 'COLLISION' in  scenario_result:
            is_complete = False
        else:
            is_complete = True

        return fitness, ego_recorders, is_complete

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
        # skip ego
        # mutate vehicle
        vehicle_pool = scenario_config.vehicle_config_pool
        for i, vehicle_config in enumerate(vehicle_pool.configs):
            vehicle_config = ApolloSimTool.mutate_vehicle_speed(vehicle_config)
            vehicle_pool.configs[i] = vehicle_config

        # mutate walker
        walker_pool = scenario_config.walker_config_pool
        for i, walker_config in enumerate(walker_pool.configs):
            walker_config = ApolloSimTool.mutate_walker_speed(walker_config)
            walker_pool.configs[i] = walker_config

        # mutate static
        static_pool = scenario_config.static_obstacle_config_pool
        for i, static_config in enumerate(static_pool.configs):
            static_config = ApolloSimTool.mutate_static_location(static_config)
            static_pool.configs[i] = static_config

        return scenario_config