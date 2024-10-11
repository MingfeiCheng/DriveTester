import copy
import random
import time

from loguru import logger

from common.data_provider import DataProvider
from scenario_runner.drive_simulator.ApolloSim.config import VehicleConfigPool, WalkerConfigPool, \
    StaticObstacleConfigPool, ApolloConfigPool
from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder
from scenario_runner.drive_simulator.ApolloSim.map.apollo_map_parser import MapParser
from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig
from testing_engine.scenario_tool import ApolloSim as ApolloSimTool
from typing import List

class ApolloSimAdaptor:

    def __init__(
            self,
            map_name: str,
            ego_lane_start_id: str,
            ego_lane_end_id: str,
            oracle_cfg: list
    ):
        DataProvider.map_name = map_name
        DataProvider.map_parser = MapParser()
        DataProvider.map_parser.load_from_pkl(map_name)

        self.ego_lane_start_id = ego_lane_start_id
        self.ego_lane_end_id = ego_lane_end_id
        self.ego_lanes, self.potential_lanes = DataProvider.map_parser.get_potential_lanes(
            self.ego_lane_start_id,
            self.ego_lane_end_id
        )
        # logger.debug(self.ego_lanes)
        # logger.debug(self.potential_lanes)
        self.max_tries = 100
        self.oracle_cfg = oracle_cfg

    def mutation_adaptor(
            self,
            candidate_value: List,
    ) -> ApolloSimScenarioConfig:
        """
        workflow:
            1. add static obstacles before add vehicles
            2. if static obstacles are enough -> add vehicles
        specific flow:
            1. ADD -> check total number
            2. If > total number limitation -> remove
        """
        # 0 Scenario ID - skip
        # 1 Vehicle_in_front
        # 2 vehicle_in_adjcent_lane
        # 3 vehicle_in_opposite_lane
        # 4 vehicle_in_front_two_wheeled
        # 5 vehicle_in_adjacent_two_wheeled
        # 6 vehicle_in_opposite_two_wheeled
        # 7 Target Speed
        # 8 Trigger Time
        logger.debug(candidate_value)
        ma = DataProvider.map_parser
        m_start_time = time.time()

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

        lane_ego_start = self.ego_lane_start_id
        pending_vehicle_lanes = []

        # 1 Vehicle_in_front
        if candidate_value[1] == 0:
            pass
        elif candidate_value[1] == 1:
            pending_lanes = ma.get_successor_lanes(lane_ego_start)
            pending_lanes += [lane_ego_start]
            if len(pending_lanes) > 0:
                pending_vehicle_lanes.append(random.choice(pending_lanes))

        # 2 vehicle_in_adjcent_lane
        if candidate_value[2] == 0:
            pass
        elif candidate_value[2] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='forward', side='both')
            if len(select_lane) > 0:
                pending_vehicle_lanes.append(random.choice(select_lane))

        # 3 vehicle_in_opposite_lane
        if candidate_value[3] == 0:
            pass
        elif candidate_value[3] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='reverse', side='both')
            if len(select_lane) > 0:
                pending_vehicle_lanes.append(random.choice(select_lane))

        pending_bicycle_lanes = []
        # 4 vehicle_in_front_two_wheeled
        if candidate_value[4] == 0:
            pass
        elif candidate_value[4] == 1:
            pending_lanes = ma.get_successor_lanes(lane_ego_start)
            pending_lanes += [lane_ego_start]
            if len(pending_lanes) > 0:
                pending_bicycle_lanes.append(random.choice(pending_lanes))

        # 5 vehicle_in_adjacent_two_wheeled
        if candidate_value[5] == 0:
            pass
        elif candidate_value[5] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='forward', side='both')
            if len(select_lane) > 0:
                pending_bicycle_lanes.append(random.choice(select_lane))

        # 6 vehicle_in_opposite_two_wheeled
        if candidate_value[6] == 0:
            pass
        elif candidate_value[6] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='reverse', side='both')
            if len(select_lane) > 0:
                pending_bicycle_lanes.append(random.choice(select_lane))

        speed = candidate_value[7]
        trigger_time = candidate_value[8]
        # generate participants
        # vehicle
        existing_actor_configs = []
        vehicle_configs = []
        npc_id = 10000
        for v_lane_id in pending_vehicle_lanes:
            new_agent = ApolloSimTool.create_new_follow_lane_waypoint_vehicle(npc_id, [v_lane_id])
            new_agent.trigger_time = trigger_time
            for item in new_agent.behavior:
                item.speed = speed
            if new_agent is not None:
                if not ApolloSimTool.conflict_checker(existing_actor_configs, new_agent):
                    vehicle_configs.append(copy.deepcopy(new_agent))
                    existing_actor_configs.append(copy.deepcopy(new_agent))
                    npc_id += 1

        for b_lane_id in pending_bicycle_lanes:
            new_agent = ApolloSimTool.create_new_follow_lane_waypoint_vehicle(npc_id, [b_lane_id], category="bicycle.normal")
            new_agent.trigger_time = trigger_time
            for item in new_agent.behavior:
                item.speed = speed
            if new_agent is not None:
                if not ApolloSimTool.conflict_checker(existing_actor_configs, new_agent):
                    vehicle_configs.append(copy.deepcopy(new_agent))
                    existing_actor_configs.append(copy.deepcopy(new_agent))
                    npc_id += 1

        logger.info('Add {} vehicles', len(vehicle_configs))

        m_end_time = time.time()
        logger.info('Mutation Spend Time: [=]{}[=]', m_end_time - m_start_time)

        # create traffic light part
        traffic_light_config = ApolloSimTool.create_traffic_light()

        return ApolloSimScenarioConfig(
            idx=0,
            ego_config_pool=ApolloConfigPool(ego_configs),
            vehicle_config_pool=VehicleConfigPool(vehicle_configs),
            walker_config_pool=WalkerConfigPool([]),
            static_obstacle_config_pool=StaticObstacleConfigPool([]),
            traffic_light_config=traffic_light_config
        )

    def feedback_adaptor(self, result: ScenarioRecorder):
        # calculate fitness
        feedback = result.scenario_feedback[0]
        # min is better
        fitness = [
            feedback['oracle.collision'],
        ]
        threshold = self.oracle_cfg['destination']['threshold']
        fitness.append(float(max((threshold - feedback['oracle.destination']) / threshold, 0.0)))

        # is complete
        scenario_result = result.scenario_overview[0]
        if 'COMPLETE' in scenario_result:
            is_complete = True
        else:
            is_complete = False
        return fitness, is_complete