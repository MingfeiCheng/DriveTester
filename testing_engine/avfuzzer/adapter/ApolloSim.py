import copy
import random
import time

import numpy as np

from loguru import logger
from typing import Optional
from common.data_provider import DataProvider
from scenario_runner.drive_simulator.ApolloSim.config import VehicleConfigPool, WalkerConfigPool, \
    StaticObstacleConfigPool, ApolloConfigPool, VehicleConfig, WalkerConfig
from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder
from scenario_runner.drive_simulator.ApolloSim.map.apollo_map_parser import MapParser
from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig
from testing_engine.scenario_tool import ApolloSim as ApolloSimTool
from typing import List

class ScenarioAdaptor:

    def __init__(
            self,
            scenario_config: ApolloSimScenarioConfig,
            fitness: float
    ):
        self.id = scenario_config.idx
        self.scenario = scenario_config
        self.fitness = fitness

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
            oracle_cfg: list
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
            population: Optional[List[ScenarioAdaptor]],
            pc: float = 0.6,
            pm: float = 0.6,
    ) -> List[ScenarioAdaptor] or ApolloSimScenarioConfig:
        # generate
        if population is None:
            scenario_config = self.create_new_scenario_config()
            return scenario_config
        else:
            pop_size = len(population)

            m_start_time = time.time()

            new_pop_lst = []
            for i in range(pop_size):
                pop_i = copy.deepcopy(population[i])
                if random.random() < pc:
                    # crossover
                    j = 0
                    while i == j:
                        j = random.randint(0, pop_size - 1)
                    pop_j = copy.deepcopy(population[j])
                    pop_i = self._crossover(pop_i, pop_j)

                if random.random() < pm:
                    pop_i = self._mutation(pop_i)
                new_pop_lst.append(pop_i)
            m_end_time = time.time()
            logger.error('Mutation Spend Time: [=]{}[=]', m_end_time - m_start_time)
            return new_pop_lst

    def feedback_adaptor(self, result: ScenarioRecorder):
        # calculate fitness
        feedback = result.scenario_feedback[0]
        # min is better
        fitness = []
        for key, value in feedback.items():
            if key == 'oracle.collision':
                fitness.append(value)

        fitness = sum(fitness) / len(fitness)

        # is complete
        scenario_result = result.scenario_overview[0]
        if 'COLLISION' in scenario_result:
            is_complete = False
        else:
            is_complete = True

        return fitness, is_complete

    # other tools
    def _mutation(self, seed: ScenarioAdaptor) -> ScenarioAdaptor:

        vds: List[VehicleConfig] = seed.scenario.vehicle_config_pool.configs
        for i in range(len(vds)):
            if random.random() > 0.5:
                continue
            vd_i = vds[i]
            # trigger time
            vd_i.trigger_time = float(np.clip(random.gauss(vd_i.trigger_time, 0.5), 0.0, 8.0))

            # speed
            route_length = len(vd_i.behavior)
            mutate_speed_interval = 10
            # mutated_indexes = random.sample(range(len(vd_i.route)), min(len(vd_i.route), 5))

            for m_id_ in range(int(route_length / mutate_speed_interval)):
                m_id = m_id_ * 10
                m_id = min(m_id, route_length - 1)
                prev_speed = vd_i.behavior[m_id].speed
                mutated_speed = float(np.clip(random.gauss(prev_speed, 2), 0.3, 20))
                vd_i.behavior[m_id].speed = mutated_speed
            vds[i] = vd_i
        seed.scenario.vehicle_config_pool.configs = vds

        wds = seed.scenario.walker_config_pool.configs
        for i in range(len(wds)):
            if random.random() > 0.5:
                continue

            wd_i = wds[i]
            wd_i.trigger_time = float(np.clip(random.gauss(wd_i.trigger_time, 0.5), 0.0, 8.0))

            for m_id_ in range(len(wd_i.behavior)):
                prev_speed = wd_i.behavior[m_id_].speed
                mutated_speed = float(np.clip(random.gauss(prev_speed, 2), 0.1, 10))
                wd_i.behavior[m_id_].speed = mutated_speed
            wds[i] = wd_i
        seed.scenario.walker_config_pool.configs = wds

        return seed

    def _crossover(self, seed_i: ScenarioAdaptor, seed_j: ScenarioAdaptor):
        # change vehicles
        vds_i = seed_i.scenario.vehicle_config_pool.configs
        vds_j = seed_j.scenario.vehicle_config_pool.configs
        for i in range(len(vds_i)):
            if random.random() < 0.5:
                try:
                    vd_ii: VehicleConfig = vds_i[i]
                    vd_ji: VehicleConfig = vds_j[i]
                    tmp = copy.deepcopy(vd_ii)
                    vd_ii = copy.deepcopy(vd_ji)
                    vd_ji = tmp

                    vds_i[i] = vd_ii
                    vds_j[i] = vd_ji
                except Exception:
                    continue

        seed_i.scenario.vehicle_config_pool.configs = vds_i
        seed_j.scenario.vehicle_config_pool.configs = vds_j

        # change walkers
        wds_i = seed_i.scenario.walker_config_pool.configs
        wds_j = seed_j.scenario.walker_config_pool.configs
        for i in range(len(wds_i)):
            if random.random() < 0.5:
                try:
                    wd_ii: WalkerConfig = wds_i[i]
                    wd_ji: WalkerConfig = wds_j[i]
                    tmp = copy.deepcopy(wd_ii)
                    wd_ii = copy.deepcopy(wd_ji)
                    wd_ji = tmp

                    wds_i[i] = wd_ii
                    wds_j[i] = wd_ji
                except Exception:
                    continue
        seed_i.scenario.walker_config_pool.configs = wds_i
        seed_j.scenario.walker_config_pool.configs = wds_j

        return seed_i

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