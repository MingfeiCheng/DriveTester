from __future__ import annotations

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import copy
import random
import argparse
import numpy as np

from typing import List, Optional
from loguru import logger
from omegaconf import DictConfig

from common.data_provider import DataProvider

from scenario_runner.drive_simulator.ApolloSim.map.apollo_map_parser import MapParser
from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig
from scenario_runner.runner.ApolloSim_runner import ScenarioRunner
from scenario_runner.drive_simulator.ApolloSim.config import WaypointConfig, LocationConfig, LaneConfig
from scenario_runner.drive_simulator.ApolloSim.config import ApolloConfig, ApolloConfigPool
from scenario_runner.drive_simulator.ApolloSim.config import TrafficLightConfig
from scenario_runner.drive_simulator.ApolloSim.config import StaticObstacleConfigPool
from scenario_runner.drive_simulator.ApolloSim.config import VehicleConfigPool, VehicleConfig
from scenario_runner.drive_simulator.ApolloSim.config import WalkerConfigPool

class ScenarioGenerator:
    """
    This class only provides a simple scenario including the start position and end position for all ADSs.

    Normally, a scenario contains:
        (1) a major ego
        (2) others including ADS controls or RULE controls
    """
    def __init__(
            self,
            map_name: str,
            start_lane_id: str,
            start_lane_s: float,
            end_lane_id: str,
            end_lane_s: float,
            save_root: str,
            tag: str
    ):
        """
        expect_route_length: expected route length for the ego
        """
        DataProvider.map_name = map_name
        DataProvider.map_parser = MapParser()
        DataProvider.map_parser.load_from_pkl(map_name)

        self.start_lane_id = start_lane_id
        self.start_lane_s = start_lane_s
        self.end_lane_id = end_lane_id
        self.end_lane_s = end_lane_s
        self.save_root = save_root
        self.tag = tag # for the save name
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

        self._select_route_pool = list()

    def _generate_route(self) -> Optional[List[WaypointConfig]]:
        # select start lane & start waypoint
        # (1) is driving lane (2) is not junction lane (3) length must enough
        start_lane_id = self.start_lane_id
        if not DataProvider.map_parser.is_driving_lane(start_lane_id):
            raise RuntimeError('Not driving lane')

        start_s = self.start_lane_s
        start_point, start_heading = DataProvider.map_parser.get_coordinate_and_heading(start_lane_id, start_s)

        start_waypoint = WaypointConfig(
            location=LocationConfig(
                x=start_point.x,
                y=start_point.y,
                z=start_point.z, # default is 0?
                pitch=0.0,
                yaw=start_heading,
                roll=0.0
            ),
            lane=LaneConfig(
                lane_id=start_lane_id,
                s=start_s,
                l=0.0
            ),
            speed=0.0,
            is_junction=DataProvider.map_parser.is_junction_lane(start_lane_id),
        )

        sample_route = [start_waypoint]
        end_lane_id = self.end_lane_id
        end_lane_length = DataProvider.map_parser.get_lane_length(end_lane_id)
        end_s = float(np.clip(end_lane_length - self.end_lane_s, 0.0, end_lane_length))
        end_point, end_heading = DataProvider.map_parser.get_coordinate_and_heading(end_lane_id, end_s)
        end_waypoint = WaypointConfig(
            location=LocationConfig(
                x=end_point.x,
                y=end_point.y,
                z=end_point.z,  # default is 0?
                pitch=0.0,
                yaw=end_heading,
                roll=0.0
            ),
            lane=LaneConfig(
                lane_id=end_lane_id,
                s=end_s,
                l=0.0
            ),
            speed=0.0,
            is_junction=DataProvider.map_parser.is_junction_lane(end_lane_id),
        )
        sample_route.append(end_waypoint)
        return sample_route

    def _get_new_vehicle(self, ego_start_lane_id, npc_id: str):
        # add lane changing and sample key waypoints

        # get ego start lane
        ego_start_lane = ego_start_lane_id

        potential_lanes = DataProvider.map_parser.get_neighbors(ego_start_lane, direct='forward', side='both')
        potential_lanes += [ego_start_lane]
        tmp_potential_lanes = copy.deepcopy(potential_lanes)
        for lane_id in tmp_potential_lanes:
            potential_lanes += DataProvider.map_parser.get_predecessor_lanes(lane_id)
            potential_lanes += DataProvider.map_parser.get_successor_lanes(lane_id)

        tmp_potential_lanes = copy.deepcopy(potential_lanes)
        for lane_id in tmp_potential_lanes:
            # potential_lanes += self._ma.get_predecessor_lanes(lane_id)
            potential_lanes += DataProvider.map_parser.get_successor_lanes(lane_id)

        potential_lanes = list(set(potential_lanes))

        # select vehicle lanes
        new_id = npc_id
        start_lane = random.choice(potential_lanes)
        vehicle_route_lanes = DataProvider.map_parser.get_random_route_from_start_lane(start_lane, 10)

        start_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(start_lane, 2)
        start_s_index = None
        start_s_indexes = list(np.arange(0, len(start_s_lst) - 1))
        random.shuffle(start_s_indexes)
        for item in start_s_indexes:

            start_s_index = item
            break

        if start_s_index is None:
            return None

        start_s_lst = start_s_lst[start_s_index: ]
        route = list()
        # add start point
        lane_speed = random.uniform(0.5, 40)
        for i, s in enumerate(start_s_lst):
            if i == 0:
                waypoint_speed = 0.0
            else:
                waypoint_speed = lane_speed
            lane_id = start_lane
            point, heading = DataProvider.map_parser.get_coordinate_and_heading(lane_id, s)
            route.append(
                WaypointConfig(
                    lane=LaneConfig(
                        lane_id=lane_id,
                        s=s,
                        l=0.0
                    ),
                    location=LocationConfig(
                        x=point.x,
                        y=point.y,
                        z=point.z,
                        pitch=0.0,
                        yaw=heading,
                        roll=0.0
                    ),
                    speed=waypoint_speed,
                    is_junction=DataProvider.map_parser.is_junction_lane(lane_id)
            ))

        # for mid
        for lane_index, lane_id in enumerate(vehicle_route_lanes[1:]):
            lane_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(lane_id, 2.0)
            lane_speed = random.uniform(0.5, 40)
            for s_index, s in enumerate(lane_s_lst):
                waypoint_speed = lane_speed
                point, heading = DataProvider.map_parser.get_coordinate_and_heading(lane_id, s)
                route.append(WaypointConfig(
                    lane=LaneConfig(
                        lane_id=lane_id,
                        s=s,
                        l=0.0
                    ),
                    location=LocationConfig(
                        x=point.x,
                        y=point.y,
                        z=point.z,
                        pitch=0.0,
                        yaw=heading,
                        roll=0.0
                    ),
                    speed=waypoint_speed,
                    is_junction=DataProvider.map_parser.is_junction_lane(lane_id)
                ))

        route[-1].speed = 0.0
        vd_trigger = random.uniform(0.0, 5.0)
        agent_config = VehicleConfig(
            idx=npc_id,
            category='vehicle.lincoln.mkz',
            initial_waypoint=route[0],
            mutable=True,
            trigger_time=vd_trigger,
            behavior=route,
            role='vehicle'
        )
        return agent_config

    def _generate_empty_scenario_config(self, route) -> 'ApolloSimScenarioConfig':

        ego_config_pool = ApolloConfigPool(
            [
                ApolloConfig(
                    idx=0,
                    category='vehicle.lincoln.mkz',
                    initial_waypoint=route[0],
                    route=route,
                    trigger_time=0.0,
                    mutable=False,
                    termination_on_failure=True,
                    role='apollo'
                ),
            ]
        )

        static_obstacle_config_pool = StaticObstacleConfigPool([])
        # we need a default static obstacle for trigger ads moving
        waypoint_vehicle_config_pool = VehicleConfigPool([self._get_new_vehicle(route[0].lane.lane_id, 10001)])
        waypoint_walker_config_pool = WalkerConfigPool([])
        traffic_light_config = TrafficLightConfig()

        scenario_cfg = ApolloSimScenarioConfig(
            '0',
            ego_config_pool=ego_config_pool,
            static_obstacle_config_pool=static_obstacle_config_pool,
            waypoint_vehicle_config_pool=waypoint_vehicle_config_pool,
            waypoint_walker_config_pool=waypoint_walker_config_pool,
            traffic_light_config=traffic_light_config
        )
        return scenario_cfg


    def generate(self):
        logger.info(f'Generate route scenario configs from given lanes: start {self.start_lane_id} dest {self.end_lane_id}')
        route = self._generate_route()
        scenario_cfg = self._generate_empty_scenario_config(route)
        scenario_runner = ScenarioRunner(DictConfig(
            {
                "container_name": "test",
                "save_traffic_recording": True,
                "save_apollo_recording": True
            }
        ))
        scenario_runner.run(scenario_cfg)

if __name__ == '__main__':
    """
    map_name: str,
    start_lane_id: str,
    start_lane_s: float,
    end_lane_id: str,
    end_lane_s: float,
    save_root: str,
    tag: str
    start_lane_id: 9813_1_-2
    start_s: 10.0
    end_lane_id: 10690_1_-2
    end_s: 4.0
    add_traffic_cone: false
    traffic_cone_interval: 4.0
    traffic_cone_threshold: 8.0
    """
    parser = argparse.ArgumentParser(description='Script to generate initial task for testing')
    parser.add_argument('--map_name', type=str, help='apollo map name', default='borregas_ave') # sunnyvale_big_loop
    parser.add_argument('--start_lane_id', type=str, default='lane_31') # 9813_1_-2' lane_31
    parser.add_argument('--start_lane_s', type=float, default=10.0)
    parser.add_argument('--end_lane_id', type=str, default='lane_15') # 1544_1_-1 lane_15
    parser.add_argument('--end_lane_s', type=str, default=4.0)
    parser.add_argument('--save_root', type=str, default='/data/c/mingfeicheng/ApolloSim/v7.0/data/seeds/debug')
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--container_name', type=str, default='test')
    args = parser.parse_args()

    DataProvider.container_name = args.container_name

    sg = ScenarioGenerator(
        map_name=args.map_name,
        start_lane_id=args.start_lane_id,
        start_lane_s=args.start_lane_s,
        end_lane_id=args.end_lane_id,
        end_lane_s=args.end_lane_s,
        save_root=args.save_root,
        tag=args.tag
    )

    sg.generate()