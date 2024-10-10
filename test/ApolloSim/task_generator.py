from __future__ import annotations

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import numpy as np

from typing import List, Optional
from loguru import logger

from common.map_parser import MapParser
from common.data_provider import DataProvider

from scenario_runner.configuration.scenario_config import ScenarioConfig
from scenario_runner.runner.scenario_runner import ScenarioRunner
from scenario_runner.drive_simulator.ApolloSim.config import WaypointConfig, LocationConfig, LaneConfig
from scenario_runner.drive_simulator.ApolloSim.config.apollo import ApolloConfig, ApolloConfigPool
from scenario_runner.drive_simulator.ApolloSim.config import TrafficLightConfig
from scenario_runner.drive_simulator.ApolloSim.agents.static_obstacle import StaticObstacleConfigPool
from scenario_runner.drive_simulator.ApolloSim.agents.vehicle import WaypointVehicleConfigPool
from scenario_runner.drive_simulator.ApolloSim import WaypointWalkerConfigPool


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

    def _generate_empty_scenario_config(self, route) -> 'ScenarioConfig':

        ego_config_pool = ApolloConfigPool(
            [
                ApolloConfig(
                    idx=0,
                    initial_waypoint=route[0],
                    route=route,
                    trigger_time=0.0,
                    termination_on_failure=True
                ),
            ]
        )

        static_obstacle_config_pool = StaticObstacleConfigPool([])
        # we need a default static obstacle for trigger ads moving
        waypoint_vehicle_config_pool = WaypointVehicleConfigPool([])
        waypoint_walker_config_pool = WaypointWalkerConfigPool([])
        traffic_light_config = TrafficLightConfig()

        scenario_cfg = ScenarioConfig(
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
        scenario_runner = ScenarioRunner(scenario_cfg)
        scenario_runner.run()

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
    parser.add_argument('--map_name', type=str, help='apollo map name', default='sunnyvale_big_loop')
    parser.add_argument('--start_lane_id', type=str, default='9813_1_-2')
    parser.add_argument('--start_lane_s', type=float, default=10.0)
    parser.add_argument('--end_lane_id', type=str, default='10690_1_-2')
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