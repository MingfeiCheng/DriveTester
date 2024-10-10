import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import copy
import random
import numpy as np

from common.map_parser import MapParser
from common.data_provider import DataProvider
from scenario_runner.drive_simulator.ApolloSim.config import WaypointConfig, LocationConfig, LaneConfig

# get ego start lane
ego_start_lane = '9813_1_-2'
DataProvider.map_name = "sunnyvale_big_loop"
DataProvider.map_parser = MapParser()
DataProvider.map_parser.load_from_pkl(DataProvider.map_name)

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
new_id = 1
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
    print(f'start_s_index is None')
    exit(-1)

start_s_lst = start_s_lst[start_s_index:]
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

x = []
y = []
for route_item in route:
    x.append(route_item.location.x)
    y.append(route_item.location.y)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
