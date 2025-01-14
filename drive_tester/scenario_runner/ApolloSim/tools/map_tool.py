import math
import random

from loguru import logger
from typing import List, Dict

from apollo_sim.map import Map
from apollo_sim.actor import Waypoint, Lane, Location

class MapTool:

    @staticmethod
    def estimate_map_region(
            sim_map: 'Map',
            start_lane_id: str,
            end_lane_id: str
    ) -> List[str]:

        # Get the direct path of lanes
        ego_potential_trace = sim_map.lane.find_path(start_lane_id, end_lane_id)

        # Collect unique junction IDs from the potential trace
        junction_lst = []
        for lane_id in ego_potential_trace:
            junction_id = sim_map.junction.get_by_lane_id(lane_id)
            if junction_id is not None and junction_id not in junction_lst:
                junction_lst.append(junction_id)

        # Collect potential lanes using a set for uniqueness
        potential_lane_ids = set()

        # Process lanes from each junction
        for junction_id in junction_lst:
            junction_lanes = sim_map.junction.get_lane_ids(junction_id)
            potential_lane_ids.update(junction_lanes)  # Add lanes in the junction

            # Add lanes within 2 steps of each lane in the junction
            for lane_id in junction_lanes:
                # append successor lanes + predecessor lanes
                k_step_successor_lanes = sim_map.lane.get_successor_id(lane_id, 2)
                k_step_predecessor_lanes = sim_map.lane.get_predecessor_id(lane_id, 2)
                potential_lane_ids.update(k_step_successor_lanes)
                potential_lane_ids.update(k_step_predecessor_lanes)

        # Process lanes from the ego trace
        for lane_id in ego_potential_trace:
            # Add lanes within 4 steps of each lane in the trace
            # forward lanes + reverse lanes
            k_step_neighbor_forward_lanes = sim_map.lane.get_neighbor_forward_lane_id(lane_id, 4)
            k_step_neighbor_reverse_lanes = sim_map.lane.get_neighbor_reverse_lane_id(lane_id, 4)
            potential_lane_ids.update(k_step_neighbor_forward_lanes)
            potential_lane_ids.update(k_step_neighbor_reverse_lanes)

        # Convert the set back to a list
        potential_lane_ids = list(potential_lane_ids)
        return ego_potential_trace, potential_lane_ids

    @staticmethod
    def sample_random_route(
            sim_map: 'Map',
            start_lane: str,
            depth: int = 10,
            consider_neighbor: bool = False
    ) -> List[str]:
        # TODO: improve this
        path_lane_sequence = [start_lane]
        for i in range(depth):
            last_lane = path_lane_sequence[-1]
            # next_lane_pool
            next_lane_successor = sim_map.lane.get_successor_id(last_lane, 1)
            if consider_neighbor:
                next_lane_successor += sim_map.lane.get_neighbor_forward_lane_id(last_lane, 1)
                next_lane_successor += sim_map.lane.get_neighbor_reverse_lane_id(last_lane, 1)
            # next_lane_neighbor_forward = sim_map.lane.get_neighbor_forward_lane_id(last_lane, 1)
            # next_lane_neighbor_reverse = sim_map.lane.get_neighbor_reverse_lane_id(last_lane, 1)
            if len(next_lane_successor) > 0:
                next_lane = random.choice(next_lane_successor)
                if next_lane not in path_lane_sequence:
                    path_lane_sequence.append(next_lane)
            else:
                break

        return path_lane_sequence

    @staticmethod
    def sample_lane_waypoints(
            sim_map: 'Map',
            lane_id: str,
            start_s: float,
            end_s: float,
            speed: float,
            s_interval: float = 2.0
    ) -> List[Waypoint]:

        waypoints = []
        s = start_s
        while s < end_s:
            x, y, heading = sim_map.lane.get_coordinate(lane_id, s, 0.0)
            waypoints.append(Waypoint(
                lane=Lane(lane_id, s, l=0.0),
                location=Location(
                    x=x,
                    y=y,
                    z=0.0,
                    pitch=0.0,
                    yaw=heading,
                    roll=0.0
                ),
                speed=speed
            ))
            s += s_interval
        return waypoints

    @staticmethod
    def sample_vehicle_following_waypoint_behavior(
            sim_map: 'Map',
            vehicle_route: List[str]
    ) -> List[Waypoint]:
        # TODO: improve this for scenario generation

        # parameters
        vehicle_min_speed = 0.2
        vehicle_max_speed = 20.0

        # obtain waypoints for the vehicle route
        behavior_waypoints = []
        for node in vehicle_route:

            start_s = 0.0
            end_s = sim_map.lane.get_length(node)

            lane_max_speed = min(vehicle_max_speed, sim_map.lane.get_speed_limit(node))
            sample_speed = random.uniform(vehicle_min_speed, lane_max_speed)

            # get waypoint
            waypoints = MapTool.sample_lane_waypoints(
                sim_map,
                node,
                start_s,
                end_s,
                sample_speed,
                2.0
            )
            behavior_waypoints += waypoints

        behavior_waypoints[-1].speed = 0.0

        return behavior_waypoints

    @staticmethod
    def sample_walker_crossing_lane_waypoint_behavior(
            sim_map: 'Map',
            crossing_lane: str
    ) -> List[Waypoint]:

        # parameters
        walker_min_speed = 0.1
        walker_max_speed = 10.0

        crossing_lane_length = sim_map.lane.get_length(crossing_lane)
        crossing_s = random.uniform(0.0, crossing_lane_length)

        # sample two waypoints
        l1 = random.uniform(5.0, 20.0)
        l2 = random.uniform(-20.0, -5.0)

        x1, y1, heading1 = sim_map.lane.get_coordinate(
            crossing_lane,
            s=crossing_s,
            l=l1
        )
        x2, y2, heading2 = sim_map.lane.get_coordinate(
            crossing_lane,
            s=crossing_s,
            l=l2
        )

        wp1 = Waypoint(
            lane=Lane(crossing_lane, crossing_s, l=l1),
            location=Location(
                x=x1,
                y=y1,
                z=0.0,
                pitch=0.0,
                yaw=heading1,
                roll=0.0
            ),
            speed=random.uniform(walker_min_speed, walker_max_speed)
        )

        wp2 = Waypoint(
            lane=Lane(crossing_lane, crossing_s, l=l2),
            location=Location(
                x=x2,
                y=y2,
                z=0.0,
                pitch=0.0,
                yaw=heading2,
                roll=0.0
            ),
            speed=random.uniform(walker_min_speed, walker_max_speed)
        )

        if random.random() > 0.5:
            start_wp = wp1
            end_wp = wp2
            angle = math.atan2(y2 - y1, x2 - x1)
            heading = (angle + math.pi) % (2 * math.pi) - math.pi
            start_wp.location.yaw = heading
            end_wp.location.yaw = heading
        else:
            start_wp = wp2
            end_wp = wp1
            angle = math.atan2(y1 - y2, x1 - x2)
            heading = (angle + math.pi) % (2 * math.pi) - math.pi
            start_wp.location.yaw = heading
            end_wp.location.yaw = heading

        return [start_wp, end_wp]

    @staticmethod
    def sample_static_obstacle_waypoint(
            sim_map: 'Map',
            surrounding_lane: str
    ) -> Waypoint:
        surrounding_lane_length = sim_map.lane.get_length(surrounding_lane)

        s = random.uniform(0.0, surrounding_lane_length)
        l = random.uniform(-5.0, 5.0)
        x, y, heading = sim_map.lane.get_coordinate(
            surrounding_lane,
            s=s,
            l=l
        )

        return Waypoint(
            lane=Lane(surrounding_lane, s, l=l),
            location=Location(
                x=x,
                y=y,
                z=0.0,
                pitch=0.0,
                yaw=heading,
                roll=0.0
            ),
            speed=0.0
        )