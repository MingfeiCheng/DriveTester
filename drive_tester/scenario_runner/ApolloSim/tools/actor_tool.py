import random

from typing import Optional, List

from apollo_sim.map import Map
from apollo_sim.registry import ACTOR_REGISTRY
from apollo_sim.actor import Waypoint, Lane, Location
from drive_tester.scenario_runner.ApolloSim.config.scenario import (ApolloConfig,
                                                                    WaypointVehicleConfig,
                                                                    WaypointWalkerConfig,
                                                                    StaticObstacleConfig,
                                                                    RuleLightConfig)
from drive_tester.scenario_runner.ApolloSim.tools.map_tool import MapTool


class ActorTool:

    @staticmethod
    def conflict_checker(existing_actor_configs, new_actor_config, max_distance=100) -> bool:
        actor_class = ACTOR_REGISTRY.get(new_actor_config.category)
        new_actor = actor_class(
            id=new_actor_config.idx,
            location=new_actor_config.location,
            role=new_actor_config.role
        )
        for existing_actor_config in existing_actor_configs:
            actor_class = ACTOR_REGISTRY.get(existing_actor_config.category)
            actor = actor_class(
                id=existing_actor_config.idx,
                location=existing_actor_config.location,
                role=existing_actor_config.role
            )
            dist = actor.dist2actor(new_actor)
            if dist < 0.5 or dist > max_distance:
                return True
        return False

    @staticmethod
    def create_apollo(sim_map: 'Map', idx: int, ego_start_lane_id: str, ego_end_lane_id: str, start_s: float = 0.5, end_s: float = 0.5) -> ApolloConfig:

        start_lane_length = sim_map.lane.get_length(ego_start_lane_id)
        start_pos_x, start_pos_y, start_yaw = sim_map.lane.get_coordinate(ego_start_lane_id, s=start_s, l=0.0)
        start_wp = Waypoint(
            lane=Lane(ego_start_lane_id, s=random.uniform(0.0, start_lane_length / 2.0), l=0.0),
            location=Location(x=start_pos_x, y=start_pos_y, z=0.0, pitch=0.0, yaw=start_yaw, roll=0.0),
            speed=0.0
        )

        end_lane_length = sim_map.lane.get_length(ego_end_lane_id)
        end_pos_x, end_pos_y, end_yaw = sim_map.lane.get_coordinate(ego_end_lane_id, s=random.uniform(end_lane_length/2.0, end_lane_length), l=0.0)
        end_wp = Waypoint(
            lane=Lane(ego_end_lane_id, s=end_lane_length-end_s, l=0.0),
            location=Location(x=end_pos_x, y=end_pos_y, z=0.0, pitch=0.0, yaw=end_yaw, roll=0.0),
            speed=0.0
        )

        return ApolloConfig(
            idx=idx,
            category='vehicle.lincoln.mkz',
            route=[
                start_wp,
                end_wp
            ],
            trigger_time=random.uniform(0, 10),
            role='ads'
        )

    @staticmethod
    def create_random_rule_light() -> RuleLightConfig:
        green_time = random.uniform(1, 5)
        red_time = random.uniform(1, 5)
        yellow_time = random.uniform(1, 3)
        initial_seed = random.randint(0, 10000)
        force_green = random.choice([True, False])
        return RuleLightConfig(
            green_time=green_time,
            yellow_time=yellow_time,
            red_time=red_time,
            initial_seed=initial_seed,
            force_green=force_green
        )

    @staticmethod
    def create_random_waypoint_vehicle(sim_map: 'Map', idx: int, map_region_lanes: List[str]) -> Optional[WaypointVehicleConfig]:
        filtered_actors = ACTOR_REGISTRY.filter_actors('vehicle.*')
        if not filtered_actors:
            raise ValueError('No actors match the pattern, Please check')
        actor_category = random.choice(list(filtered_actors.keys()))

        # Select vehicle start lane
        start_lane = random.choice(list(map_region_lanes))
        vehicle_route = MapTool.sample_random_route(sim_map, start_lane, 10)

        # obtain waypoints for the vehicle route
        behavior_waypoints = MapTool.sample_vehicle_following_waypoint_behavior(
            sim_map,
            vehicle_route
        )

        return WaypointVehicleConfig(
            idx=idx,
            category=actor_category,
            behavior=behavior_waypoints,
            trigger_time=random.uniform(0, 10),
            role='vehicle'
        )


    @staticmethod
    def create_random_waypoint_walker(sim_map: 'Map', idx: int, ego_route_lanes: List[str]) -> Optional[WaypointWalkerConfig]:
        filtered_actors = ACTOR_REGISTRY.filter_actors('walker.*')
        if not filtered_actors:
            raise ValueError('No actors match the pattern, Please check')
        actor_category = random.choice(list(filtered_actors.keys()))

        cross_lane = random.choice(ego_route_lanes)
        behavior = MapTool.sample_walker_crossing_lane_waypoint_behavior(
            sim_map,
            cross_lane
        )

        return WaypointWalkerConfig(
            idx=idx,
            category=actor_category,
            behavior=behavior,
            trigger_time=random.uniform(0, 10),
            role='walker'
        )

    @staticmethod
    def create_random_static_obstacle(sim_map: 'Map', idx: int, map_region_lanes: List[str]) -> Optional[StaticObstacleConfig]:
        filtered_actors = ACTOR_REGISTRY.filter_actors('static.*')
        if not filtered_actors:
            raise ValueError('No actors match the pattern, Please check')
        actor_category = random.choice(list(filtered_actors.keys()))

        surrounding_lane = random.choice(map_region_lanes)
        waypoint = MapTool.sample_static_obstacle_waypoint(sim_map, surrounding_lane)

        return StaticObstacleConfig(
            idx=idx,
            category=actor_category,
            waypoint=waypoint,
            role='static'
        )
