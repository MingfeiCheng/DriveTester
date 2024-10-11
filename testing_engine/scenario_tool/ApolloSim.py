import copy
import random
import math
import numpy as np

from scenario_runner.drive_simulator.ApolloSim.config import *
from scenario_runner.drive_simulator.ApolloSim.library import agent_library

NPC_MIN_SPEED = 0.3
NPC_MAX_SPEED = 20.0
NPC_MIN_SPEED_WALKER = 0.1
NPC_MAX_SPEED_WALKER = 11.0

def conflict_checker(existing_actor_configs: List[BaseConfigClass], new_actor_config: BaseConfigClass):
    actor_class = agent_library.get(new_actor_config.category)
    new_actor = actor_class(
        new_actor_config.idx,
        new_actor_config.initial_waypoint.location,
        new_actor_config.role
    )
    for existing_actor_config in existing_actor_configs:
        # def __init__(self, idx: int, location: LocationConfig, role: str = 'default'):
        actor_class = agent_library.get(existing_actor_config.category)
        actor = actor_class(
            existing_actor_config.idx,
            existing_actor_config.initial_waypoint.location,
            existing_actor_config.role
        )
        if actor.bbox_distance(new_actor) < 0.5:
            return True
    return False

def create_new_follow_lane_waypoint_vehicle(npc_id: int, potential_lanes: List[str], category='vehicle.lincoln.mkz') -> VehicleConfig:
    # add lane changing and sample key waypoints

    # Select vehicle start lane
    start_lane = random.choice(list(potential_lanes))
    vehicle_route_lanes = DataProvider.map_parser.get_random_route_from_start_lane(start_lane, 10)

    # Get waypoint for start lane
    start_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(start_lane, 2)
    if not start_s_lst:
        return None

    start_s_index = random.choice(range(len(start_s_lst) - 1))
    start_s_lst = start_s_lst[start_s_index:]

    # Prepare the route starting points
    lane_speed = random.uniform(NPC_MIN_SPEED, NPC_MAX_SPEED)
    route = [
        WaypointConfig(
            lane=LaneConfig(lane_id=start_lane, s=s, l=0.0),
            location=LocationConfig(
                x=point.x, y=point.y, z=point.z, pitch=0.0, yaw=heading, roll=0.0
            ),
            speed=(0.0 if i == 0 else lane_speed),
            is_junction=DataProvider.map_parser.is_junction_lane(start_lane)
        )
        for i, s in enumerate(start_s_lst)
        for point, heading in [DataProvider.map_parser.get_coordinate_and_heading(start_lane, s)]
    ]

    # Extend the route for mid-lane waypoints
    for lane_id in vehicle_route_lanes[1:]:
        lane_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(lane_id, 2.0)
        lane_speed = random.uniform(NPC_MIN_SPEED, NPC_MAX_SPEED)
        route.extend(
            WaypointConfig(
                lane=LaneConfig(lane_id=lane_id, s=s, l=0.0),
                location=LocationConfig(
                    x=point.x, y=point.y, z=point.z, pitch=0.0, yaw=heading, roll=0.0
                ),
                speed=lane_speed,
                is_junction=DataProvider.map_parser.is_junction_lane(lane_id)
            )
            for s in lane_s_lst
            for point, heading in [DataProvider.map_parser.get_coordinate_and_heading(lane_id, s)]
        )

    # Set last waypoint speed to 0
    route[-1].speed = 0.0

    # Create vehicle configuration
    vd_trigger = random.uniform(0.0, 5.0)
    agent_config = VehicleConfig(
        idx=npc_id,
        category=category,
        initial_waypoint=route[0],
        mutable=True,
        trigger_time=vd_trigger,
        behavior=route,
        role='vehicle'
    )
    return agent_config

def create_new_stationary_vehicle(npc_id: int, potential_lanes: List[str], category='vehicle.lincoln.mkz') -> VehicleConfig:
    # add lane changing and sample key waypoints

    # Select vehicle start lane
    start_lane = random.choice(list(potential_lanes))
    vehicle_route_lanes = DataProvider.map_parser.get_random_route_from_start_lane(start_lane, 10)

    # Get waypoint for start lane
    start_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(start_lane, 2)
    if not start_s_lst:
        return None

    start_s_index = random.choice(range(len(start_s_lst) - 1))
    start_s_lst = start_s_lst[start_s_index:]

    # Prepare the route starting points
    lane_speed = 0.0
    route = [
        WaypointConfig(
            lane=LaneConfig(lane_id=start_lane, s=s, l=0.0),
            location=LocationConfig(
                x=point.x, y=point.y, z=point.z, pitch=0.0, yaw=heading, roll=0.0
            ),
            speed=(0.0 if i == 0 else lane_speed),
            is_junction=DataProvider.map_parser.is_junction_lane(start_lane)
        )
        for i, s in enumerate(start_s_lst)
        for point, heading in [DataProvider.map_parser.get_coordinate_and_heading(start_lane, s)]
    ]

    # Extend the route for mid-lane waypoints
    for lane_id in vehicle_route_lanes[1:]:
        lane_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(lane_id, 2.0)
        lane_speed = 0.0
        route.extend(
            WaypointConfig(
                lane=LaneConfig(lane_id=lane_id, s=s, l=0.0),
                location=LocationConfig(
                    x=point.x, y=point.y, z=point.z, pitch=0.0, yaw=heading, roll=0.0
                ),
                speed=lane_speed,
                is_junction=DataProvider.map_parser.is_junction_lane(lane_id)
            )
            for s in lane_s_lst
            for point, heading in [DataProvider.map_parser.get_coordinate_and_heading(lane_id, s)]
        )

    # Set last waypoint speed to 0
    route[-1].speed = 0.0

    # Create vehicle configuration
    vd_trigger = random.uniform(0.0, 5.0)
    agent_config = VehicleConfig(
        idx=npc_id,
        category=category,
        initial_waypoint=route[0],
        mutable=True,
        trigger_time=vd_trigger,
        behavior=route,
        role='vehicle'
    )
    return agent_config

def create_new_linear_vehicle(npc_id: int, potential_lanes: List[str], category='vehicle.lincoln.mkz') -> VehicleConfig:
    # add lane changing and sample key waypoints

    # Select vehicle start lane
    start_lane = random.choice(list(potential_lanes))
    vehicle_route_lanes = DataProvider.map_parser.get_random_route_from_start_lane(start_lane, 10)

    # Get waypoint for start lane
    start_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(start_lane, 2)
    if not start_s_lst:
        return None

    start_s_index = random.choice(range(len(start_s_lst) - 1))
    start_s_lst = start_s_lst[start_s_index:]

    # Prepare the route starting points
    lane_speed = random.uniform(NPC_MIN_SPEED, NPC_MAX_SPEED)
    route = [
        WaypointConfig(
            lane=LaneConfig(lane_id=start_lane, s=s, l=0.0),
            location=LocationConfig(
                x=point.x, y=point.y, z=point.z, pitch=0.0, yaw=heading, roll=0.0
            ),
            speed=(0.0 if i == 0 else lane_speed),
            is_junction=DataProvider.map_parser.is_junction_lane(start_lane)
        )
        for i, s in enumerate(start_s_lst)
        for point, heading in [DataProvider.map_parser.get_coordinate_and_heading(start_lane, s)]
    ]

    # Extend the route for mid-lane waypoints
    for lane_id in vehicle_route_lanes[1:]:
        lane_s_lst = DataProvider.map_parser.get_waypoint_s_for_lane(lane_id, 2.0)
        lane_speed = random.uniform(NPC_MIN_SPEED, NPC_MAX_SPEED)
        route.extend(
            WaypointConfig(
                lane=LaneConfig(lane_id=lane_id, s=s, l=0.0),
                location=LocationConfig(
                    x=point.x, y=point.y, z=point.z, pitch=0.0, yaw=heading, roll=0.0
                ),
                speed=lane_speed,
                is_junction=DataProvider.map_parser.is_junction_lane(lane_id)
            )
            for s in lane_s_lst
            for point, heading in [DataProvider.map_parser.get_coordinate_and_heading(lane_id, s)]
        )

    # Create vehicle configuration
    vd_trigger = random.uniform(0.0, 5.0)
    agent_config = VehicleConfig(
        idx=npc_id,
        category=category,
        initial_waypoint=route[0],
        mutable=True,
        trigger_time=vd_trigger,
        behavior=[route[0], route[-1]],
        role='vehicle'
    )
    return agent_config

def create_new_waypoint_walker(npc_id: int, ego_trace: List[str]) -> WalkerConfig:
    cross_lane = random.choice(ego_trace)
    cross_lane_length = DataProvider.map_parser.get_lane_length(cross_lane)

    speed = random.uniform(0.1, 10.0)
    s = random.uniform(0.0, cross_lane_length)
    l1 = random.uniform(5.0, 20.0)
    x1, y1, heading1 = DataProvider.map_parser.get_coordinate(
        cross_lane,
        s=s,
        l=l1
    )
    wp1 = WaypointConfig(
        lane=LaneConfig(
            lane_id=cross_lane,
            s=s,
            l=l1
        ),
        location=LocationConfig(
            x=x1,
            y=y1,
            z=0.0,
            pitch=0.0,
            yaw=heading1,
            roll=0.0
        ),
        speed=speed,
    )

    l2 = random.uniform(-20.0, -5.0)
    x2, y2, heading2 = DataProvider.map_parser.get_coordinate(
        cross_lane,
        s=s,
        l=l2
    )
    wp2 = WaypointConfig(
        lane=LaneConfig(
            lane_id=cross_lane,
            s=s,
            l=l2
        ),
        location=LocationConfig(
            x=x2,
            y=y2,
            z=0.0,
            pitch=0.0,
            yaw=heading2,
            roll=0.0
        ),
        speed=speed,
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

    walker_config = WalkerConfig(
        idx=npc_id,
        category='walker.pedestrian.normal',
        initial_waypoint=start_wp,
        mutable=True,
        trigger_time=random.uniform(0.0, 5.0),
        behavior=[start_wp, end_wp],
        role='walker'
    )
    return walker_config

def create_new_static_obstacle(npc_id: int, potential_lanes: List[str]) -> StaticObstacleConfig:
    cross_lane = random.choice(potential_lanes)
    cross_lane_length = DataProvider.map_parser.get_lane_length(cross_lane)

    s = random.uniform(0.0, cross_lane_length)
    l = random.uniform(-5.0, 5.0)
    x, y, heading = DataProvider.map_parser.get_coordinate(
        cross_lane,
        s=s,
        l=l
    )

    static_config = StaticObstacleConfig(
        idx=npc_id,
        category="static.traffic_cone",
        initial_waypoint=WaypointConfig(
            lane=LaneConfig(
                lane_id=cross_lane,
                s=s,
                l=l
            ),
            location=LocationConfig(
                x=x,
                y=y,
                z=0.0,
                pitch=0.0,
                yaw=heading,
                roll=0.0
            ),
            speed=0.0,
        ),
        mutable=True,
        trigger_time=0.0,
        role='static'
    )
    return static_config

def create_traffic_light():
    return TrafficLightConfig()

def create_new_apollo(
        ego_id: str,
        ego_start_lane_id: str,
        ego_end_lane_id: str,
        termination_on_failure: bool = True,
        mutable: bool = False,
) -> ApolloConfig:
    start_lane_id = ego_start_lane_id
    if not DataProvider.map_parser.is_driving_lane(start_lane_id):
        raise RuntimeError('Not driving lane')

    start_s = 5.0
    end_lane_s = 5.0

    start_point, start_heading = DataProvider.map_parser.get_coordinate_and_heading(start_lane_id, start_s)

    start_waypoint = WaypointConfig(
        location=LocationConfig(
            x=start_point.x,
            y=start_point.y,
            z=start_point.z,  # default is 0?
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
    end_lane_id = ego_end_lane_id
    end_lane_length = DataProvider.map_parser.get_lane_length(end_lane_id)
    end_s = float(np.clip(end_lane_length - end_lane_s, 0.0, end_lane_length))
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

    return ApolloConfig(
        idx=ego_id,
        category='vehicle.lincoln.mkz',
        initial_waypoint=sample_route[0],
        route=sample_route,
        trigger_time=0.0,
        mutable=mutable,
        termination_on_failure=termination_on_failure,
        role='apollo'
    )


def mutate_vehicle_speed(vehicle_config: VehicleConfig):
    # only for waypoint vehicle
    original_behavior = copy.deepcopy(vehicle_config.behavior)
    for i, waypoint in enumerate(original_behavior):
        waypoint.speed = float(np.clip(random.gauss(waypoint.speed, 2.0), NPC_MIN_SPEED, NPC_MAX_SPEED))
        original_behavior[i] = waypoint
    vehicle_config.behavior = original_behavior
    vehicle_config.trigger_time = float(np.clip(random.gauss(vehicle_config.trigger_time, 0.5), 0.0, 8.0))
    return vehicle_config

def mutate_walker_speed(walker_config: WalkerConfig):
    # only for waypoint vehicle
    original_behavior = copy.deepcopy(walker_config.behavior)
    for i, waypoint in enumerate(original_behavior):
        waypoint.speed = float(np.clip(random.gauss(waypoint.speed, 2.0), NPC_MIN_SPEED_WALKER, NPC_MAX_SPEED_WALKER))
        original_behavior[i] = waypoint
    walker_config.behavior = original_behavior
    walker_config.trigger_time = float(np.clip(random.gauss(walker_config.trigger_time, 0.5), 0.0, 8.0))
    return walker_config

def mutate_static_location(static_config: StaticObstacleConfig):
    static_config.initial_waypoint.location.x += random.uniform(0.0, 0.5)
    static_config.initial_waypoint.location.y += random.uniform(0.0, 0.5)
    return static_config