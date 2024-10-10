def get_basic_config():
    return {
        "max_speed": 6.0,
        "max_speed_junction": 6.0,
        "max_acceleration": 3.0,
        "max_deceleration": -3.0,
        "collision_threshold": 5.0,
        "ignore_vehicle": False,
        "ignore_walker": False,
        "ignore_static_obstacle": False,
        "ignore_traffic_light": False,
        "min_distance": 1.0, # to filter next waypoint
        "finish_buffer": 20.0,
        "collision_distance_threshold": 5.0,
        "remove_after_finish" : False
    }