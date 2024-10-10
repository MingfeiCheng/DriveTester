def get_basic_config():
    return {
        "max_speed": 25.0,
        "max_speed_junction": 10.0,
        "max_acceleration": 6.0,
        "max_deceleration": -6.0,
        "max_steering": 0.8,
        "collision_threshold": 5.0,
        "ignore_vehicle": False,
        "ignore_walker": False,
        "ignore_static_obstacle": False,
        "ignore_traffic_light": False,
        "min_distance": 2.0, # to filter next waypoint
        "finish_buffer": 20.0,
        "collision_distance_threshold": 5.0,
        "pid_lateral_cfg": {
            'K_P': 1.25,
            'K_D': 0.3,
            'K_I': 0.75
        },
        "pid_longitudinal_cfg": {
            'K_P': 5.0,
            'K_D': 1.0,
            'K_I': 0.5
        },
        "remove_after_finish" : False
    }