import copy
import math
import os.path
import time

import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
from typing import Dict
from shapely.geometry import Point
from collections import deque

from common.data_provider import DataProvider
from common.logger_tools import get_instance_logger
from scenario_runner.drive_simulator.ApolloSim.library import AgentClass, agent_library, VehicleControl
from ..vehicle_controller import PIDController
from scenario_runner.drive_simulator.ApolloSim.config.vehicle import VehicleConfig, WaypointConfig
from .parameters import get_basic_config
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge

normalise_angle = lambda angle: math.atan2(math.sin(angle), math.cos(angle))

class CommandVehicle(object):

    MIN_DISTANCE_PERCENTAGE = 0.95

    def __init__(
            self,
            agent_config: VehicleConfig,
            traffic_bridge: TrafficBridge,
            parameters: Dict = get_basic_config(),
    ):
        self.agent_config = agent_config
        # obtain agent
        agent_class = agent_library.get(self.agent_config.category)
        self.agent = agent_class(
            idx=self.agent_config.idx,
            location=self.agent_config.initial_waypoint.location,
            role=self.agent_config.role
        )
        self.route = self.agent_config.route
        self.traffic_bridge = traffic_bridge

        self.debug = DataProvider.debug
        # 1. inner parameters' configuration
        self._ignore_vehicle = parameters.get('ignore_vehicle', False)
        self._ignore_walker = parameters.get('ignore_walker', False)
        self._ignore_static_obstacle = parameters.get('ignore_static_obstacle', False)
        self._ignore_traffic_light = parameters.get('ignore_traffic_light', False)
        self._max_speed = parameters.get('max_speed', 25.0)
        self._max_speed_junction = parameters.get('max_speed_junction', 10.0)
        self._min_distance = parameters.get('min_distance', 2.0) * self.MIN_DISTANCE_PERCENTAGE
        self._max_acceleration = parameters.get('max_acceleration', 6.0)
        self._max_deceleration = parameters.get('max_deceleration', -6.0)
        self._max_steering = parameters.get('max_steering', 0.8)
        self._collision_threshold = parameters.get('collision_threshold', 5.0)
        self._finish_buffer = parameters.get('finish_buffer', 20.0)
        self._remove_after_finish = parameters.get('remove_after_finish', False)
        self._collision_distance_threshold = parameters.get('collision_distance_threshold', 5.0)
        # PID controller configuration
        pid_lateral_cfg = parameters.get('pid_lateral_cfg', {
            'K_P': 1.25,
            'K_D': 0.3,
            'K_I': 0.75
        })
        pid_longitudinal_cfg = parameters.get('pid_longitudinal_cfg', {
            'K_P': 5.0,
            'K_D': 1.0,
            'K_I': 0.5
        })

        # 2. auto config
        self._buffer_size = 5
        self._initial_waypoint = self.route[0]

        self._waypoints_queue = deque(maxlen=5000)
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # set waypoints
        for i, waypoint in enumerate(self.route):
            if i == 0:
                continue
            self._waypoints_queue.append(waypoint)

        self._controller = PIDController(
            long_cfg=pid_longitudinal_cfg,
            lat_cfg=pid_lateral_cfg,
        )

        self._finish_time = 0.0

        if self.debug:
            debug_folder = DataProvider.debug_folder()
            self.vehicle_debug_folder = os.path.join(debug_folder, f'traffic_manager/vehicle')
            if not os.path.exists(self.vehicle_debug_folder):
                os.makedirs(self.vehicle_debug_folder)

            log_file = os.path.join(self.vehicle_debug_folder, f"{self.agent.id}.log")
            if os.path.exists(log_file):
                os.remove(log_file)
            self.logger = get_instance_logger(f"waypoint_vehicle_{self.agent.id}", log_file)
            self.logger.info("Logger initialized for this instance")

            self.plan_x = []
            self.plan_y = []
            for i, waypoint in enumerate(self.route):
                self.plan_x.append(waypoint.location.x)
                self.plan_y.append(waypoint.location.y)

            self.actual_x = []
            self.actual_y = []

        # publish information to the bridge
        self.traffic_bridge.register_actor(
            self.agent.id,
            self.agent,
        )
        self.step = 0
        self.thread_run = None

    def start(self):
        if self.traffic_bridge.is_termination:
            return

        if self.debug:
            self.logger.info(f'Start vehicle thread: {self.agent.id}')
        self.thread_run = Thread(target=self._run)
        self.thread_run.start()

    def stop(self):
        self.thread_run.join()
        self.thread_run = None

    def get_agent(self) -> AgentClass:
        return self.agent

    def is_finished(self) -> bool:
        if self._finish_time > self._finish_buffer:
            return True
        else:
            return False

    def _run(self):
        while not self.traffic_bridge.is_termination:
            # update state
            self.tick(1 / DataProvider.SIM_FREQUENCY)
            time.sleep(1 / DataProvider.SIM_FREQUENCY)

        if self.debug:
            plt.figure()
            plt.plot(self.plan_x, self.plan_y, 'b-')
            plt.plot(self.actual_x, self.actual_y, 'r-')
            plt.show()
            plt.savefig(f"{self.vehicle_debug_folder}/{self.agent.id}_traj.png")
            plt.close()

    def tick(
            self,
            delta_time: float,
    ):
        self.step += 1
        curr_location = Point([self.agent.location.x, self.agent.location.y])

        if self.debug:
            self.logger.info(f'=============Start {self.step}=============')
            self.logger.info(f"waypoint_queue length: {len(self._waypoints_queue)}")
            self.logger.info(f"waypoint_buffer length: {len(self._waypoint_buffer)}")

        # 1. if the queue is empty, stop the car
        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            self.agent.speed = 0.0
            self._finish_time += delta_time
            return

        # 3. buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        target_waypoint: WaypointConfig = copy.deepcopy(self._waypoint_buffer[0])
        if target_waypoint.is_junction:
            target_speed = min(target_waypoint.speed, self._max_speed_junction)
        else:
            target_speed = min(target_waypoint.speed, self._max_speed)

        # 4. calculate planning speed
        # 4.1 detect collision
        obstacles = self.traffic_bridge.get_actors()  # obtain observation from the traffic bridge
        hazard_detected = self._obstacle_detected(obstacles, delta_time)
        if hazard_detected:
            target_speed = 0.0

        target_waypoint.speed = target_speed
        # 4.2 detect traffic light
        # TODO: should be implemented

        # Run control
        throttle, brake, steer = self._controller.run_step(
            self.agent,
            target_waypoint,
            delta_time,
        ) # current acceleration & steering

        vehicle_control = VehicleControl(throttle, brake, steer)
        if self.debug:
            self.logger.info(f"delta_time: {delta_time}")
            self.logger.info(f"remain roue length: {len(self._waypoints_queue)}")
            self.logger.info(f"throttle: {throttle}")
            self.logger.info(f"brake: {brake}")
            self.logger.info(f"steer: {steer}")
            self.logger.info(f"target heading (waypoint): {target_waypoint.location.yaw}")
            self.logger.info(f"current heading: {self.agent.location.yaw}")
            self.logger.info(f"target_waypoint: {target_waypoint.lane.lane_id} ({target_waypoint.location.x}, {target_waypoint.location.y})")
            self.logger.info(f"current_waypoint: ({self.agent.location.x}, {self.agent.location.y})")
            self.logger.info(f"distance: {((self.agent.location.x - target_waypoint.location.x) ** 2 + (self.agent.location.y - target_waypoint.location.y) ** 2) ** 0.5}")
            self.logger.info(f"hazard_detected: {hazard_detected}")
            self.logger.info('=============End=============')
            self.actual_x.append(self.agent.location.x)
            self.actual_y.append(self.agent.location.y)

        self.agent.apply_control(vehicle_control)

        # 5. purge the queue of obsolete waypoints
        max_index = -1
        next_location = Point([self.agent.location.x, self.agent.location.y])
        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoint_location = Point([waypoint.location.x, waypoint.location.y])
            if waypoint_location.distance(next_location) < self._min_distance or waypoint_location.distance(curr_location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

    def _obstacle_detected(
            self,
            agent_dict: Dict[str, AgentClass],
            delta_t: float
    ):
        # Total distance traveled
        # distance = curr_state.speed * time_to_stop - 0.5 * self._max_acceleration * time_to_stop ** 2
        distance = self.agent.speed * delta_t + 0.5 * self.agent.acceleration * delta_t ** 2
        distance = float(np.clip(distance, 0.0, None))
        brake_distance = self._collision_distance_threshold + distance * 1.2 # >= self._collision_vehicle_threshold

        buffer_polygon, _, _ = self.agent.get_polygon(buffer=brake_distance)

        # Step 1: get current points and future polygons
        curr_obs_polygon, _, _ = self.agent.get_polygon()
        curr_bbs = [curr_obs_polygon]
        curr_location = Point([self.agent.location.x, self.agent.location.y])
        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoint_location = Point([waypoint.location.x, waypoint.location.y])
            if waypoint_location.distance(curr_location) > brake_distance:
                break
            tmp_state = copy.deepcopy(self.agent)
            tmp_state.x = waypoint.location.x
            tmp_state.y = waypoint.location.y
            tmp_state.heading = waypoint.location.heading
            tmp_state_polygon, _, _ = tmp_state.get_polygon()
            curr_bbs.append(tmp_state_polygon)

        for agent_id, agent in agent_dict.items():
            # self.logger.debug(f'vehicle id: {vehicle.id}, curr id: {curr_state.id}')
            if agent_id == self.agent.id:
                continue
            if self._ignore_static_obstacle and agent.category.split('.')[0] == 'static':
                continue
            if self._ignore_vehicle and agent.category.split('.')[0] == 'vehicle':
                continue
            if self._ignore_walker and agent.category.split('.')[0] == 'walker':
                continue

            obstacle_polygon, _, _ = agent.get_polygon()
            if buffer_polygon.intersects(obstacle_polygon):
                return True
            for curr_polygon in curr_bbs:
                if curr_polygon.intersects(obstacle_polygon):
                    return True

        return False