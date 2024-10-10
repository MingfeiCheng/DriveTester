import math
from typing import Dict

import numpy as np

from collections import deque
from shapely.geometry import Point

from scenario_runner.drive_simulator.ApolloSim.config.vehicle import WaypointConfig
from scenario_runner.drive_simulator.ApolloSim.library import AgentClass

class PIDController:

    def __init__(
            self,
            long_cfg: Dict,
            lat_cfg: Dict
    ):
        self.longitudinal_control = PIDLongitudinalController(**long_cfg)
        self.lateral_control = PIDLateralController(**lat_cfg)

    def run_step(self, agent: AgentClass, target_waypoint: WaypointConfig, dt: float):

        curr_speed = agent.speed
        curr_heading = agent.location.yaw

        # heading
        curr_point = Point(agent.location.x, agent.location.y)
        next_point = Point((agent.location.x + target_waypoint.location.x) /2.0, (agent.location.y + target_waypoint.location.y) / 2.0)

        target_heading = math.atan2(next_point.y - curr_point.y, next_point.x - curr_point.x)
        error = target_heading - curr_heading

        # Normalize the error to be within [-pi, pi]
        heading_error = (error + math.pi) % (2 * math.pi) - math.pi
        # angle = np.pi / 2 - np.arctan2(next_point.y - curr_point.y, next_point.x - curr_point.x)
        if curr_speed < 0.01:
            heading_error = 0.0

        steer = self.lateral_control.run_step(heading_error, dt)
        steer = np.clip(steer, -1.0, 1.0) # we need this for adding more dynamic models

        # speed
        target_speed = target_waypoint.speed
        delta = target_speed - curr_speed
        throttle_brake = self.longitudinal_control.run_step(delta, dt)
        throttle_brake = np.clip(throttle_brake, -1.0, 1.0)
        if throttle_brake > 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        return throttle, brake, steer

class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0):
        """
        Constructor method.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._error_buffer = deque(maxlen=10)

    def run_step(self, error: float, dt: float) -> float:
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / dt
            _ie = sum(self._error_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class PIDLateralController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0):
        """
        Constructor method.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._error_buffer = deque(maxlen=10)

    def run_step(self, error: float, dt: float) -> float:
        if error > math.pi:
            error = error - 2 * math.pi
        if error < -math.pi:
            error = error + 2 * math.pi
        error = float(np.clip(error, -1.0, 1.0))
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / dt
            _ie = sum(self._error_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)