import copy
import math

from threading import Lock
from typing import TypeVar, Tuple, List
from shapely.geometry import Polygon, Point

from modules.common.proto.geometry_pb2 import Point3D

from common.data_provider import DataProvider

from ..config import LocationConfig, BBoxConfig

"""
  front_edge_to_center: 3.705
  back_edge_to_center: 0.995
  left_edge_to_center: 1.03
  right_edge_to_center: 1.03

  length: 4.70
  width: 2.06
  height: 2.05
  min_turn_radius: 5.05386147161
  max_acceleration: 2.0
  max_deceleration: -6.0
  max_steer_angle: 8.20304748437
  max_steer_angle_rate: 6.98131700798
  steer_ratio: 16
  wheel_base: 2.837007
  wheel_rolling_radius: 0.33
  max_abs_speed_when_stopped: 0.2
  brake_deadzone: 14.5
  throttle_deadzone: 15.7
"""

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def right_rotation(coord, theta):
    """
    theta : degree
    """
    # theta = math.radians(theta)
    x = coord[1]
    y = coord[0]
    x1 = x * math.cos(theta) - y * math.sin(theta)
    y1 = x * math.sin(theta) + y * math.cos(theta)
    return [y1, x1]

class VehicleControl(object):

    def __init__(self, throttle: float = 0.0, brake: float = 0.0, steering: float = 0.0):
        self.throttle = throttle
        self.brake = brake
        self.steering = steering

class WalkerControl(object):
    def __init__(self, acceleration: float = 0.0, heading:float = 0.0):
        self.acceleration = acceleration
        self.heading = heading

class BaseModel(object):

    id: int = 0

    # basic information - fixed
    category: str = 'unknown'
    bbox: BBoxConfig = BBoxConfig(
        length=0.0,
        width=0.0,
        height=0.0
    )

    max_acceleration: float = 0.0
    max_deceleration: float = 0.0

    front_edge_to_center: float = 0.0
    back_edge_to_center: float = 0.0
    left_edge_to_center: float = 0.0
    right_edge_to_center: float = 0.0

    max_steer_angle: float = 0.0 # radians * 180 / math.pi
    steer_ratio: float = 0.0

    wheelbase: float = 0.0
    max_abs_speed_when_stopped: float = 0.0

    # initialization parameters
    # motion state
    location: LocationConfig = LocationConfig(
        x=0.0,
        y=0.0,
        z=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
    )
    speed: float = 0.0
    angular_speed: float = 0.0
    acceleration: float = 0.0

    # control information
    control: VehicleControl or WalkerControl = VehicleControl(
        throttle=0.0,
        brake=0.0,
        steering=0.0,
    )

    role: str = 'unknown'

    _thread_lock: Lock = Lock()

    def __init__(self, idx: int, location: LocationConfig, role: str = 'default'):
        self.id = int(idx)
        self.location = location
        self.speed = 0.0
        self.angular_speed = 0.0
        self.acceleration = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.role = role

    def apply_control(self, control: VehicleControl):
        """
        All in range of [0, 1]
        NOW WE ONLY support linear model for each object
        """
        throttle = control.throttle
        brake = control.brake
        steering = control.steering

        assert 0 <= throttle <= 1
        assert 0 <= brake <= 1
        assert -1 <= steering <= 1

        with self._thread_lock:
            delta_time = 1 / DataProvider.SIM_FREQUENCY

            # 1. Compute current acceleration based on throttle and brake
            if throttle > 0.0 and brake == 0:
                # Accelerating based on throttle
                curr_acceleration = throttle * abs(self.max_acceleration)
            elif throttle == 0.0 and brake > 0:
                # Decelerating based on brake
                curr_acceleration = -brake * abs(self.max_deceleration)
            elif throttle == 0.0 and brake == 0:
                curr_acceleration = 0.0

            # 2. Update speed and ensure it doesn't go below zero
            curr_speed = self.speed
            if curr_speed <= 0.0:
                curr_acceleration = max(0.0, curr_acceleration)
            next_speed = curr_speed + curr_acceleration * delta_time # according to the frequency
            next_speed = max(0.0, next_speed)  # Ensure speed is non-negative

            # 3. Compute the steering angle in radians and angular velocity
            steering_angle = math.radians(steering * (self.max_steer_angle * 180 / math.pi) / self.steer_ratio)  # degree?
            if abs(steering_angle) > 1e-4:  # Avoid near-zero angle issues
                avg_speed = (curr_speed + next_speed) / 2.0  # Use average speed
                curr_angular_speed = avg_speed * math.tan(steering_angle) / self.wheelbase
            else:
                curr_angular_speed = 0.0

            # 4. Update position using the next speed
            next_x = self.location.x + next_speed * math.cos(self.location.yaw) * delta_time
            next_y = self.location.y + next_speed * math.sin(self.location.yaw) * delta_time

            # 5. Update heading and normalize it
            next_heading = normalize_angle(self.location.yaw + curr_angular_speed * delta_time)

            # 6. Create the next state
            self.location.x = next_x
            self.location.y = next_y
            self.location.yaw = next_heading
            self.speed = next_speed
            self.acceleration = curr_acceleration
            self.angular_speed = curr_angular_speed
            self.control = copy.deepcopy(control)

    def get_forward_vector(self) -> List:
        init_vector = [1, 0]
        forward_vector = right_rotation(init_vector, -self.location.yaw)
        return forward_vector

    def get_polygon(self, buffer: float = 0.0) -> Tuple[Polygon, List[Point3D], List]:
        half_w = self.bbox.width / 2.0

        front_l = self.bbox.length - self.back_edge_to_center
        back_l = -1 * self.back_edge_to_center
        front_l += buffer

        sin_h = math.sin(self.location.yaw)
        cos_h = math.cos(self.location.yaw)
        vectors = [(front_l * cos_h - half_w * sin_h,
                    front_l * sin_h + half_w * cos_h),
                   (back_l * cos_h - half_w * sin_h,
                    back_l * sin_h + half_w * cos_h),
                   (back_l * cos_h + half_w * sin_h,
                    back_l * sin_h - half_w * cos_h),
                   (front_l * cos_h + half_w * sin_h,
                    front_l * sin_h - half_w * cos_h)]

        points = []
        apollo_points = []  # Apollo Points
        for x, y in vectors:
            points.append([self.location.x + x, self.location.y + y])
            p = Point3D()
            p.x = self.location.x + x
            p.y = self.location.y + y
            p.z = 0.0
            apollo_points.append(p)

        return Polygon(points), apollo_points, points

    def bbox_distance(self, agent: 'AgentClass') -> float:
        self_polygon, _, _ = self.get_polygon(buffer=0.0)
        agent_polygon, _, _ = agent.get_polygon(buffer=0.0)
        return self_polygon.distance(agent_polygon)

    def center_distance(self, agent: 'AgentClass') -> float:
        return ((self.location.x - agent.location.x)**2 + (self.location.y - agent.location.y)**2) ** 0.5

    def dist_bbox2point(self, point: Point):
        self_polygon, _, _ = self.get_polygon()
        return self_polygon.distance(point)

    def dist_bbox2polygon(self, polygon: Polygon):
        self_polygon, _, _ = self.get_polygon()
        return self_polygon.distance(polygon)

    def dist_center2point(self, point: Point):
        center = Point(self.location.x, self.location.y)
        return center.distance(point)

AgentClass = TypeVar("AgentClass", bound=BaseModel)