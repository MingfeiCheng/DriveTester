import math
import copy
import numpy as np

from ..base import BaseModel, WalkerControl, DataProvider, normalize_angle
from .. import register_agent
from ...config import BBoxConfig, LocationConfig


@register_agent("walker.pedestrian.normal")
class PedestrianNormal(BaseModel):

    category = "walker.pedestrian.normal"
    bbox = BBoxConfig(
        length=0.5,
        width=0.5,
        height=1.8,
    )

    max_acceleration = 10.0
    max_deceleration = 10.0

    front_edge_to_center = 0.25
    back_edge_to_center = 0.25
    left_edge_to_center = 0.25
    right_edge_to_center = 0.25

    def __init__(self, idx: int, location: LocationConfig, role: str = 'walker'):
        super(PedestrianNormal, self).__init__(idx, location, role)

    def apply_control(self, control: WalkerControl):
        acceleration = control.acceleration
        heading = control.heading

        with self._thread_lock:
            delta_time = 1 / DataProvider.SIM_FREQUENCY
            curr_acceleration = float(np.clip(acceleration, -abs(self.max_deceleration), abs(self.max_acceleration)))
            curr_speed = self.speed
            next_speed = curr_speed + curr_acceleration * delta_time  # according to the frequency
            next_speed = max(0.0, next_speed)  # Ensure speed is non-negative

            next_heading = normalize_angle(heading)

            next_x = self.location.x + next_speed * math.cos(next_heading) * delta_time
            next_y = self.location.y + next_speed * math.sin(next_heading) * delta_time

            # 6. Create the next state
            self.location.x = next_x
            self.location.y = next_y
            self.location.yaw = next_heading
            self.speed = next_speed
            self.acceleration = curr_acceleration
            self.angular_speed = next_speed
            self.control = copy.deepcopy(control)

