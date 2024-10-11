from dataclasses import dataclass

from .. import register_agent
from ..base import BaseModel, LocationConfig, BBoxConfig

from threading import Lock

# https://github.com/ApolloAuto/apollo/blob/v7.0.0/modules/calibration/data/mkz_example/vehicle_param.pb.txt

"""
vehicle_param {
  brand: LINCOLN_MKZ
  vehicle_id {
      other_unique_id: "mkz"
  }
  front_edge_to_center: 3.89
  back_edge_to_center: 1.043
  left_edge_to_center: 1.055
  right_edge_to_center: 1.055

  length: 4.933
  width: 2.11
  height: 1.48
  min_turn_radius: 5.05386147161
  max_acceleration: 2.0
  max_deceleration: -6.0
  max_steer_angle: 8.20304748437
  max_steer_angle_rate: 6.98131700798
  steer_ratio: 16
  wheel_base: 2.8448
  wheel_rolling_radius: 0.335
  max_abs_speed_when_stopped: 0.2
  brake_deadzone: 14.5
  throttle_deadzone: 15.7
}
"""
@register_agent("vehicle.lincoln.mkz")
@dataclass
class LincolnMKZ(BaseModel):

    # basic information - fixed
    category: str = 'vehicle.lincoln.mkz'
    bbox: BBoxConfig = BBoxConfig(
        length=4.933,
        width=2.11,
        height=1.48
    )

    max_acceleration: float = 2.0 #5.59 # not accuracy
    max_deceleration: float = -6.0

    front_edge_to_center: float = 3.89
    back_edge_to_center: float = 1.043
    left_edge_to_center: float = 1.055
    right_edge_to_center: float = 1.055

    max_steer_angle: float = 8.20304748437 # radians * 180 / math.pi
    steer_ratio: float = 16.0

    wheelbase: float = 2.8448
    max_abs_speed_when_stopped: float = 0.2

    def __init__(self, idx: int, location: LocationConfig, role: str):
        super(LincolnMKZ, self).__init__(idx, location, role)
