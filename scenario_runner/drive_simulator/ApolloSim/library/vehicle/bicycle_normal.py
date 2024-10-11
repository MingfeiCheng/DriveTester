from dataclasses import dataclass

from .. import register_agent
from ..base import BaseModel, LocationConfig, BBoxConfig

from threading import Lock

@register_agent("bicycle.normal")
@dataclass
class LincolnMKZ(BaseModel):

    # basic information - fixed
    category: str = 'bicycle.normal'
    bbox: BBoxConfig = BBoxConfig(
        length=3.0,
        width=1.0,
        height=1.8
    )

    max_acceleration: float = 2.0 #5.59 # not accuracy
    max_deceleration: float = -6.0

    front_edge_to_center: float = 1.5
    back_edge_to_center: float = 1.5
    left_edge_to_center: float = 0.5
    right_edge_to_center: float = 0.5

    max_steer_angle: float = 8.20304748437 # radians * 180 / math.pi
    steer_ratio: float = 16.0

    wheelbase: float = 2.8448
    max_abs_speed_when_stopped: float = 0.2

    def __init__(self, idx: int, location: LocationConfig, role: str):
        super(LincolnMKZ, self).__init__(idx, location, role)
