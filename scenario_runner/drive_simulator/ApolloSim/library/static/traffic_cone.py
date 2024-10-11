from .. import register_agent
from ..base import BaseModel, LocationConfig, BBoxConfig
from dataclasses import dataclass

@register_agent("static.traffic_cone")
@dataclass
class TrafficCone(BaseModel):

    # basic information - fixed
    category: str = 'static.traffic_cone'
    bbox: BBoxConfig = BBoxConfig(
        length=0.5,
        width=0.5,
        height=1.0
    )

    max_acceleration: float = 0.0 #5.59 # not accuracy
    max_deceleration: float = 0.0

    front_edge_to_center: float = 0.25
    back_edge_to_center: float = 0.25
    left_edge_to_center: float = 0.25
    right_edge_to_center: float = 0.25

    max_steer_angle: float = 0 # radians * 180 / math.pi
    steer_ratio: float = 0

    wheelbase: float = 0
    max_abs_speed_when_stopped: float = 0

    def __init__(self, idx: int, location: LocationConfig, role: str):
        super(TrafficCone, self).__init__(idx, location, role)