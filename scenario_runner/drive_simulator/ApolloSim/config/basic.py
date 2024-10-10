from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class LaneConfig:

    lane_id: str
    s: float
    l: float

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'LaneConfig':
        return cls(**json_node)

@dataclass
class LocationConfig:

    x: float
    y: float
    z: float
    pitch: float
    yaw: float # heading
    roll: float

    @property
    def heading(self):
        return self.yaw

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'LocationConfig':
        return cls(**json_node)

@dataclass
class VelocityConfig:
    x: float
    y: float
    z: float

    @property
    def scale_value(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5

@dataclass
class BBoxConfig:
    length: float
    width: float
    height: float

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'BBoxConfig':
        return cls(**json_node)

@dataclass
class WaypointConfig:

    location: LocationConfig
    lane: LaneConfig
    speed: float
    is_junction: bool = False

    @classmethod
    def from_json(cls, json_node: Dict) -> 'WaypointConfig':
        json_node['location'] = LocationConfig.from_json(json_node['location'])
        json_node['lane'] = LaneConfig.from_json(json_node['lane'])
        return cls(**json_node)

    def json_data(self) -> Dict:
        return asdict(self)

@dataclass
class CommandConfig:

    command: int # 0-keep lane, 1-turn right, 2-turn left lane
    timestamp: float

    @classmethod
    def from_json(cls, json_node: Dict) -> 'CommandConfig':
        return cls(**json_node)

    def json_data(self) -> Dict:
        return asdict(self)