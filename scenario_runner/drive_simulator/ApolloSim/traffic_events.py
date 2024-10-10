from enum import Enum

class EventType(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    COMPLETE = 0
    COLLISION = 1
    STUCK = 2
    TIMEOUT = 3

class TrafficEvent:

    event_type: EventType
    event_detail: str

    def __init__(self, event_type: EventType, event_detail: str):
        self.event_type = event_type
        self.event_detail = event_detail

    def json_data(self):
        return {
            "event_type": self.event_type.name,
            "event_detail": self.event_detail,
        }