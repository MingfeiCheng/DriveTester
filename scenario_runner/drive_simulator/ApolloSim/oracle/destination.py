import numpy as np

from shapely.geometry import Point

from . import BaseOracle, TrafficBridge, register_oracle
from scenario_runner.drive_simulator.ApolloSim.config import WaypointConfig
from scenario_runner.drive_simulator.ApolloSim.traffic_events import EventType

@register_oracle('oracle.destination')
class DestinationOracle(BaseOracle):

    oracle_name = 'oracle.destination'

    def __init__(self, agent_id: str, traffic_bridge: TrafficBridge, destination: WaypointConfig, threshold: float = 3.0):
        super().__init__(agent_id, traffic_bridge)

        self._destination_point = Point(destination.location.x, destination.location.y)
        self._threshold = threshold

        self._min_distance = np.inf

    def tick(self):
        agent_state = self.traffic_bridge.query_state(self.agent_id)

        # Check if the angle is within the range [-10, 10] degrees
        dist2dest_bbox = agent_state.dist_bbox2point(self._destination_point)
        dist2dest_center = agent_state.dist_center2point(self._destination_point)
        if dist2dest_bbox < self._min_distance:
            self._min_distance = dist2dest_bbox
        if dist2dest_bbox <= self._threshold and agent_state.speed < 0.5 and dist2dest_center < 2 * self._threshold:
            return EventType.COMPLETE, self._min_distance
        return None, self._min_distance