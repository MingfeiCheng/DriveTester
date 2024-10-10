import time

from . import BaseOracle, TrafficBridge, register_oracle
from ..traffic_events import EventType

@register_oracle('oracle.stuck')
class StuckOracle(BaseOracle):

    oracle_name = 'oracle.stuck'

    def __init__(self, agent_id: str, traffic_bridge: TrafficBridge, speed_threshold = 0.3, max_stuck_time = 90):
        """

        :param agent_id:
        :param traffic_bridge:
        :param speed_threshold: lower speed threshold
        :param max_stuck_time: seconds
        """
        super(StuckOracle, self).__init__(agent_id, traffic_bridge)

        self.speed_threshold = speed_threshold
        self.max_stuck_time = max_stuck_time

        self.stuck_time_start_time = None # seconds
        self.stuck_time = 0
        self._stuck_time_feedback = 0

    def tick(self):

        agent_state = self.traffic_bridge.query_state(self.agent_id)

        agent_speed = agent_state.speed
        if agent_speed < self.speed_threshold:
            if self.stuck_time_start_time is None:
                self.stuck_time_start_time = time.time()
            else:
                self.stuck_time = time.time() - self.stuck_time_start_time
        else:
            self.stuck_time_start_time = None
            self.stuck_time = 0

        if self.stuck_time > self._stuck_time_feedback:
            self._stuck_time_feedback = self.stuck_time

        if self.stuck_time > self.max_stuck_time:
            return EventType.STUCK, self._stuck_time_feedback
        else:
            return None, self._stuck_time_feedback