import time

from . import BaseOracle, TrafficBridge, register_oracle
from ..traffic_events import EventType

@register_oracle('oracle.timeout')
class TimeoutOracle(BaseOracle):

    oracle_name = 'oracle.timeout'

    def __init__(self, agent_id: str, traffic_bridge: TrafficBridge, time_limit = 300):
        """

        :param agent_id:
        :param traffic_bridge:
        :param time_limit: seconds
        """
        super(TimeoutOracle, self).__init__(agent_id, traffic_bridge)
        self.time_limit = time_limit
        self.start_time = None

    def tick(self):
        if self.start_time is None:
            self.start_time = time.time()

        spend_time = time.time() - self.start_time

        if spend_time > self.time_limit:
            return EventType.TIMEOUT, spend_time

        return None, spend_time