import numpy as np

from . import BaseOracle, TrafficBridge, register_oracle
from scenario_runner.drive_simulator.ApolloSim.traffic_events import EventType

@register_oracle('oracle.collision')
class CollisionOracle(BaseOracle):

    oracle_name = 'oracle.collision'

    def __init__(self, agent_id: str, traffic_bridge: TrafficBridge):
        super(CollisionOracle, self).__init__(agent_id, traffic_bridge)
        self.min_distance = np.inf

    def tick(self):

        agent_state = self.traffic_bridge.query_state(self.agent_id)
        other_agents_state = self.traffic_bridge.get_actors()

        for other_agent_id, other_agent_state in other_agents_state.items():
            if other_agent_id == self.agent_id:
                continue

            center_dist = agent_state.center_distance(other_agent_state)
            if center_dist < 30:
                agent_distance = agent_state.bbox_distance(other_agent_state)
                if agent_distance < self.min_distance:
                    self.min_distance = agent_distance

                if agent_distance < 0.01:
                    return EventType.COLLISION, self.min_distance

        return None, self.min_distance