from typing import Optional

from scenario_runner.drive_simulator.ApolloSim.config.traffic_light import TrafficLightConfig
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge
from scenario_runner.drive_simulator.ApolloSim.agents.traffic_light.traffic_light_agent import TrafficLightAgent

class TrafficLightManager(object):
    category = 'traffic_light'

    def __init__(
            self,
            config: Optional[TrafficLightConfig],
            traffic_bridge: TrafficBridge
    ):
        self.traffic_bridge = traffic_bridge
        self.config = config
        self.tf_agent = TrafficLightAgent(config, traffic_bridge)

    def start(self):
        self.tf_agent.start()

    def stop(self):
        self.tf_agent.stop()

    def publish_state(self):
        self.traffic_bridge.update_traffic_light(self.tf_agent.get_state())