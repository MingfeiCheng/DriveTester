import copy

from scenario_runner.drive_simulator.ApolloSim.library import agent_library
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge
from scenario_runner.drive_simulator.ApolloSim.config.static import StaticObstacleConfigPool

class StaticObstacleManager(object):

    def __init__(
            self,
            config_pool: StaticObstacleConfigPool,
            traffic_bridge: TrafficBridge
    ):
        self.config_pool = config_pool
        self.traffic_bridge = traffic_bridge

        self.agents = list()
        for config in self.config_pool.configs:
            agent_class = agent_library.get(config.category)
            agent = agent_class(
                idx=config.idx,
                location=copy.deepcopy(config.initial_waypoint.location),
                role=config.role
            )
            self.agents.append(agent)

            # publish to the bridge
            self.traffic_bridge.register_actor(
                agent.id,
                agent
            )

    def start(self):
        # do not need as the static obstacles are stable
        pass

    def stop(self):
        pass

    def publish_state(self):
        for agent in self.agents:
            self.traffic_bridge.update_actor(agent.id, agent)