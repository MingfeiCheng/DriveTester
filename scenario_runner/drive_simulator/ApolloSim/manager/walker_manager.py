from ..config import WalkerConfigPool
from scenario_runner.drive_simulator.ApolloSim.agents.walker.waypoint_walker import WaypointWalker
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge

class WalkerManager(object):

    def __init__(
            self,
            config_pool: WalkerConfigPool,
            traffic_bridge: TrafficBridge
    ):

        self.traffic_bridge = traffic_bridge
        self.config_pool = config_pool

        # create agent list
        self.agent_list = list()
        for config in self.config_pool.configs:
            self.agent_list.append(WaypointWalker(config, self.traffic_bridge))

    def start(
            self
    ):
        for agent in self.agent_list:
            # add tracking for each agent
            agent.start()

    def stop(self):
        for agent in self.agent_list:
            agent.stop()

    def publish_state(self):
        for agent in self.agent_list:
            agent_actor = agent.get_agent()
            self.traffic_bridge.update_actor(agent_actor.id, agent_actor)