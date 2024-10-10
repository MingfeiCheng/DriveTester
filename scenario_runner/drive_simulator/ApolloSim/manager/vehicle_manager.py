from ..config import VehicleConfigPool, WaypointConfig, CommandConfig
from scenario_runner.drive_simulator.ApolloSim.agents.vehicle import WaypointVehicle, CommandVehicle
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge

class VehicleManager(object):

    def __init__(
            self,
            config_pool: VehicleConfigPool,
            traffic_bridge: TrafficBridge,
    ):
        self.traffic_bridge = traffic_bridge
        self.config_pool = config_pool

        # create waypoint agent list
        self.agent_list = list()
        for config in self.config_pool.configs:
            if isinstance(config.behavior[0], WaypointConfig):
                self.agent_list.append(WaypointVehicle(config, self.traffic_bridge))
            elif isinstance(config.behavior[0], CommandConfig):
                self.agent_list.append(CommandVehicle(config, self.traffic_bridge))
            else:
                raise NotImplementedError(f"Unsupported vehicle behavior: {type(config.behavior[0])}")

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