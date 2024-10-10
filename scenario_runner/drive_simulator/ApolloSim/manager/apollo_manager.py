import time

from common.data_provider import DataProvider

from scenario_runner.drive_simulator.ApolloSim.ads.apollo.agent import ApolloAgent
from ..config import ApolloConfigPool
from ..recorder import ScenarioRecorder
from ..traffic_messenger import TrafficBridge

class ApolloManager:

    config_pool: ApolloConfigPool
    agent_list: list
    traffic_bridge: TrafficBridge

    def __init__(
            self,
            scenario_idx: str,
            config_pool: ApolloConfigPool,
            traffic_bridge: TrafficBridge,
            result_recorder: ScenarioRecorder,
    ):
        self.traffic_bridge = traffic_bridge
        self.result_recorder =result_recorder
        self.config_pool = config_pool
        self.scenario_idx = scenario_idx
        self.agent_list = list()
        for config in self.config_pool.configs:
            self.agent_list.append(ApolloAgent(config, self.traffic_bridge, self.result_recorder))

    def start(self):
        for agent in self.agent_list:
            if DataProvider.save_apollo_recording:
                # start recorder
                container_record_folder = DataProvider.container_record_folder()
                agent.recorder_operator('start', container_record_folder, f"{self.scenario_idx}_{agent.idx}")
            agent.start()

    def stop(self):
        for agent in self.agent_list:
            if DataProvider.save_apollo_recording:
                agent.recorder_operator('stop')
                container_record_folder = DataProvider.container_record_folder()
                agent.move_recording(container_record_folder, f"{self.scenario_idx}_{agent.idx}")
                time.sleep(0.1)
            agent.stop()

    def publish_state(self):
        for agent in self.agent_list:
            self.traffic_bridge.update_actor(agent.agent.id, agent.agent)