from typing import List

from apollo_sim.sim_env import SimEnv

class SubScenarioManager(object):

    def __init__(
            self,
            scenario_idx: str,
            config_pool: List,
            sim_env: SimEnv,
            output_folder: str,
            debug: bool = False,
            **kwargs
    ):
        self.scenario_idx = scenario_idx
        self.config_pool = config_pool
        self.sim_env = sim_env
        self.output_folder = output_folder
        self.debug = debug

        self.agent_list = list()

        self._initialize()

    def _initialize(self):
        raise NotImplementedError

    def start(self):
        for agent in self.agent_list:
            agent.start()

    def stop(self):
        for agent in self.agent_list:
            agent.stop()