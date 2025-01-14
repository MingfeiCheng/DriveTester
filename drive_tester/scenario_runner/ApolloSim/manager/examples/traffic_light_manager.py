from loguru import logger

from apollo_sim.agents.traffic_light.rule_light import RuleLightConfig, RuleLightAgent
from apollo_sim.sim_env import SimEnv

from drive_tester.scenario_runner.ApolloSim.manager import SubScenarioManager

class TrafficLightManager(SubScenarioManager):

    def __init__(
            self,
            scenario_idx: str,
            config: RuleLightConfig,
            sim_env: SimEnv,
            output_folder: str,
            # other configs
            debug: bool = False,
    ):
        super(TrafficLightManager, self).__init__(
            scenario_idx=scenario_idx,
            config_pool=[config],
            sim_env=sim_env,
            output_folder=output_folder,
            debug=debug
        )

    def _initialize(self):
        self.traffic_light_agent = RuleLightAgent(
            config=self.config_pool[0],
            sim_env=self.sim_env,
            debug=self.debug,
            output_folder=self.output_folder,
            scenario_idx=self.scenario_idx
        )

    def start(self):
        self.traffic_light_agent.start()

    def stop(self):
        self.traffic_light_agent.stop()