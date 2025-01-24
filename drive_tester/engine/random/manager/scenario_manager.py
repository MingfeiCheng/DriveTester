from apollo_sim.sim_env import SimEnv
from drive_tester.scenario_runner.ApolloSim.config.scenario import ScenarioConfig
from drive_tester.scenario_runner.ApolloSim.manager import ScenarioManager
from drive_tester.scenario_runner.ApolloSim.manager.examples import StaticObstacleManager, VehicleManager, WalkerManager, TrafficLightManager, ApolloManager

from drive_tester.registry import MANAGER_REGISTRY

@MANAGER_REGISTRY.register("manager.random")
class RandomScenarioManager(ScenarioManager):

    def __init__(
            self,
            sim_env: SimEnv,
            scenario_config: ScenarioConfig,
            output_folder: str,
            debug: bool = False,
    ):

        super(RandomScenarioManager, self).__init__(
            sim_env=sim_env,
            scenario_config=scenario_config,
            output_folder=output_folder,
            debug=debug
        )

    def _initialize(self):
        # apollo manager
        self.manager_apollo = ApolloManager(
            scenario_idx=self.scenario_config.idx,
            config_pool=self.scenario_config.apollo,
            sim_env=self.sim_env,
            output_folder=self.output_folder,
            debug=self.debug,
        )
        self.manager_static_obstacle = StaticObstacleManager(
            scenario_idx=self.scenario_config.idx,
            config_pool=self.scenario_config.static_obstacle,
            sim_env=self.sim_env,
            output_folder=self.output_folder,
            debug=self.debug,
        )
        self.manager_vehicle = VehicleManager(
            scenario_idx=self.scenario_config.idx,
            config_pool=self.scenario_config.vehicle,
            sim_env=self.sim_env,
            output_folder=self.output_folder,
            debug=self.debug,
        )
        self.manager_walker = WalkerManager(
            scenario_idx=self.scenario_config.idx,
            config_pool=self.scenario_config.walker,
            sim_env=self.sim_env,
            output_folder=self.output_folder,
            debug=self.debug,
        )
        self.manager_traffic_light = TrafficLightManager(
            scenario_idx=self.scenario_config.idx,
            config=self.scenario_config.traffic_light,
            sim_env=self.sim_env,
            output_folder=self.output_folder,
            debug=self.debug,
        )

        self.manager_lst = [
            self.manager_apollo,
            self.manager_static_obstacle,
            self.manager_vehicle,
            self.manager_walker,
            self.manager_traffic_light
        ]
