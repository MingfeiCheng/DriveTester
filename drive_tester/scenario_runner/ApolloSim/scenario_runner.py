import os
import time
import traceback

from typing import Optional,Dict

from omegaconf import DictConfig
from loguru import logger

from apollo_sim.sim_env import SimEnv

from drive_tester.scenario_runner import Runner
from drive_tester.scenario_runner.ApolloSim.manager import ScenarioConfig
from drive_tester.scenario_runner.ApolloSim.common.runner_config import GlobalRunnerConfig
from drive_tester.common.global_config import GlobalConfig
from drive_tester.registry import RUNNER_REGISTRY, MANAGER_REGISTRY

@RUNNER_REGISTRY.register("runner.ApolloSim")
class ScenarioRunner(Runner):
    """
    Executes a scenario based on the specification
    """
    name = 'ApolloSim'

    def __init__(
            self,
            runner_config: DictConfig,
            scenario_config: DictConfig
    ):
        super(ScenarioRunner, self).__init__(runner_config, scenario_config)

        # setup parameters
        GlobalRunnerConfig.apollo_root = runner_config.get("apollo_root", "/apollo")
        GlobalRunnerConfig.map_root = runner_config.get("map_root", "/apollo/modules/map/data")
        GlobalRunnerConfig.off_screen = runner_config.get("off_screen", False)
        GlobalRunnerConfig.sim_frequency = runner_config.get("sim_frequency", 25.0)
        GlobalRunnerConfig.sim_port = runner_config.get("sim_port", 15000)

        GlobalRunnerConfig.map_name = scenario_config.get("map_name", "borregas_ave")

        GlobalRunnerConfig.debug = GlobalConfig.debug
        GlobalRunnerConfig.save_record = GlobalConfig.save_record
        GlobalRunnerConfig.output_root = GlobalConfig.output_root

        GlobalRunnerConfig.print()

        self.simulator = SimEnv(
            map_name=GlobalRunnerConfig.map_name,
            apollo_root=GlobalRunnerConfig.apollo_root,
            map_root=GlobalRunnerConfig.map_root,
            save_record=GlobalRunnerConfig.save_record,
            off_screen=GlobalRunnerConfig.off_screen,
            sim_frequency=GlobalRunnerConfig.sim_frequency,
            port=GlobalRunnerConfig.sim_port,
            debug=GlobalRunnerConfig.debug
        )

    def run(
            self,
            scenario_instance: ScenarioConfig,
            manager_name: str
    ) -> Optional[Dict]:
        try:
            scenario_result = self._run_scenario(scenario_instance, manager_name)
        except Exception as e:
            self.simulator.stop()
            traceback.print_exc()
            logger.error('Running error for scenario: {}, Please check.'.format(scenario_instance.idx))
            return None
        return scenario_result

    def _run_scenario(self, scenario_instance: ScenarioConfig, manager_name: str) -> Dict:
        """
        Execute the scenario based on the specification
        |_scenario
            |- scenario_idx
                |- scenario.json
                |- records_simulator.pkl
                |- simulation_result.json
                |- records_apollo (folder)
                |- debug (folder)
        """
        if scenario_instance is None:
            raise RuntimeError('Error: No chromosome or not initialized')

        self.simulator.reload_world()

        # create some related folders
        scenario_folder = GlobalRunnerConfig.scenario_folder(scenario_instance.idx)
        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)

        # 1. save scenario file
        scenario_instance.export(os.path.join(scenario_folder, 'scenario.json'))

        # 2. build scenario
        scenario_manager_class = MANAGER_REGISTRY.get(manager_name)
        scenario_manager = scenario_manager_class(
            self.simulator,
            scenario_instance,
            output_folder=scenario_folder,
            debug=GlobalRunnerConfig.debug
        )
        logger.info(f'--> Run Scenario ID: {scenario_instance.idx}')

        # 4. run all components
        m_start_time = time.time()
        scenario_manager.run()
        # ----> wait until the scenario is terminated
        m_end_time = time.time()
        simulation_spend_time = m_end_time - m_start_time
        logger.info('--> [Simulation Time] Simulation Spend Time (seconds): [=]{}[=]', simulation_spend_time)

        # 5. export related things
        # simulator recordings
        if GlobalRunnerConfig.save_record:
            sim_record_file = os.path.join(scenario_folder, 'records_simulator.pkl')
            self.simulator.export_record(sim_record_file)

        # simulation result
        result_file = os.path.join(scenario_folder, 'simulation_result.json')
        scenario_result = self.simulator.export_result(result_file)

        scenario_result['path'] = scenario_folder
        return scenario_result

    def shutdown(self):
        self.simulator.close()


