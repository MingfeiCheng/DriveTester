import copy
import os.path
import time
import traceback

from omegaconf import DictConfig
from tqdm import tqdm
from loguru import logger

from common.data_provider import DataProvider
from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig
from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder
from scenario_runner.drive_simulator.ApolloSim.traffic_manager import TrafficManager
from . import register_runner

@register_runner("ApolloSim")
class ScenarioRunner:
    """
    Executes a scenario based on the specification
    """
    def __init__(
            self,
            cfg: DictConfig
    ):
        # setup configs
        DataProvider.container_name = cfg.get("container_name", "test")
        DataProvider.save_traffic_recording = cfg.get("save_traffic_recording", True)
        DataProvider.save_apollo_recording = cfg.get("save_apollo_recording", True)

    def run(
            self,
            scenario_config: ApolloSimScenarioConfig
    ) -> ScenarioRecorder:
        try:
            scenario_recorder = self._run_scenario(scenario_config)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise RuntimeError("Has bugs in running function")
        return copy.deepcopy(scenario_recorder)

    @staticmethod
    def _run_scenario(scenario_config: ApolloSimScenarioConfig):
        """
        Execute the scenario based on the specification
        """
        # 0. build scenario
        traffic_manager = TrafficManager(
            scenario_config
        )

        # 1. config preconditions
        logger.info(f'--> Run Scenario ID: {scenario_config.idx}')

        if scenario_config is None:
            logger.error('Error: No chromosome or not initialized')
            return None

        # 2. run all components
        traffic_manager.start()
        bar = tqdm()
        m_start_time = time.time()
        # Begin Scenario Cycle
        while not traffic_manager.is_termination:
            bar.set_description('--> Scenario time: {}.'.format(round((time.time() - m_start_time), 2)))
            time.sleep(0.05)
        print('') # for better view
        # scenario ended
        traffic_recording_path = os.path.join(DataProvider.traffic_recording_folder(), f"{scenario_config.idx}.pkl")
        traffic_manager.stop(traffic_recording_path)
        m_end_time = time.time()
        simulation_spend_time = m_end_time - m_start_time
        logger.info('--> [Simulation Time] Simulation Spend Time (seconds): [=]{}[=]', simulation_spend_time)

        scenario_recorder = traffic_manager.scenario_result
        scenario_recorder.update_timer('simulation', simulation_spend_time)
        scenario_recorder.print_result()
        return scenario_recorder
