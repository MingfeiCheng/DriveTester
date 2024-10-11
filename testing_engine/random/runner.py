import copy
import sys
import time

from loguru import logger
from omegaconf import DictConfig

from common.data_provider import DataProvider
from testing_engine.common.result_recorder import ResultRecorder
from .adapter.ApolloSim import ApolloSimAdaptor

from .. import register_tester

@register_tester("tester.random")
class RandomTester(object):
    """
    The random fuzzer is:
    """
    def __init__(
            self,
            scenario_runner,
            scenario_cfg: DictConfig,
            algorithm_cfg: DictConfig,
            oracle_cfg: DictConfig
    ):

        self.scenario_runner = scenario_runner
        self.map_name = scenario_cfg.map_name

        algorithm_parameters = algorithm_cfg.get("parameters", {
            "run_hour": 4,
            "num_vehicle": 5,
            "num_walker": 2,
            "num_static": 2
        })

        self.run_hour = algorithm_parameters.get("run_hour", 4)
        self.num_vehicle = algorithm_parameters.get("num_vehicle", 1)
        self.num_walker = algorithm_parameters.get("num_walker", 1)
        self.num_static = algorithm_parameters.get("num_static", 1)

        # Seed Scenario
        self.ego_start_lane_id = scenario_cfg.get("start_lane_id", None)
        self.ego_end_lane_id = scenario_cfg.get("end_lane_id", None)

        # config population
        self.save_root = DataProvider.output_folder() # this is the fuzzing result folder to save the results
        self.result_recorder = ResultRecorder(
            self.save_root,
            self.run_hour
        )

        # create adapter
        self.adapter = ApolloSimAdaptor(
            self.map_name,
            self.ego_start_lane_id,
            self.ego_end_lane_id,
            self.num_vehicle,
            self.num_walker,
            self.num_static
        )

        # check first
        if self.result_recorder.terminal_check():
            logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
            sys.exit(0)

        self.current_index = self.result_recorder.current_index
        self.last_update_time = None
        # Load scenario task
        logger.info('Loaded Random Tester')

    def run(self):
        # generate seed scenario from the task
        scenario = None
        logger.info('Start testing')
        self.last_update_time = time.time()
        while True:
            logger.info('============ Iteration {} ============'.format(self.current_index))
            if self.result_recorder.terminal_check():
                logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                return

            mutated_scenario = self.adapter.mutation_adaptor(
                scenario
            )
            mutated_scenario.idx = self.current_index
            mutated_scenario.export()
            self.current_index += 1
            scenario_recorder = self.scenario_runner.run(mutated_scenario)
            if scenario_recorder is None:
                logger.warning(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                continue
            self.result_recorder.update(
                mutated_scenario.idx,
                delta_time=time.time() - self.last_update_time,
                result_details=scenario_recorder.to_json()
            )
            self.last_update_time = time.time()