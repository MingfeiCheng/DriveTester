import os
import copy
import sys
import time

from loguru import logger
from omegaconf import DictConfig

from common.data_provider import DataProvider
from common.logger_tools import get_instance_logger
from testing_engine.common.result_recorder import ResultRecorder
from .adapter.ApolloSim import ApolloSimAdaptor, ScenarioAdaptor

from .. import register_tester

@register_tester('tester.driverfuzz')
class DriveFuzzer:

    def __init__(
            self,
            scenario_runner,
            scenario_cfg: DictConfig,
            algorithm_cfg: DictConfig,
            oracle_cfg: DictConfig
    ):

        self.scenario_runner = scenario_runner
        self.map_name = scenario_cfg.map_name

        # Seed Scenario
        self.ego_start_lane_id = scenario_cfg.get("start_lane_id", None)
        self.ego_end_lane_id = scenario_cfg.get("end_lane_id", None)

        algorithm_parameters = algorithm_cfg.get("parameters", None)
        if algorithm_parameters is None:
            raise RuntimeError('Please config parameters')
        self.run_hour = algorithm_parameters.get("run_hour", 4)
        self.num_vehicle = algorithm_parameters.get("num_vehicle", 1)
        self.num_walker = algorithm_parameters.get("num_walker", 1)
        self.num_static = algorithm_parameters.get("num_static", 1)

        # config population
        self.save_root = DataProvider.output_folder()  # this is the fuzzing result folder to save the results
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
            self.num_static,
        )

        self.current_index = self.result_recorder.current_index
        self.best_seed = None

        debug_folder = os.path.join(DataProvider.output_folder(), 'debug/algorithm')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        fitness_log_file = os.path.join(debug_folder, f"fitness.log")
        mutation_log_file = os.path.join(debug_folder, f"mutation.log")
        if os.path.isfile(fitness_log_file):
            os.remove(fitness_log_file)
        if os.path.isfile(mutation_log_file):
            os.remove(mutation_log_file)

        self.logger_fitness = get_instance_logger(f"fitness", fitness_log_file)
        self.logger_mutation = get_instance_logger("mutation", mutation_log_file)

        self.logger_fitness.info("Logger initialized for fitness")
        self.logger_mutation.info("Logger initialized for mutation")
        self.last_update_time = None

    def run(self):
        # minimize is better
        logger.info('===== Start Fuzzer (DriveFuzz) =====')
        # generate initial scenario

        self.last_update_time = time.time()
        while True:

            if self.result_recorder.terminal_check():
                logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                return

            if self.best_seed:
                mutate_scenario = copy.deepcopy(self.best_seed.scenario)
            else:
                mutate_scenario = None

            mutate_scenario = self.adapter.mutation_adaptor(mutate_scenario)
            mutate_scenario.idx = self.current_index
            mutate_scenario.export()
            self.current_index += 1
            scenario_recorder = self.scenario_runner.run(mutate_scenario)
            if scenario_recorder is None:
                logger.warning(f'The scenario {mutate_scenario.idx} has bugs, continue to next')
                continue

            self.result_recorder.update(
                mutate_scenario.idx,
                delta_time=time.time() - self.last_update_time,
                result_details=scenario_recorder.to_json()
            )
            self.last_update_time = time.time()

            fitness, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
            if is_complete:
                if self.best_seed is None or fitness < self.best_seed.fitness:
                    self.best_seed = ScenarioAdaptor(
                        mutate_scenario,
                        fitness
                    )
                self.logger_fitness.info(f"{self.current_index},{self.best_seed.fitness},{is_complete}")
            else:
                self.best_seed = None
                self.logger_fitness.info(f"{self.current_index},{fitness},{is_complete}")