import os
import sys
import copy
import numpy as np
import time
from typing import List
from omegaconf import DictConfig
from loguru import logger

from common.logger_tools import get_instance_logger
from testing_engine.avfuzzer.adapter.ApolloSim import ApolloSimAdaptor, ScenarioAdaptor
from common.data_provider import DataProvider
from ..common.result_recorder import ResultRecorder
from .. import register_tester

@register_tester("tester.avfuzzer")
class AVFuzzer:

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
        self.local_run_hour = algorithm_parameters.get("local_run_hour", 1)
        self.num_vehicle = algorithm_parameters.get("num_vehicle", 1)
        self.num_walker = algorithm_parameters.get("num_walker", 1)
        self.num_static = algorithm_parameters.get("num_static", 1)
        self.population_size = algorithm_parameters.get("population_size", 4)
        self.pm = algorithm_parameters.get("pm", 0.6)
        self.pc = algorithm_parameters.get("pc", 0.6)

        self.best_seed = None
        self.best_fitness_after_restart = 10
        self.best_fitness_lst = []

        self.minLisGen = 5  # Min gen to start LIS
        self.curr_population = list()
        self.prev_population = list()
        self.last_restart_iteration = 0

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
            oracle_cfg
        )

        # check first
        if self.result_recorder.terminal_check():
            logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
            sys.exit(0)

        self.last_result_time = None
        self.current_index = self.result_recorder.current_index
        self.current_iteration = 0

        debug_folder = os.path.join(DataProvider.output_folder(), 'debug/algorithm')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        fitness_log_file = os.path.join(debug_folder, f"fitness.log")
        if os.path.isfile(fitness_log_file):
            os.remove(fitness_log_file)

        self.logger_fitness = get_instance_logger(f"fitness", fitness_log_file)
        self.logger_fitness.info("Logger initialized for fitness")


    def _seed_selection(self, curr_population: List[ScenarioAdaptor], prev_population: List[ScenarioAdaptor]):
        # fitness -> min is better
        tmp_population = curr_population + prev_population

        tmp_fitness = list()
        for i in range(len(tmp_population)):
            tmp_p_i_fitness = tmp_population[i].fitness
            tmp_fitness.append(tmp_p_i_fitness + 1e-5)

        tmp_fitness_sum = float(sum(tmp_fitness))
        tmp_probabilities = np.array([(tmp_f / tmp_fitness_sum) for tmp_f in tmp_fitness])
        tmp_probabilities = 1 - np.array(tmp_probabilities)
        tmp_probabilities /= tmp_probabilities.sum()

        next_parent = list()
        # next_parent = [copy.deepcopy(self.best_seed)]
        for i in range(self.population_size):
            select = np.random.choice(tmp_population, p=tmp_probabilities)
            next_parent.append(copy.deepcopy(select))

        return next_parent

    def _run_global(self, start_time):
        # minimize is better
        logger.info('===== Start Fuzzer (AVFuzzer) =====')
        self.last_result_time = time.time()
        while len(self.curr_population) < self.population_size:
            mutated_scenario = self.adapter.mutation_adaptor(
                None
            )
            mutated_scenario.idx = self.current_index
            mutated_scenario.export()
            self.current_index += 1
            scenario_recorder = self.scenario_runner.run(mutated_scenario)
            if scenario_recorder is None:
                logger.warning(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                continue

            fitness, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
            mutated_seed = ScenarioAdaptor(
                mutated_scenario,
                fitness
            )
            self.result_recorder.update(
                mutated_scenario.idx,
                delta_time=time.time() - self.last_result_time,
                result_details=scenario_recorder.to_json()
            )
            self.last_result_time = time.time()

            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")

            if not is_complete:
                return

            self.curr_population.append(copy.deepcopy(mutated_seed))

        noprogress = False
        while True:  # i th generation.
            if self.result_recorder.terminal_check():
                logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                return

            self.current_iteration += 1
            logger.info('===== Iteration {} =====', self.current_iteration)

            if not noprogress:
                if self.current_iteration == 1 or len(self.prev_population) == 0:
                    self.prev_population = copy.deepcopy(self.curr_population)
                else:
                    self.prev_population = self._seed_selection(self.curr_population, self.prev_population)
                # mutation
                self.curr_population = self.adapter.mutation_adaptor(
                    self.prev_population,
                    self.pc,
                    self.pm
                )
            else:
                # restart
                for i in range(self.population_size):
                    mutated_scenario = self.adapter.mutation_adaptor(None)
                    mutated_seed = ScenarioAdaptor(
                        mutated_scenario,
                        1.0
                    )
                    self.curr_population[i] = copy.deepcopy(mutated_seed)
                self.best_seed = None

            # run
            for i in range(self.population_size):
                if self.result_recorder.terminal_check():
                    logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                    return

                curr_seed = self.curr_population[i]
                curr_seed.scenario.idx = self.current_index
                curr_seed.scenario.export()
                self.current_index += 1

                scenario_recorder = self.scenario_runner.run(curr_seed.scenario)
                if scenario_recorder is None:
                    logger.warning(f'The scenario {curr_seed.scenario.idx} has bugs, continue to next')
                    continue

                self.result_recorder.update(
                    curr_seed.scenario.idx,
                    delta_time=time.time() - self.last_result_time,
                    result_details=scenario_recorder.to_json()
                )
                self.last_result_time = time.time()

                fitness, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
                curr_seed.fitness = fitness
                self.logger_fitness.info(f"{curr_seed.id},{curr_seed.fitness}")

                # check conditions
                if not is_complete:
                    logger.info('Find violation, exit fuzzer.') # todo: restart
                    return

                self.curr_population[i] = curr_seed
                if self.best_seed is None or curr_seed.fitness < self.best_seed.fitness:
                    self.best_seed = copy.deepcopy(curr_seed)

            self.best_fitness_lst.append(self.best_seed.fitness)
            if noprogress:
                self.best_fitness_after_restart = self.best_seed.fitness
                noprogress = False

            # check progress with previous 5 fitness
            ave = 0
            if self.current_iteration >= self.last_restart_iteration + 5:
                for j in range(self.current_iteration - 5, self.current_iteration):
                    ave += self.best_fitness_lst[j]
                ave /= 5
                if ave <= self.best_seed.fitness:
                    self.last_restart_iteration = self.current_iteration
                    noprogress = True

            #################### End the Restart Process ###################
            if self.best_seed.fitness < self.best_fitness_after_restart:
                if self.current_iteration > (self.last_restart_iteration + self.minLisGen):  # Only allow one level of recursion
                    ################## Start LIS #################
                    lis_best_seed, find_bug = self._run_local(start_time)
                    if find_bug:
                        logger.info('Find violation or timeout, exit fuzzer.') # todo restart
                        return

                    if lis_best_seed.fitness < self.best_seed.fitness:
                        self.curr_population[0] = copy.deepcopy(lis_best_seed)
                    logger.info(' === End of Local Iterative Search === ')

    @staticmethod
    def _terminal_check(start_time: time, run_hour: float) -> bool:
        curr_time = time.time()
        t_delta = curr_time - start_time
        if t_delta / 3600.0 > run_hour:
            logger.info(f'Finish fuzzer as reach the time limited: {t_delta / 3600.0}/{run_hour}')
            return True
        return False

    def _run_local(self, start_time):

        local_start_time = time.time()

        local_pm = 0.6 * 1.5
        local_pc = 0.6 * 1.5

        curr_population = list()
        prev_population = list()

        local_best = copy.deepcopy(self.best_seed)

        logger.info('===== Start Local (AVFuzzer) =====')
        # generate initial scenario
        prev_population = [copy.deepcopy(local_best) for _ in range(self.population_size)]
        curr_population = self.adapter.mutation_adaptor(prev_population, local_pc, local_pm)
        for i in range(self.population_size):

            if self.result_recorder.terminal_check() or self._terminal_check(local_start_time, self.local_run_hour):
                logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                return

            mutated_seed = curr_population[i]
            mutated_seed.scenario.idx = self.current_index
            mutated_seed.scenario.export()
            self.current_index += 1

            scenario_result = self.scenario_runner.run(mutated_seed.scenario)
            if scenario_result is None:
                logger.warning(f'The scenario {mutated_seed.scenario.idx} has bugs, continue to next')
                continue

            self.result_recorder.update(
                mutated_seed.scenario.idx,
                delta_time=time.time() - self.last_result_time,
                result_details=scenario_result.to_json()
            )
            self.last_result_time = time.time()

            fitness, is_complete = self.adapter.feedback_adaptor(scenario_result)
            mutated_seed.fitness = fitness
            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")


            # check conditions
            if not is_complete:
                logger.info('Find violation, exit fuzzer.') # TODO: check this
                return None, True

            curr_population[i] = mutated_seed

        while True:  # i th generation.
            if self._terminal_check(start_time, self.run_hour) or self._terminal_check(local_start_time, self.local_run_hour):
                return copy.deepcopy(local_best), False

            prev_population = self._seed_selection(curr_population, prev_population)
            curr_population = self.adapter.mutation_adaptor(prev_population, local_pc, local_pm)
            # run
            for i in range(self.population_size):
                if self._terminal_check(start_time, self.run_hour) or self._terminal_check(local_start_time, self.local_run_hour):
                    return copy.deepcopy(local_best), False

                mutated_seed = curr_population[i]
                mutated_seed.scenario.idx = self.current_index
                mutated_seed.scenario.export()
                self.current_index += 1
                scenario_result = self.scenario_runner.run(mutated_seed.scenario)
                if scenario_result is None:
                    logger.warning(f'The scenario {mutated_seed.scenario.idx} has bugs, continue to next')
                    continue

                self.result_recorder.update(
                    mutated_seed.scenario.idx,
                    delta_time=time.time() - self.last_result_time,
                    result_details=scenario_result.to_json()
                )
                self.last_result_time = time.time()

                fitness, is_complete = self.adapter.feedback_adaptor(scenario_result)
                mutated_seed.fitness = fitness
                self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")

                # check conditions
                if not is_complete:
                    logger.info('Find violation, exit fuzzer.')  #
                    return None, True

                curr_population[i] = mutated_seed
                if local_best is None or mutated_seed.fitness < local_best.fitness:
                    local_best = copy.deepcopy(mutated_seed)

    def run(self):
        start_time = time.time()
        while True:
            if self._terminal_check(start_time, self.run_hour):
                return
            self._run_global(start_time)