import os
import sys
import copy
import traceback

import numpy as np

from omegaconf import DictConfig, OmegaConf
from loguru import logger
import time

from common.logger_tools import get_instance_logger
from .models.coverage import CoverageModel
from .adapter.ApolloSim import ApolloSimAdaptor, ScenarioAdaptor
from common.data_provider import DataProvider
from ..common.result_recorder import ResultRecorder
from .. import register_tester

@register_tester("tester.behavexplor")
class BehAVExplorFuzzer:

    def __init__(
            self,
            scenario_runner,
            scenario_cfg: DictConfig,
            algorithm_cfg: DictConfig,
            oracle_cfg: DictConfig
    ):
        self.scenario_runner = scenario_runner
        self.map_name = scenario_cfg.map_name
        algorithm_parameters = algorithm_cfg.get("parameters", None)
        if algorithm_parameters is None:
            raise RuntimeError('Please config parameters')

        self.run_hour = algorithm_parameters.get("run_hour", 4)
        self.num_vehicle = algorithm_parameters.get("num_vehicle", 1)
        self.num_walker = algorithm_parameters.get("num_walker", 1)
        self.num_static = algorithm_parameters.get("num_static", 1)
        self.window_size = algorithm_parameters.get("window_size", 10)
        self.cluster_num = algorithm_parameters.get("cluster_num", 20)
        self.threshold_coverage = algorithm_parameters.get("threshold_coverage", 0.4)
        self.threshold_energy = algorithm_parameters.get("threshold_energy", 0.8)
        self.feature_resample = algorithm_parameters.get("feature_resample", 'linear')
        self.initial_corpus_size = algorithm_parameters.get("initial_corpus_size", 4)

        # Seed Scenario
        self.ego_start_lane_id = scenario_cfg.get("start_lane_id", None)
        self.ego_end_lane_id = scenario_cfg.get("end_lane_id", None)

        # config population
        self.save_root = DataProvider.output_folder()  # this is the fuzzing result folder to save the results
        self.result_recorder = ResultRecorder(
            self.save_root,
            self.run_hour
        )

        # create adapter
        DataProvider.oracle_cfg = copy.deepcopy(OmegaConf.to_container(oracle_cfg))
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

        self.current_index = self.result_recorder.current_index

        self.coverage_model = CoverageModel(
            self.window_size,
            self.cluster_num,
            self.threshold_coverage
        )

        self.corpus = list()  # save all elements in the fuzzing
        self.corpus_energy = list()
        self.corpus_fail = list()
        self.corpus_mutation = list()
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

        self.safe_violations = list()
        self.mutate_violations = list()
        self.nods_violations = list()

        self.last_update_time = None


    def _seed_selection(self):
        select_probabilities = copy.deepcopy(self.corpus_energy)
        select_probabilities = np.array(select_probabilities) + 1e-5
        select_probabilities /= (select_probabilities.sum())
        source_seed_index = np.random.choice(list(np.arange(0, len(self.corpus))), p=select_probabilities)
        return source_seed_index

    def run(self):
        # minimize is better
        logger.info('===== Start Fuzzer (BehAVExplor) =====')
        # generate initial scenario
        seed_scenario = None
        self.last_update_time = time.time()
        while len(self.corpus) < self.initial_corpus_size:
            mutated_scenario = self.adapter.mutation_adaptor(
                seed_scenario,
                mutation_stage='large'
            )
            mutated_scenario.idx = self.current_index
            mutated_scenario.export()
            self.current_index += 1
            scenario_recorder = self.scenario_runner.run(mutated_scenario)
            if scenario_recorder is None:
                logger.warning(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                continue
            fitness, ego_record, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
            mutated_seed = ScenarioAdaptor(
                mutated_scenario,
                ego_record,
                fitness
            )

            self.result_recorder.update(
                mutated_scenario.idx,
                delta_time=time.time() - self.last_update_time,
                result_details=scenario_recorder.to_json()
            )
            self.last_update_time = time.time()

            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
            self.logger_mutation.info(f"{mutated_seed.id},{mutated_seed.id},initial,1.0")

            # check conditions
            self.corpus.append(copy.deepcopy(mutated_seed))
            # the following items are used for energy
            self.corpus_fail.append(0)
            self.corpus_mutation.append(0)
            self.corpus_energy.append(1)

        # initialize the coverage model
        if len(self.corpus) > 0:
            try:
                m_start_time = time.time()
                self.coverage_model.initialize(self.corpus)
                m_end_time = time.time()
                logger.info('Coverage Spend Time: [=]{}[=]', m_end_time - m_start_time)
            except Exception as e:
                traceback.print_exc()
                pass

        while True:
            logger.info('============ Iteration {} ============'.format(self.current_index - self.initial_corpus_size))
            if self.result_recorder.terminal_check():
                logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                return

            source_seed_index = self._seed_selection() # select based on energy, high energy is better
            source_seed = self.corpus[source_seed_index]
            source_seed_energy = self.corpus_energy[source_seed_index]
            source_seed_fail = self.corpus_fail[source_seed_index]
            source_seed_mutation = self.corpus_mutation[source_seed_index]

            if source_seed_energy > self.threshold_energy:
                mutation_stage = 'small'
            else:
                mutation_stage = 'large'

            mutated_scenario = self.adapter.mutation_adaptor(
                source_seed.scenario,
                mutation_stage
            )
            mutated_scenario.idx = self.current_index
            mutated_scenario.export()
            self.current_index += 1

            scenario_recorder = self.scenario_runner.run(mutated_scenario)
            if scenario_recorder is None:
                logger.warning(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                continue
            fitness, ego_record, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
            mutated_seed = ScenarioAdaptor(
                mutated_scenario,
                ego_record,
                fitness
            )
            self.result_recorder.update(
                mutated_scenario.idx,
                delta_time=time.time() - self.last_update_time,
                result_details=scenario_recorder.to_json()
            )
            self.last_update_time = time.time()

            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
            self.logger_mutation.info(f"{mutated_seed.id},{source_seed.id},{mutation_stage},{source_seed_energy}")

            m_start_time = time.time()

            # add mutation
            source_seed_mutation += 1
            # update energy & fail
            benign = True
            if not is_complete:
                source_seed_fail += 1
                benign = False

            if mutation_stage == 'large':
                source_seed_energy = source_seed_energy - 0.15
            else:
                # update energy of source_seed
                delta_fail = source_seed_fail / float(source_seed_mutation)
                if benign:
                    delta_fail = min(delta_fail, 1.0)
                    delta_fail = -1 * (1 - delta_fail)
                else:
                    delta_fail = min(delta_fail, 1.0)

                delta_fitness = source_seed.fitness - mutated_seed.fitness  # min is better
                delta_select = -0.1
                source_seed_energy = source_seed_energy + 0.5 * delta_fail + 0.5 * np.tanh(delta_fitness) + delta_select

            # update information
            self.corpus_energy[source_seed_index] = float(np.clip(source_seed_energy, 1e-5, 4.0))
            self.corpus_fail[source_seed_index] = source_seed_fail
            self.corpus_mutation[source_seed_index] = source_seed_mutation
            m_end_time = time.time()
            logger.info('Energy Spend Time: [=]{}[=]', m_end_time - m_start_time)

            # calculate the diversity based on the record
            m_start_time = time.time()
            follow_up_seed_is_new, follow_up_seed_div, follow_up_seed_ab = self.coverage_model.feedback_coverage_behavior(mutated_seed)
            if follow_up_seed_is_new or mutated_seed.fitness < source_seed.fitness:
                self.corpus.append(copy.deepcopy(mutated_seed))
                self.corpus_fail.append(0)
                self.corpus_mutation.append(0)
                self.corpus_energy.append(1)
            m_end_time = time.time()
            logger.info('Coverage Spend Time: [=]{}[=]', m_end_time - m_start_time)