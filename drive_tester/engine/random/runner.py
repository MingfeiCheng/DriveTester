import sys
import time

from loguru import logger
from omegaconf import DictConfig

from drive_tester.common.result_recorder import ResultRecorder
from drive_tester.common.global_config import GlobalConfig
from drive_tester.engine.base import Fuzzer
from drive_tester.registry import ENGINE_REGISTRY

@ENGINE_REGISTRY.register("fuzzer.random")
class RandomFuzzer(Fuzzer):
    """
    The random fuzzer is:
    """
    manager_name = "manager.random"

    def __init__(
            self,
            scenario_runner,
            fuzzer_config: DictConfig
    ):
        super(RandomFuzzer, self).__init__(
            scenario_runner,
            fuzzer_config
        )

        # config parameters
        self.run_hour = fuzzer_config.get("run_hour", 4)

        # create result recorder
        self.result_recorder = ResultRecorder(
            GlobalConfig.output_root,
            self.run_hour
        )
        ###### check first
        if self.result_recorder.terminal_check():
            logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
            sys.exit(0)

        self.current_index = self.result_recorder.current_index
        self.last_update_time = None

        # create adapter
        self.adapter = self._create_adapter()

        # Load scenario task
        logger.info('Loaded Random Tester')

    def _create_adapter(self):
        if self.scenario_runner.name == "ApolloSim":
            from drive_tester.engine.random.adapter import ApolloSimAdapter
            return ApolloSimAdapter(
                self.scenario_runner.simulator,
                self.scenario_config
            )
        else:
            raise ValueError(f'Not support scenario runner: {self.scenario_runner.name}')

    def run(self):
        # generate seed scenario from the task
        logger.info('Start testing')
        self.last_update_time = time.time()
        while True:
            logger.info('============ Iteration {} ============'.format(self.current_index))
            if self.result_recorder.terminal_check():
                logger.info(f'The fuzzing has finished {self.run_hour} hours. Exiting.')
                return

            # random mutation (adapter for fuzzer output -> scenario runner input)
            mutated_scenario = self.adapter.mutation_adapter(
                self.current_index
            )

            # runner
            self.current_index += 1
            scenario_result = self.scenario_runner.run(
                mutated_scenario,
                self.manager_name
            )

            # check the result (adapter for scenario runner output -> fuzzer input)
            fuzzer_result = self.adapter.result_adapter(scenario_result)

            if fuzzer_result is None:
                logger.error(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                return

            self.result_recorder.update(
                mutated_scenario.idx,
                delta_time=time.time() - self.last_update_time,
                result_details=fuzzer_result.to_json()
            )

            self.last_update_time = time.time()