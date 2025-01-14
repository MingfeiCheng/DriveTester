from omegaconf import DictConfig
from drive_tester.scenario_runner import RunnerClass

class Fuzzer(object):

    def __init__(
            self,
            scenario_runner: RunnerClass,
            fuzzer_config: DictConfig,
    ):
        self.scenario_runner = scenario_runner
        self.scenario_config = scenario_runner.scenario_config
        self.fuzzer_config = fuzzer_config

    def run(
            self
    ):
        raise NotImplementedError("Method run not implemented")