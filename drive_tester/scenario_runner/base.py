from omegaconf import DictConfig
from typing import TypeVar

class Runner(object):

    name = 'Undefined'

    def __init__(
            self,
            runner_config: DictConfig,
            scenario_config: DictConfig
    ):
        self.runner_config = runner_config
        self.scenario_config = scenario_config

    def run(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError("Method run not implemented")

RunnerClass = TypeVar('RunnerClass', bound=Runner)