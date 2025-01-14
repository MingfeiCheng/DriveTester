import os
import time
import signal

from tqdm import tqdm

from apollo_sim.sim_env import SimEnv
from drive_tester.scenario_runner.ApolloSim.config.scenario import ScenarioConfig

class ScenarioManager:

    def __init__(
            self,
            sim_env: SimEnv,
            scenario_config: ScenarioConfig,
            output_folder: str,
            debug: bool = False,
    ):
        self.sim_env = sim_env
        self.scenario_config = scenario_config
        self.output_folder = output_folder
        self.debug = debug
        if self.debug:
            self.debug_folder = os.path.join(output_folder, "debug")
            if not os.path.exists(self.debug_folder):
                os.makedirs(self.debug_folder)

        signal.signal(signal.SIGINT, self.stop)

        self.manager_lst = list()
        self._initialize()

    def _initialize(self):
        raise NotImplementedError("This method should be implemented in subclass")

    def run(self):
        self.start() # start simulator & all managers
        bar = tqdm()
        while not self.sim_env.termination:
            bar.set_description(f"-> Scenario {self.scenario_config.idx}: Frame: {self.sim_env.frame_count} Game Time: {self.sim_env.game_time:.4f} Real Time: {self.sim_env.real_time:.4f}")
            time.sleep(0.05)
        self.stop() # stop simulator & all managers

    def start(self):
        self.sim_env.start()
        for manager in self.manager_lst:
            manager.start()

    def stop(self, signum = None, frame = None):
        self.sim_env.stop()
        for manager in self.manager_lst:
            manager.stop()

        if signum is not None:
            exit(signum)



