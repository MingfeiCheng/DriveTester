import time

from threading import Thread
from scenario_runner.drive_simulator.ApolloSim.config.traffic_light import TrafficLightConfig
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge

class TrafficLightAgent(object):
    category = 'traffic_light'
    role = 'traffic_light'

    def __init__(
            self,
            config: TrafficLightConfig,
            traffic_bridge: TrafficBridge,
    ):
        self.config = config
        self.traffic_bridge = traffic_bridge
        self.start_time = time.time()
        self.thread_run = None
        self.curr_state = self.config.traffic_signals_status

    def start(self):
        if self.traffic_bridge.is_termination:
            return

        self.thread_run = Thread(target=self._run)
        self.thread_run.start()

    def stop(self):
        self.thread_run.join()
        self.thread_run = None

    def get_state(self):
        return self.curr_state

    def _run(self):
        self.start_time = time.time()
        while not self.traffic_bridge.is_termination:
            time.sleep(0.01)
            # update state
            self.tick()

    def tick(self):
        pass
