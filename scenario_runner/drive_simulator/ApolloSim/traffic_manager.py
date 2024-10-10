import time
import pickle

from typing import Optional, List, Dict
from loguru import logger
from threading import Thread

from common.data_provider import DataProvider

from scenario_runner.configuration.ApolloSim_config import ApolloSimScenarioConfig

from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge
from scenario_runner.drive_simulator.ApolloSim.manager import TrafficLightManager
from scenario_runner.drive_simulator.ApolloSim.manager import StaticObstacleManager
from scenario_runner.drive_simulator.ApolloSim.manager import VehicleManager
from scenario_runner.drive_simulator.ApolloSim.manager import WalkerManager
from scenario_runner.drive_simulator.ApolloSim.manager import ApolloManager

class TrafficManager:

    local_spend_time: float
    running: bool

    apollo_manager: ApolloManager
    vehicle_manager: VehicleManager
    walker_manager: WalkerManager
    static_obstacle_manager: StaticObstacleManager
    traffic_light_manager: TrafficLightManager

    thread_observation: Thread
    thread_recording: Thread

    traffic_recording: Dict[str, List]

    def __init__(
            self,
            scenario_config: ApolloSimScenarioConfig
    ):
        self.scenario_config = scenario_config
        self.bridge = TrafficBridge()
        self.scenario_result = ScenarioRecorder()

        # inner parameters
        self.local_spend_time = 0.0
        self.thread_observation = None
        self.thread_recording = None
        self.traffic_recording = {}

        self.apollo_manager = ApolloManager(
            self.scenario_config.idx,
            self.scenario_config.ego_config_pool,
            self.bridge,
            self.scenario_result
        )

        self.vehicle_manager = VehicleManager(
            self.scenario_config.waypoint_vehicle_config_pool,
            self.bridge
        )
        self.walker_manager = WalkerManager(
            self.scenario_config.waypoint_walker_config_pool,
            self.bridge
        )
        self.static_obstacle_manager = StaticObstacleManager(
            self.scenario_config.static_obstacle_config_pool,
            self.bridge
        )
        self.traffic_light_manager = TrafficLightManager(
            self.scenario_config.traffic_light_config,
            self.bridge
        )

    @property
    def is_termination(self):
        return self.bridge.is_termination

    def _tick_observation(self):
        self.local_spend_time = 0.0
        while not self.is_termination:
            step_start_time = time.time()
            self.traffic_light_manager.publish_state()
            self.apollo_manager.publish_state()
            self.vehicle_manager.publish_state()
            self.walker_manager.publish_state()
            self.static_obstacle_manager.publish_state()
            step_end_time = time.time()
            if step_end_time - step_start_time > 1 / float(DataProvider.SIM_FREQUENCY):
                pass
            else:
                time.sleep(1 / float(DataProvider.SIM_FREQUENCY) - (step_end_time - step_start_time))
            self.local_spend_time += (time.time() - step_start_time)

    def _tick_recording(self):
        recording_frame = 0
        while not self.is_termination:
            step_start_time = time.time()
            frame_recording = self.bridge.recording()
            self.traffic_recording[recording_frame] = frame_recording
            step_end_time = time.time()
            if step_end_time - step_start_time > 1 / float(DataProvider.SIM_FREQUENCY):
                pass
            else:
                time.sleep(1 / float(DataProvider.SIM_FREQUENCY) - (step_end_time - step_start_time))
            recording_frame += 1

    def start(self):
        """
        Starts to forward localization
        """
        if self.is_termination:
            logger.debug(f"Traffic Manager: is_termination is {self.is_termination}, please check!!!")
            return

        self.apollo_manager.start()
        self.vehicle_manager.start()
        self.walker_manager.start()
        self.static_obstacle_manager.start()
        self.traffic_light_manager.start()

        self.thread_observation = Thread(target=self._tick_observation)
        self.thread_observation.start()

        if DataProvider.save_traffic_recording:
            self.thread_recording = Thread(target=self._tick_recording)
            self.thread_recording.start()

    def stop(self, record_path: Optional[str] = None):

        self.thread_observation.join()
        self.thread_observation = None

        if DataProvider.save_traffic_recording:
            self.thread_recording.join()
            self.thread_recording = None

        self.apollo_manager.stop()
        self.vehicle_manager.stop()
        self.walker_manager.stop()
        self.static_obstacle_manager.stop()
        self.traffic_light_manager.stop()

        if record_path is not None and DataProvider.save_traffic_recording:
            self.save_recorder(record_path)

    def save_recorder(self, record_path: str):
        with open(record_path, 'wb') as f:
            pickle.dump(self.traffic_recording, f)
        logger.info(f"Saved traffic recording to {record_path}")
        self.scenario_result.update_traffic_record(record_path)
