import os
import shutil
import time

from loguru import logger
from threading import Thread, Lock

from common.data_provider import DataProvider
from common.logger_tools import get_instance_logger

from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge
from scenario_runner.drive_simulator.ApolloSim.oracle import oracle_library
from scenario_runner.drive_simulator.ApolloSim.library import agent_library, VehicleControl

from .wrapper import ApolloWrapper
from scenario_runner.drive_simulator.ApolloSim.config.apollo import ApolloConfig
from scenario_runner.drive_simulator.ApolloSim.recorder import ScenarioRecorder

class ApolloAgent:
    """
    Traffic -> ApolloWrapper
    ApolloWrapper -> Scenario
    Runner to start and running in the scenario, for passing information to ApolloWrapper
    TODO: may need rerouting once stuck?
    """
    PERCEPTION_FREQUENCY = 25.0 #25.0  # 10.0 # 25.0
    LOCALIZATION_FREQUENCY = 100.0

    _update_lock = Lock()

    def __init__(
            self,
            apollo_config: ApolloConfig,
            traffic_bridge: TrafficBridge,
            result_recorder: ScenarioRecorder
    ):
        """
        Constructor
        """
        self.traffic_bridge = traffic_bridge
        self.apollo_config = apollo_config
        self.result_recorder = result_recorder

        self.idx = self.apollo_config.idx

        # create logger if debug
        self.debug = DataProvider.debug
        if self.debug:
            debug_folder = DataProvider.debug_folder()
            apollo_debug_folder = os.path.join(debug_folder, f'apollo')
            if not os.path.exists(apollo_debug_folder):
                os.makedirs(apollo_debug_folder)

            log_file = os.path.join(apollo_debug_folder, f"{self.apollo_config.idx}.log")
            self.logger = get_instance_logger(f"apollo_instance_{self.apollo_config.idx}", log_file)
            self.logger.info(f"Logger initialized for apollo_instance_{self.apollo_config.idx}")
        else:
            self.logger = None

        # local parameters
        self.thread_perception = None
        self.thread_chassis_localization = None
        self.thread_oracle = None
        self.thread_state = None

        self.termination_on_failure = self.apollo_config.terminate_on_failure
        self.apollo_wrapper = ApolloWrapper(self.apollo_config, self.logger)

        # publish state
        # init current state
        agent_class = agent_library.get(self.apollo_config.category)
        self.agent = agent_class(
            idx=self.apollo_config.idx,
            location=self.apollo_config.initial_waypoint.location,
            role=self.apollo_config.role
        )

        # publish information to the bridge
        self.traffic_bridge.register_actor(
            self.agent.id,
            self.agent,
        )

        # register oracles
        # NOTE that all apollo have the same oracle instances
        self.oracle_instances = list()
        for oracle_key, oracle_cfg in self.oracle_list().items():
            oracle_cls = oracle_library.get(oracle_key)
            self.oracle_instances.append(
                oracle_cls(
                    self.agent.id,
                    self.traffic_bridge,
                    **oracle_cfg
                )
            )

        self.local_spend_time = 0.0

    def oracle_list(self):
        return {
            'oracle.collision': {
            },
            'oracle.stuck': {
                'speed_threshold': 0.3, # m/s
                'max_stuck_time': 90 # seconds
            },
            'oracle.timeout': {
                'time_limit': 600 # 600 seconds
            },
            'oracle.destination': {
                'destination': self.apollo_config.route[-1],
                'threshold': 5.0
            }
        }

    def _tick_chassis_localization(self):
        self.local_spend_time = 0.0
        while not self.traffic_bridge.is_termination:
            step_start_time = time.time()
            # 0. mix route here
            self.apollo_wrapper.publish_route(self.local_spend_time)
            # 1. publish current state to apollo
            self.apollo_wrapper.publish_chassis(self.agent)  # TODO: implement this function 3q
            self.apollo_wrapper.publish_localization(self.agent)
            # update frame for localization
            step_end_time = time.time()
            if step_end_time - step_start_time > 1 / self.LOCALIZATION_FREQUENCY:
                pass
            else:
                time.sleep(1 / self.LOCALIZATION_FREQUENCY - (step_end_time - step_start_time))
            step_end_time = time.time()
            self.local_spend_time = step_end_time - step_start_time

    def _tick_perception(self):
        while not self.traffic_bridge.is_termination:
            step_start_time = time.time()
            # Update Traffic Information to Apollo
            perception_obs = self.traffic_bridge.get_actors()
            traffic_light_config = self.traffic_bridge.get_traffic_light()
            self.apollo_wrapper.publish_traffic_light(traffic_light_config)
            self.apollo_wrapper.publish_obstacles(perception_obs)

            # update frame
            step_end_time = time.time()
            if step_end_time - step_start_time > 1 / self.PERCEPTION_FREQUENCY:
                pass
            else:
                time.sleep(1 / self.PERCEPTION_FREQUENCY - (step_end_time - step_start_time))

    def _tick_state(self):
        delta_time = float(1 / float(DataProvider.SIM_FREQUENCY))
        while not self.traffic_bridge.is_termination:
            # with self._update_lock:
            agent_control = VehicleControl(self.apollo_wrapper.throttle_percentage, self.apollo_wrapper.brake_percentage, self.apollo_wrapper.steering_percentage)
            self.agent.apply_control(agent_control)
            time.sleep(delta_time)

    def _tick_oracle(self):
        """
        :return:
        """
        while (not self.traffic_bridge.is_termination) and len(self.oracle_instances) > 0:
            should_terminal = False
            for oracle_instance in self.oracle_instances:
                oracle_result, oracle_feedback = oracle_instance.tick()
                self.result_recorder.update_scenario_feedback(self.idx, oracle_instance.oracle_name, oracle_feedback)
                self.result_recorder.update_traffic_events(self.idx, oracle_instance.oracle_name, oracle_result)
                if oracle_result is not None and self.termination_on_failure:
                    should_terminal = True
            if should_terminal:
                self.traffic_bridge.set_termination()

    def start(self):
        """
        Starts to forward localization
        """
        if self.traffic_bridge.is_termination:
            logger.debug(f"Apollo Agent: is_termination is {self.traffic_bridge.is_termination}, please check!!!")
            return

        self.thread_chassis_localization = Thread(target=self._tick_chassis_localization)
        self.thread_chassis_localization.start()

        self.thread_perception = Thread(target=self._tick_perception)
        self.thread_perception.start()

        self.thread_oracle = Thread(target=self._tick_oracle)
        self.thread_oracle.start()

        self.thread_state = Thread(target=self._tick_state)
        self.thread_state.start()

    def stop(self):

        self.thread_state.join()
        self.thread_oracle.join()
        self.thread_perception.join()
        self.thread_chassis_localization.join()

        self.thread_state = None
        self.thread_oracle = None
        self.thread_perception = None
        self.thread_chassis_localization = None

        self.apollo_config = None
        self.apollo_wrapper.stop()

    ######### Other Tools #########
    def recorder_operator(self, operation, record_folder=None, scenario_id=None):
        self.apollo_wrapper.recorder_operator(
            operation,
            record_folder=record_folder,
            scenario_id=scenario_id
        )

    def move_recording(self, apollo_record_folder: str, scenario_id: str, delete_flag: bool = True):
        # May need parser of recording

        local_apollo_recording_folder = os.path.join(DataProvider.apollo_recording_folder(), scenario_id)
        if os.path.exists(local_apollo_recording_folder):
            shutil.rmtree(local_apollo_recording_folder)

        self.apollo_wrapper.move_recording(
            apollo_record_folder,
            scenario_id,
            local_apollo_recording_folder,
            delete_flag
        )
        self.result_recorder.update_apollo_record(self.idx, local_apollo_recording_folder)
        logger.info(f'Move Apollo Recording for {self.idx} to {local_apollo_recording_folder}')