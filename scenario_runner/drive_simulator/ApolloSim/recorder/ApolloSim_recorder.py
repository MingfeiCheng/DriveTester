import json
import threading

from typing import Dict, Optional

from scenario_runner.drive_simulator.ApolloSim.traffic_events import TrafficEvent

class ScenarioRecorder:
    """
    This saves all the apollo instances in the scenario.
    This scenario result only records the results of a scenario
    """

    _update_lock = threading.Lock()

    traffic_events: Dict[str, Dict[str, TrafficEvent]] # id: events
    traffic_record_path: str
    apollo_record_path: Dict[str, str]
    scenario_feedback: Dict[str, Dict[str, float]] # id: feed_name: feed_value

    timer: Dict[str, float]

    def __init__(self):
        self.traffic_events = {}
        self.traffic_record_path = None
        self.apollo_record_path = {}
        self.scenario_feedback = {}
        self.timer = {}

    def update_traffic_events(self, idx: str, oracle_name: str, traffic_event: Optional[TrafficEvent]):
        with self._update_lock:
            if idx not in self.traffic_events:
                self.traffic_events[idx] = {}
            self.traffic_events[idx][oracle_name] = traffic_event

    def update_apollo_record(self, idx: str, apollo_record: str):
        with self._update_lock:
            self.apollo_record_path[idx] = apollo_record

    def update_traffic_record(self, traffic_record: str):
        with self._update_lock:
            self.traffic_record_path = traffic_record

    def update_scenario_feedback(self, idx: str, feedback_name: str, feedback_value: float):
        with self._update_lock:
            if idx not in self.scenario_feedback:
                self.scenario_feedback[idx] = {}
            self.scenario_feedback[idx][feedback_name] = feedback_value

    def update_timer(self, component: str, consume_time: float):
        with self._update_lock:
            if component not in self.timer:
                self.timer[component] = consume_time

    def export_to_json(self, result_path: str):

        result = {
            "traffic_events": self.traffic_events,
            "traffic_record_path": self.traffic_record_path,
            "apollo_record_path": self.apollo_record_path,
            "scenario_feedback": self.scenario_feedback
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    def print_result(self):
        result = {
            "traffic_events": self.traffic_events,
            "traffic_record_path": self.traffic_record_path,
            "apollo_record_path": self.apollo_record_path,
            "scenario_feedback": self.scenario_feedback
        }

        # Printing traffic events
        print("Traffic Events:")
        for event_id, events in result["traffic_events"].items():
            print(f"  ID: {event_id}")
            print(f"    Events: {events}")

        # Printing traffic record path
        print("\nTraffic Record Path:")
        print(f"  {result['traffic_record_path']}")

        # Printing apollo record path
        print("\nApollo Record Path:")
        for apollo_id, path in result["apollo_record_path"].items():
            print(f"  {apollo_id}: {path}")

        # Printing scenario feedback
        print("\nScenario Feedback:")
        for feedback_id, feedback_dict in result["scenario_feedback"].items():
            print(f"  ID: {feedback_id}")
            for feed_name, feed_value in feedback_dict.items():
                print(f"    {feed_name}: {feed_value}")