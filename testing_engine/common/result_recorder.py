import json
import os
import copy

from loguru import logger

class ResultRecorder(object):
    """
    This is saving the fuzzing results
    {
        "run_time": 0.
    }
    """
    def __init__(
            self,
            save_root: str,
            run_hour: float
    ):
        self.save_root = save_root
        self.run_hour = run_hour
        self.file_path = os.path.join(self.save_root, 'result.json')
        self.result_data = {}
        self.load()

    @property
    def current_index(self) -> int:
        return self.result_data['current_index']

    def terminal_check(self) -> bool:
        if float(self.result_data['run_time']) / 3600.0 > self.run_hour:
            logger.info(f"Finish fuzzer as reach the time limited: {float(self.result_data['run_time']) / 3600.0}/{self.run_hour}")
            return True
        return False

    def load(self):
        if os.path.exists(self.file_path):
            # must be resumed here
            with open(self.file_path, 'r') as f:
                self.result_data = json.load(f)
        else:
            self.result_data = {
                "run_time": 0.0, # seconds
                "current_index": 0,
                "overview": {},
                "details": {}
            }

    def update(
            self,
            current_index: int,
            delta_time: float, # seconds
            result_details: dict
    ):
        self.result_data['current_index'] = current_index
        self.result_data['run_time'] += delta_time
        self.result_data['details'][current_index] = copy.deepcopy(result_details)
        self.result_data['overview'][current_index] = result_details['scenario_overview']

        with open(self.file_path, 'w') as f:
            json.dump(self.result_data, f, indent=4)