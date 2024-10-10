from dataclasses import dataclass, asdict
from typing import Dict, List

from common.data_provider import DataProvider

@dataclass
class TrafficLightConfig:

    traffic_signals: List
    traffic_signals_status: Dict[str, str]

    def __init__(self):
        self.traffic_signals = DataProvider.map_parser.get_signals()
        self.traffic_signals_status = dict()
        for k in self.traffic_signals:
            self.traffic_signals_status[k] = 'GREEN'

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'TrafficLightConfig':
        return cls()