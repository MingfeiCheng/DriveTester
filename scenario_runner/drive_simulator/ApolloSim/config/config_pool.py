from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, TypeVar

from .basic import WaypointConfig

@dataclass
class BaseConfig:
    idx: int
    role: str
    category: str
    initial_waypoint: WaypointConfig
    mutable: bool
    trigger_time: float

    def __init__(
            self,
            idx: int,
            category: str, # for registering agent
            initial_waypoint: WaypointConfig,
            mutable: bool = True,
            trigger_time: float = 0.0,
            role: str = 'unknown',  # for distinguish in the simulator
    ):
        self.idx = idx
        self.role = role
        self.category = category
        self.initial_waypoint = initial_waypoint

        self.mutable = mutable
        self.trigger_time = trigger_time

    @classmethod
    def from_json(cls, json_node: Dict) -> 'BaseConfigClass':
        json_node['initial_waypoint'] = WaypointConfig.from_json(json_node['initial_waypoint'])
        return cls(**json_node)

    def json_data(self) -> Dict:
        return asdict(self)

BaseConfigClass = TypeVar("BaseConfigClass", bound=BaseConfig)

@dataclass
class BaseConfigPool:
    configs: List[BaseConfigClass]
    ids: List[int]
    fixed_ids: List[int]
    mutant_ids: List[int]

    def __init__(self, configs: List):
        self.configs = configs
        self.ids = list()
        self.fixed_ids = list()
        self.mutant_ids = list()

        for item in self.configs:
            self.ids.append(item.idx)
            if item.mutable:
                self.mutant_ids.append(item.idx)
            else:
                self.fixed_ids.append(item.idx)

    def get_config(self, idx) -> Optional[BaseConfigClass]:
        for _, _config in enumerate(self.configs):
            if _config.idx == idx:
                return _config
        return None

    def add_config(self, config: BaseConfigClass) -> bool:
        if config.idx in self.ids:
            # existing config
            return False

        self.configs.append(config)
        self.ids.append(config.idx)
        if config.mutable:
            self.mutant_ids.append(config.idx)
        else:
            self.fixed_ids.append(config.idx)
        return True

    def remove_config(self, idx: int):

        if idx not in self.ids:
            return
        target_config = None
        for item in self.configs:
            if item.idx == idx:
                target_config = item
                break

        if target_config is None or (not target_config.mutable):
            return

        self.configs.remove(target_config)
        self.ids.remove(idx)
        self.mutant_ids.remove(idx)

    def update_config(self, idx: int, config: BaseConfigClass) -> bool:

        if idx not in self.ids:
            return False

        for config_index, _config in enumerate(self.configs):
            if _config.idx == idx:
                self.configs[config_index] = config
                break
        return True

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'BaseConfigPoolClass':
        configs_js = json_node['configs']
        configs = list()
        for a_i, a_js in enumerate(configs_js):
            configs.append(BaseConfigClass.from_json(a_js))
        json_node['configs'] = configs
        return cls(**json_node)

BaseConfigPoolClass = TypeVar("BaseConfigPoolClass", bound=BaseConfigPool)
