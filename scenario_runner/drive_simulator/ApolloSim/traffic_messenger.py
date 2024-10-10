import copy
import threading
import time

from typing import Dict, Any, Optional

from .library import AgentClass

class TrafficBridge(object):
    """
    This is the bridge for communication between different agents in the simulator, including Apollo instances
    """
    _actor_state: Dict[Any, AgentClass]
    _traffic_light_config: Optional[Dict[str, Any]]

    _is_termination: bool

    _update_lock = threading.Lock()

    def __init__(self):
        self.reset()

    def reset(self):
        with self._update_lock:
            self._actor_state = dict()
            self._traffic_light_config = dict()
            self._is_termination = False

    def recording(self) -> Dict[str, Any]:
        with self._update_lock:
            return copy.deepcopy(
                {
                    "timestamp": time.time(),
                    "actor_state": self._actor_state,
                    "traffic_light_config": self._traffic_light_config,
                }
            )

    def cleanup(self):
        with self._update_lock:
            self._actor_state.clear()
            self._traffic_light_config = None
            self._is_termination = False

    @property
    def is_termination(self) -> bool:
        return self._is_termination

    def set_termination(self):
        with self._update_lock:
            self._is_termination = True

    def register_actor(self, actor_id: Any, actor_state: AgentClass):
        if actor_id in self._actor_state:
            raise KeyError(f"actor_id {actor_id} exists, please check.")

        with self._update_lock:
            self._actor_state[actor_id] = actor_state # register for initial

    def update_actor(self, actor_id: Any, actor_state: AgentClass):
        if actor_id not in self._actor_state:
            raise KeyError(f"actor_id {actor_id} exists, please check and register first.")

        with self._update_lock:
            self._actor_state[actor_id] = actor_state

    def remove_actor(self, actor_id: Any):
        with self._update_lock:
            if actor_id in self._actor_state:
                del self._actor_state[actor_id]

    def get_actors(self) -> Dict[Any, AgentClass]:
        # Note that, this is use deepcopy, in case of modification by other threads
        return copy.deepcopy(self._actor_state)

    def query_state(self, idx: Any) -> AgentClass:
        actor_state = self._actor_state.get(idx, None)
        if actor_state is None:
            raise RuntimeError(f"Actor {idx} is None, Please confirm with developer.")
        return actor_state

    # For traffic light
    def update_traffic_light(self, config: Dict[str, Any]):
        with self._update_lock:
            self._traffic_light_config = copy.deepcopy(config)

    def get_traffic_light(self) -> Dict[str, Any]:
        return self._traffic_light_config