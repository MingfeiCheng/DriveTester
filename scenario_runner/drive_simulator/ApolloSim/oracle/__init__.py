import os
import importlib
from scenario_runner.drive_simulator.ApolloSim.traffic_messenger import TrafficBridge

class BaseOracle(object):

    oracle_name = 'oracle.unknown'

    def __init__(self, agent_id: str, traffic_bridge: TrafficBridge, **kwargs):
        self.agent_id = agent_id
        self.traffic_bridge = traffic_bridge

    def tick(self):
        raise NotImplementedError("Not implemented yet for {}".format(type(self).__name__))

oracle_library = {}

def register_oracle(name):
    def decorator(cls):
        oracle_library[name] = cls
        return cls
    return decorator

# Dynamically import all modules in the agents folder
def discover_oracle(package_name="oracle"):
    package_dir = os.path.dirname(__file__)
    for module_name in os.listdir(package_dir):
        if module_name.endswith(".py") and module_name != "__init__.py":
            module_name = module_name[:-3]  # Strip .py extension
            importlib.import_module(f"scenario_runner.drive_simulator.ApolloSim.{package_name}.{module_name}")

# Discover and register agents when the package is imported
discover_oracle()