import os
import importlib
from .base import AgentClass, VehicleControl, WalkerControl

agent_library = {}

def register_agent(name):
    def decorator(cls):
        agent_library[name] = cls
        return cls
    return decorator

def discover_agents(package_name="library"):
    package_dir = os.path.dirname(__file__)
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py" and file != "base.py":
                # Create a module path from the folder structure
                rel_dir = os.path.relpath(root, package_dir)
                module_name = file[:-3]  # Strip .py extension
                full_module_name = f"scenario_runner.drive_simulator.ApolloSim.{package_name}.{rel_dir.replace(os.sep, '.')}.{module_name}"
                importlib.import_module(full_module_name)

discover_agents()