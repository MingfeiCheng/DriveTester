import os
import importlib

runner_library = {}

def register_runner(name):
    def decorator(cls):
        runner_library[name] = cls
        return cls
    return decorator

# Dynamically import all modules in the agents folder
def discover_runner(package_name="scenario_runner.runner"):
    package_dir = os.path.dirname(__file__)
    for module_name in os.listdir(package_dir):
        if module_name.endswith(".py") and module_name != "__init__.py":
            module_name = module_name[:-3]  # Strip .py extension
            importlib.import_module(f"{package_name}.{module_name}")

# Discover and register agents when the package is imported
discover_runner()