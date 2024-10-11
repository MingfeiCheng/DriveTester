import os
import importlib

tester_library = {}

def register_tester(name):
    def decorator(cls):
        tester_library[name] = cls
        return cls
    return decorator

# Dynamically import all modules in the agents folder
def discover_tester(package_name="testing_engine"):
    package_dir = os.path.dirname(__file__)
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file == "runner.py":
                # Create a module path from the folder structure
                rel_dir = os.path.relpath(root, package_dir)
                module_name = file[:-3]  # Strip .py extension
                full_module_name = f"{package_name}.{rel_dir.replace(os.sep, '.')}.{module_name}"
                importlib.import_module(full_module_name)
# Discover and register agents when the package is imported
discover_tester()