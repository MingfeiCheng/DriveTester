import os
import importlib

from drive_tester.scenario_runner.base import Runner, RunnerClass
#
# runner_library = {}
#
# def register_runner(name):
#     def decorator(cls):
#         runner_library[name] = cls
#         return cls
#     return decorator
#
# # Dynamically import all modules in the agents folder
# def discover_runner(package_name="scenario_runner"):
#     package_dir = os.path.dirname(__file__)
#     for root, _, files in os.walk(package_dir):  # Walk through subdirectories
#         for file in files:
#             if file.endswith(".py") and file != "__init__.py" and file != "base.py":
#                 # Construct module path relative to the package
#                 relative_path = os.path.relpath(os.path.join(root, file), package_dir)
#                 module_name = relative_path.replace(os.sep, ".")[:-3]  # Replace path separators with '.' and strip .py
#
#                 # Import the module dynamically
#                 full_module_name = f"{package_name}.{module_name}"
#                 importlib.import_module(full_module_name)
#
# # Discover and register agents when the package is imported
# discover_runner()