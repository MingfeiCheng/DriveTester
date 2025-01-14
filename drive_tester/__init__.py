import os

from drive_tester.registry.utils import discover_modules

discover_modules(os.path.dirname(__file__), package_name="drive_tester")

__version__ = "7.0.0"  # Replace with your actual version
