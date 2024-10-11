import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

class Constants:

    SIM_FREQUENCY = 25.0  # HZ

    # ApolloSim requirements
    PROJECT_ROOT = root_dir
    APOLLO_ROOT = f"{root_dir}/Apollo"
    OUTPUT_ROOT = f"{root_dir}/outputs"
    APOLLOSIM_MAP_ROOT = f"{root_dir}/data/ApolloSim_maps"