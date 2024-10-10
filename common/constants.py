import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dit = os.path.dirname(current_dir)

class Constants:

    SIM_FREQUENCY = 25.0  # HZ

    # ApolloSim requirements
    PROJECT_ROOT = root_dit
    APOLLO_ROOT = f"{root_dit}/Apollo"
    OUTPUT_ROOT = f"{root_dit}/outputs"
    APOLLOSIM_MAP_ROOT = f"{root_dit}/data/ApolloSim_maps"