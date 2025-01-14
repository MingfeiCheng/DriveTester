import os

from loguru import logger

class GlobalRunnerConfig:

    # SimEnv
    apollo_root: str = ""
    map_root: str = ""
    off_screen: bool = False
    sim_frequency: float = 25.0
    sim_port: float = 15000
    map_name: str = ""

    # Fuzzer
    debug: bool = False
    save_record: bool = False
    output_root: str = "" # also used in fuzzing

    # tmp and dynamic folders
    curr_scenario_folder: str = ""

    @staticmethod
    def scenario_folder(scenario_idx: str):
        scenario_folder = os.path.join(GlobalRunnerConfig.output_root, f'scenario/{scenario_idx}')
        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)
        return scenario_folder

    @staticmethod
    def print():
        logger.info("Global Runner Config [ApolloSim]:")
        logger.info(f"  apollo_root: {GlobalRunnerConfig.apollo_root}")
        logger.info(f"  map_root: {GlobalRunnerConfig.map_root}")
        logger.info(f"  map_name: {GlobalRunnerConfig.map_name}")
        logger.info(f"  off_screen: {GlobalRunnerConfig.off_screen}")
        logger.info(f"  sim_frequency: {GlobalRunnerConfig.sim_frequency}")
        logger.info(f"  sim_port: {GlobalRunnerConfig.sim_port}")
        logger.info(f"  debug: {GlobalRunnerConfig.debug}")
        logger.info(f"  save_record: {GlobalRunnerConfig.save_record}")
        logger.info(f"  output_root: {GlobalRunnerConfig.output_root}")
