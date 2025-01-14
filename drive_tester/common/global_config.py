import os

from loguru import logger

class GlobalConfig:

    # Fuzzer
    debug: bool = False
    resume: bool = True
    save_record: bool = False
    output_root: str = "" # also used in fuzzing

    # tmp and dynamic folders
    curr_scenario_folder: str = ""

    @staticmethod
    def scenario_folder(scenario_idx: str):
        scenario_folder = os.path.join(GlobalConfig.output_root, f'scenario/{scenario_idx}')
        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)
        return scenario_folder

    @staticmethod
    def print():
        logger.info("Global Config:")
        logger.info(f"  debug: {GlobalConfig.debug}")
        logger.info(f"  resume: {GlobalConfig.resume}")
        logger.info(f"  save_record: {GlobalConfig.save_record}")
        logger.info(f"  output_root: {GlobalConfig.output_root}")