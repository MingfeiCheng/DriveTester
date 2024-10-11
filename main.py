import os
import sys
import hydra
import shutil

from importlib import import_module
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from common.data_provider import DataProvider
from testing_engine import tester_library
from scenario_runner import runner_library

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

@hydra.main(config_path='configs', config_name='random', version_base=None)
def main(cfg: DictConfig):
    # config parameters
    DataProvider.debug = cfg.system.debug
    DataProvider.resume = cfg.system.resume
    DataProvider.tag = f"{cfg.testing_engine.algorithm.name}_{cfg.scenario.map_name}_{cfg.scenario.start_lane_id}_{cfg.scenario.end_lane_id}_{cfg.system.tag}"

    if DataProvider.debug:
        level = 'DEBUG'
    else:
        level = 'INFO'

    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])

    logger.info(f'project root: {DataProvider.project_root}')
    logger.info(f'apollo root: {DataProvider.apollo_root}')
    logger.info(f'output root: {DataProvider.output_root}')

    # confirm the output folder for the run
    output_folder = DataProvider.output_folder()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        if not DataProvider.resume:
            logger.warning(f"Due to the output folder exists, and not resume, delete!. {output_folder}")
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)

    logger.info(f'run save folder: {output_folder}')

    # save folder
    if not os.path.exists(DataProvider.output_folder()):
        os.makedirs(DataProvider.output_folder)

    logger_file = os.path.join(DataProvider.output_folder(), 'run.log')
    if os.path.exists(logger_file):
        os.remove(logger_file)
    _ = logger.add(logger_file, level=level)

    OmegaConf.save(config=cfg, f=os.path.join(DataProvider.output_folder(), 'run_config.yaml'))

    # create scenario runner
    runner_class = runner_library.get(f"runner.{cfg.scenario_runner.name}")
    logger.info(f'Load runner class from: {runner_class}')
    scenario_runner = runner_class(cfg.scenario_runner.parameters)

    # direct to specific method, such as mr, avfuzz..
    fuzz_class = tester_library.get(f"tester.{cfg.testing_engine.algorithm.name}")
    logger.info(f'Load fuzzing class from: {fuzz_class}')

    fuzz_instance = fuzz_class(
        scenario_runner,
        cfg.scenario,
        cfg.testing_engine.algorithm,
        cfg.testing_engine.oracle
    )
    fuzz_instance.run()

if __name__ == '__main__':
    main()
    logger.info('DONE Fuzzing!')