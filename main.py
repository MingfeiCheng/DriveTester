import os
import sys
import hydra

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from drive_tester.common.global_config import GlobalConfig
from drive_tester.registry import ENGINE_REGISTRY, RUNNER_REGISTRY

@hydra.main(config_path='drive_tester/configs', config_name='DriveTester', version_base=None)
def main(cfg: DictConfig):

    # config parameters
    GlobalConfig.debug = cfg.debug
    GlobalConfig.resume = cfg.resume
    GlobalConfig.save_record = cfg.save_record

    output_root = cfg.output_root
    r_output_root = os.path.join(output_root, cfg.scenario_runner.name, cfg.scenario.name, cfg.engine.name, cfg.tag)
    if not os.path.exists(r_output_root):
        os.makedirs(r_output_root)
    GlobalConfig.output_root = r_output_root
    output_root = GlobalConfig.output_root
    # print global config
    GlobalConfig.print()

    if GlobalConfig.debug:
        level = 'DEBUG'
    else:
        level = 'INFO'
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])
    logger_file = os.path.join(output_root, 'run.log')
    _ = logger.add(logger_file, level=level, mode="a")  # Use mode="a" for append
    # save configs
    OmegaConf.save(config=cfg, f=os.path.join(output_root, 'run_config.yaml'))

    # direct to specific method, such as mr, avfuzzer..
    fuzzer_class = ENGINE_REGISTRY.get(f"fuzzer.{cfg.engine.name}")
    logger.info(f'Load fuzzer class from: {fuzzer_class}')

    # create scenario runner class
    runner_class = RUNNER_REGISTRY.get(f"runner.{cfg.scenario_runner.name}")
    logger.info(f'Load runner class from: {runner_class}')

    runner_instance = runner_class(
        cfg.scenario_runner,
        cfg.scenario
    )
    fuzzer_instance = fuzzer_class(
        runner_instance,
        cfg.engine
    )
    fuzzer_instance.run()
    runner_instance.shutdown()

if __name__ == '__main__':
    main()
    logger.info('DONE DriveTester @.@!')