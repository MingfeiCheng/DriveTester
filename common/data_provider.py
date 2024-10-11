import os

from omegaconf import DictConfig

from common.constants import Constants

class DataProvider:

    SIM_FREQUENCY = Constants.SIM_FREQUENCY

    # some global configs
    debug: bool = True

    # parameters
    resume: bool = True # resume from previous one
    tag: str = "" # tag for recording this run
    map_name: str = ""
    container_name: str = None # prefix

    map_parser = None
    oracle_cfg: dict = {}

    # saver flags
    save_traffic_recording: bool = True # this is for saving traffic recording -> TODO: change to online io
    save_apollo_recording: bool = True # this is for saving apollo recordings

    # folders
    project_root: str = Constants.PROJECT_ROOT
    apollo_root: str = Constants.APOLLO_ROOT
    map_root: str = Constants.APOLLOSIM_MAP_ROOT
    output_root: str = Constants.OUTPUT_ROOT # Should be config

    container_root_folder: str = '/apollo' # TODO: confirm this folder

    @staticmethod
    def debug_folder():
        debug_folder = os.path.join(DataProvider.output_folder(), 'debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        return debug_folder

    @staticmethod
    def traffic_recording_folder():
        traffic_recording_folder = os.path.join(DataProvider.output_folder(), 'traffic')
        if not os.path.exists(traffic_recording_folder):
            os.makedirs(traffic_recording_folder)
        return traffic_recording_folder

    @staticmethod
    def apollo_recording_folder():
        apollo_recording_folder = os.path.join(DataProvider.output_folder(), 'apollo')
        if not os.path.exists(apollo_recording_folder):
            os.makedirs(apollo_recording_folder)
        return apollo_recording_folder

    @staticmethod
    def scenario_folder():
        scenario_folder = os.path.join(DataProvider.output_folder(), 'scenario')
        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)
        return scenario_folder

    @staticmethod
    def container_record_folder():
        container_record_folder = os.path.join(DataProvider.container_root_folder, 'ApolloSim/records')
        return container_record_folder

    @staticmethod
    def output_folder():
        output_folder = os.path.join(DataProvider.output_root, DataProvider.tag)
        return output_folder