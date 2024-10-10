import os
import subprocess
import time
import docker
import traceback

from typing import Optional
from .cyber_bridge import CyberBridge
from .dreamview import Dreamview
from loguru import logger
from common.data_provider import DataProvider

class ApolloContainer:
    """
    Class to represent Apollo container
    """
    APOLLO_MODULES = ['Routing', 'Prediction', 'Planning', 'Control']

    def __init__(
            self,
            name: str
    ) -> None:
        self.name = "apollo_dev_mingfeicheng" # test for name
        self.user = os.environ.get("USER")
        # create docker container if it is not exist
        self.hd_map = DataProvider.map_name

        self.dreamview = None
        self.dreamview_port = 8888

        self.bridge: Optional[CyberBridge] = None
        self.bridge_port = 9090

        self.create_container()

    def create_container(self):
        client = docker.from_env()
        has_create = False
        try:
            container = client.containers.get(self.name)
            has_create = True
            logger.info(f'Apollo instance {self.name} already exists')
        except Exception as e:
            pass

        if not has_create:
            logger.info(f'Apollo instance {self.name} does not exist -> Create...')
            start_script_path = os.path.join(DataProvider.project_root, "scripts/dev_start_ctn.sh")
            options = "-y -l -f"
            docker_script_dir = os.path.join(DataProvider.apollo_root, "docker", "scripts")
            cmd = f"bash {start_script_path} {options}"
            subprocess.run(
                cmd,
                env={
                    "CURR_DIR": docker_script_dir,
                    "DEV_CONTAINER": self.name,
                    "USER": os.environ.get("USER"),
                    "APOLLO_ROOT_DIR": DataProvider.apollo_root,
                },
                shell=True
            )

        self.start_container(restart=True)

    @property
    def host(self) -> str:
        """
        Gets the ip address of the container

        :type: str
        """
        assert self.is_running(), f'Container {self.name} is not running.'
        ctn = docker.from_env().containers.get(self.name)
        host = ctn.attrs['NetworkSettings']['IPAddress']
        if host == '':
            return 'localhost'
        return ctn.attrs['NetworkSettings']['IPAddress']

    def is_running(self) -> bool:
        """
        Checks if the container is running

        :returns: True if running, False otherwise
        :rtype: bool
        """
        try:
            return docker.from_env().containers.get(self.name).status == 'running'
        except Exception:
            return False

    def start_container(self, restart=False, wait_time=3.0):
        """
        Starts an Apollo container instance

        param bool restart: force container to restart
        """
        open_bridge = False
        open_dm = False
        while (not self.is_running()) or (not open_bridge) or (not open_dm):
            cmd = f'docker restart {self.name}'
            logger.info(f"Restart apollo container: {cmd}")
            _ = subprocess.run(cmd, shell=True)
            time.sleep(wait_time)

            open_bridge = self.start_bridge(2.0)
            open_dm = self.start_dreamview(self.hd_map, restart=True)
            wait_time += 1.0

    def stop_container(self, wait_time=1.0):
        """
        Starts an Apollo container instance

        param bool restart: force container to restart
        """
        cmd = f'docker kill {self.name}'
        logger.info(cmd)
        _ = subprocess.run(cmd, shell=True)
        time.sleep(wait_time)

    ##### Dreamview Operation #####
    def start_dreamview(self, hd_map, dv_mode= 'Mkz Lgsvl', apollo_type='Mkz_Example', wait_time=3.0, restart=False) -> bool:
        # setup dreamview
        if restart:
            cmd_op = 'restart'
        else:
            cmd_op = 'start'
        tries = 0.0
        while True:
            try:
                tries += 1
                if tries > 10:
                    logger.error('Dreamview can not be opened.')
                    return False
                cmd = f"docker exec --user {self.user} {self.name} ./scripts/bootstrap.sh {cmd_op}"
                logger.info(cmd)
                _ = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(wait_time)
                self.dreamview = Dreamview(self.host, self.dreamview_port)
                self.dreamview.set_hd_map(hd_map)
                self.dreamview.set_setup_mode(dv_mode)
                self.dreamview.set_vehicle(apollo_type)
                time.sleep(wait_time)
                logger.info('Apollo dreamview running at http://{}:{}', self.host, self.dreamview_port)
                logger.info(f"HD_MAP: {hd_map} DV_MODE: {dv_mode} APOLLO_TYPE: {apollo_type}")
            except Exception as e:
                logger.warning('Apollo dreamview may has some unexpected error:')
                logger.warning(traceback.format_exc())
                time.sleep(wait_time)
                # wait_time += 1.0
            else:
                break
        return True

    ##### Apollo Cyber Bridge #####
    def is_bridge_started(self) -> bool:
        """
        Checks if the bridge has been started already

        :returns: True if running, False otherwise
        :rtype: bool
        """
        try:
            b = CyberBridge(self.host, self.bridge_port)
            b.conn.close()
            return True
        except:
            return False

    def start_bridge(self, wait_time=3.0) -> bool:
        """
        Start cyber bridge
        """
        try:
            cmd = f"docker exec --user {self.user} -d {self.name} ./scripts/bridge.sh"
            _ = subprocess.run(cmd, shell=True)
            logger.info(cmd)
            time.sleep(wait_time)
            self.bridge = CyberBridge(self.host, self.bridge_port)
            logger.info(f'Apollo cyber bridge connected: {self.host}:{self.bridge_port}')
            # self.bridge.conn.close()
        except (ConnectionRefusedError, AssertionError):
            logger.warning('Apollo cyber bridge connection failed: ')
            logger.warning(traceback.format_exc())
            return False

        return True

    def stop_bridge(self):
        self.bridge.stop()

    ##### Apollo Modules Operation (Dreamview Version) #####
    def start_modules(self, wait_time=1.0):
        for dv_m in self.APOLLO_MODULES:
            self.dreamview.enable_module(dv_m, wait_time)

        logger.info(f'Started Apollo modules: {self.APOLLO_MODULES}')

    def start_sim_control(self, wait_time=1.0):
        cmd = f"docker exec --user {self.user} -d {self.name} /apollo/bazel-bin/modules/sim_control/sim_control_main sim_test &"
        _ = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(cmd)
        time.sleep(wait_time)

    ##### Apollo Recorder Operation (Only support terminal as some unstable reasons) #####
    def start_recorder(self, record_folder: str, record_id: str):
        """
        Starts cyber_recorder
        """
        cmd = f"docker exec --user {self.user} {self.name} rm -rf {record_folder}/{record_id}"
        _ = subprocess.run(cmd, shell=True)
        cmd = f"docker exec --user {self.user} {self.name} mkdir -p {record_folder}/{record_id}"
        _ = subprocess.run(cmd, shell=True)

        container_cmd_recorder = "/apollo/bazel-bin/cyber/tools/cyber_recorder/cyber_recorder"
        container_cmd_cmd = f"{container_cmd_recorder} record -o {record_folder}/{record_id}/recording -a &"
        cmd = f"docker exec --user {self.user} -d {self.name} {container_cmd_cmd}"
        logger.info(cmd)
        _ = subprocess.run(cmd, shell=True)
        time.sleep(1.0)
        logger.info(f"Started Apollo recorder.")

    def stop_recorder(self):
        """
        Stops cyber_recorder
        """
        container_cmd = "python3 /apollo/scripts/record_bag.py --stop --stop_signal SIGINT"
        cmd = f"docker exec --user {self.user} {self.name} {container_cmd}"
        logger.info(cmd)
        _ = subprocess.run(cmd, shell=True)
        time.sleep(1.0)
        logger.info(f"Stopped Apollo recorder.")

    def start_apollo(self):
        self.start_modules(1.0)
        logger.info('Start all of apollo.')
        time.sleep(0.5)

    def copy_record(self, record_folder: str, record_id: str, target_folder: str, delete=False):
        if not delete:
            cmd = f'docker cp {self.name}:{record_folder}/{record_id} {target_folder}'
            _ = subprocess.run(cmd, shell=True)
        else:
            cmd = f'docker cp {self.name}:{record_folder}/{record_id} {target_folder}'
            _ = subprocess.run(cmd, shell=True)
            cmd = f'docker exec --user {self.user} {self.name} rm -rf {record_folder}/{record_id}'
            _ = subprocess.run(cmd, shell=True)

        logger.info(cmd)
