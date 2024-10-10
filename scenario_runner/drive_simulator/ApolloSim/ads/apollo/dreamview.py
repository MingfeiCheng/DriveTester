import sys
import json
import time

from websocket import create_connection
from loguru import logger

class Dreamview:
    """
    Class to wrap Dreamview connection

    :param str ip: IP address of Dreamview websocket
    :param int port: port of Dreamview websocket
    """

    def __init__(self, ip: str, port: int) -> None:
        self.url = f"ws://{ip}:{port}/websocket"
        self.ws = create_connection(self.url)
        logger.info(f'Dreamview connects to {self.url}')

    def reconnect(self):
        """
        Closes the websocket connection and re-creates it so that data can be received again
        """
        self.ws.close()
        self.ws = create_connection(self.url)
        # logger.info(f'Dreamview reconnects to {self.url}')
        return

    def send_data(self, data: dict):
        """
        Helper function to send data to Dreamview

        :param dict data: data to be sent
        """
        self.ws.send(json.dumps(data))

    ##### Sim Control Module #####
    def start_sim_control(self):
        """
        Starts SimControl via websocket
        """
        logger.info('Start Sim Control from Dreamview.')
        self.send_data({
            "type": "StartSimControl"
        })

    def stop_sim_control(self):
        """
        Stops SimControl via websocket
        """
        logger.info('Stop Sim Control from Dreamview.')
        self.send_data({
            "type": "StopSimControl"
        })

    ##### Setup HD Map #####
    def set_hd_map(self, hd_map):
        word_list = []
        for s in hd_map.split('_'):
            word_list.append(s[0].upper() + s[1:])

        mapped_map = ' '.join(word_list)

        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "CHANGE_MAP", "value": mapped_map})
        )

        if not self.get_current_map() == mapped_map:
            folder_name = hd_map.replace(" ", "_")
            error_message = (
                "HD Map {0} was not set. Verify the files exist in "
                "/apollo/modules/map/data/{1} and restart Dreamview -- Aborting..."
            )
            logger.error(
                error_message.format(
                    mapped_map, folder_name
                )
            )
            sys.exit(1)
        logger.info(f'Set HD_MAP {hd_map} from Dreamview')
        return

    def get_current_map(self):
        """
        Returns the current HD Map loaded in Dreamview
        """
        try:
            self.reconnect()
        except ConnectionRefusedError as e:
            logger.error("Not able to get the current HD map loaded in Dreamview.")
            logger.error("Original exception: " + str(e))
            return None

        data = json.loads(self.ws.recv())
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        return data["data"]["currentMap"]

    ##### Setup HD Vehicle Type #####
    def set_vehicle(self, vehicle):
        # Lincoln2017MKZ from LGSVL has a GPS offset of 1.348m behind the center of the vehicle, lgsvl.Vector(0.0, 0.0, -1.348) (x, z, y)
        # But need to confirm the offset of CARLA
        """
        Folders in /apollo/modules/calibration/data/ are the available vehicle calibrations
        Vehicle options in Dreamview are the folder names with the following changes:
            - underscores (_) are replaced with spaces
            - the first letter of each word is capitalized

        vehicle parameter is the modified folder name.
        vehicle should match one of the options in the middle drop down in the top-right corner of Dreamview.
        """

        word_list = []
        for s in vehicle.split('_'):
            word_list.append(s[0].upper() + s[1:])

        mapped_vehicle = ' '.join(word_list)

        self.ws.send(
            json.dumps(
                {"type": "HMIAction", "action": "CHANGE_VEHICLE", "value": mapped_vehicle}
            )
        )

        if not self.get_current_vehicle() == mapped_vehicle:
            folder_name = vehicle.replace(" ", "_")
            error_message = (
                "Vehicle calibration {0} was not set. Verify the files exist in "
                "/apollo/modules/calibration/data/{1} and restart Dreamview -- Aborting..."
            )
            logger.error(
                error_message.format(
                    mapped_vehicle, folder_name
                )
            )
            sys.exit(1)
        logger.info(f'Set Vehicle {vehicle} from Dreamview')
        return

    def get_current_vehicle(self):
        """
        Returns the current Vehicle configuration loaded in Dreamview
        """
        try:
            self.reconnect()
        except ConnectionRefusedError as e:
            logger.error("Not able to get the current vehicle configuration loaded in Dreamview.")
            logger.error("Original exception: " + str(e))
            return None

        data = json.loads(self.ws.recv())
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        return data["data"]["currentVehicle"]

    ##### Setup Apollo Mode #####
    def set_setup_mode(self, mode):
        """
        mode is the name of the Apollo 5.0 mode as seen in the left-most drop down in the top-right corner of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "CHANGE_MODE", "value": mode})
        )
        logger.info(f'Setup Apollo mode {mode} from Sim Control.')
        return

    ##### Enable Required Modules #####
    def  enable_module(self, module, wait_time=3.0):
        tries = 0
        while not self.check_module_status(module):
            tries += 1
            if tries > 60:
                raise RuntimeError(f'Apollo module {module} can not be started.')
            self.ws.send(
                json.dumps({"type": "HMIAction", "action": "START_MODULE", "value": module})
            )
            time.sleep(wait_time)
            wait_time += 1.0
        logger.info(f'{module} is running.')
        return

    def disable_module(self, module, wait_time=3.0):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        tries = 0
        while self.check_module_status(module):
            tries += 1
            if tries > 60:
                raise RuntimeError(f'Apollo module {module} can not be closed.')
            self.ws.send(
                json.dumps({"type": "HMIAction", "action": "STOP_MODULE", "value": module})
            )
            time.sleep(wait_time)
            wait_time += 1
        logger.info(f'Apollo module {module} is closed.')
        return

    def _get_module_status(self):
        """
        Returns a dict where the key is the name of the module and value is a bool based on the module's current status
        """
        self.reconnect()
        data = json.loads(
            self.ws.recv()
        )  # This first recv() call returns the SimControlStatus in the form '{"enabled":false,"type":"SimControlStatus"}'
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        return data["data"]["modules"]

    def get_sim_control_status(self):
        """
        Returns a dict where the key is the name of the module and value is a bool based on the module's current status
        """
        self.reconnect()
        data = json.loads(
            self.ws.recv()
        )  # This first recv() call returns the SimControlStatus in the form '{"enabled":false,"type":"SimControlStatus"}'
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())

        return data["data"]["dynamicModels"]

    def check_module_status(self, module):
        """
        Checks if all modules in a provided list are enabled
        """
        module_status = self._get_module_status()
        for _module, _status in module_status.items():
            if not _status and module == _module:
                return False
        return True