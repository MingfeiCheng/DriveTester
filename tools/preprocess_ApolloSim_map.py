import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scenario_runner.drive_simulator.ApolloSim.map.apollo_map_parser import MapParser

map_parser = MapParser()
map_name = 'borregas_ave'
map_parser.convert_map(map_name)