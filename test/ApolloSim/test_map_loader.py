import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.map_parser import MapParser
from common.data_provider import DataProvider

DataProvider.map_root = "/data/c/mingfeicheng/ApolloSim/v7.0/data/maps"

map_parser = MapParser()
map_name = "borregas_ave"
map_parser.load_from_pkl(map_name)

map_parser = MapParser()
map_name = "sunnyvale_loop"
map_parser.load_from_pkl(map_name)

map_parser = MapParser()
map_name = "sunnyvale_big_loop"
map_parser.load_from_pkl(map_name)