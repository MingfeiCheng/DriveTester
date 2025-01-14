import sys

sys.path.append('/data/c/mingfeicheng/dev/DriveTester_private')

from apollo_sim.tools.map_processor import preprocess_apollo_map

if __name__ == '__main__':
    project_root = "/data/c/mingfeicheng/dev/DriveTester_private"

    preprocess_apollo_map('borregas_ave', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data')
    preprocess_apollo_map('san_mateo', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data')
    preprocess_apollo_map('sunnyvale_loop', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data')
    preprocess_apollo_map('sunnyvale_big_loop', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data')
    preprocess_apollo_map('sunnyvale_with_two_offices', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data', f'{project_root}/drive_tester/scenario_runner/ApolloSim/data')
