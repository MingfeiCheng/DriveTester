import os
import subprocess

PROJECT_ROOT="/data/c/mingfeicheng/ApolloSim/v7.0"
APOLLO_ROOT="/data/c/mingfeicheng/ApolloSim/Apollo7.0"

ctn_name = 'apollo_test'
start_script_path = os.path.join(PROJECT_ROOT, "scripts/dev_start_ctn.sh")
options = "-y -l -f"
docker_script_dir = os.path.join(APOLLO_ROOT, "docker", "scripts")
cmd = f"bash {start_script_path} {options}"
subprocess.run(
    cmd,
    env={
        "CURR_DIR": docker_script_dir,
        "DEV_CONTAINER": ctn_name,
        "USER": os.environ.get("USER"),
        "APOLLO_ROOT_DIR": APOLLO_ROOT
    },
    shell=True
)