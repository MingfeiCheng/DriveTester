# ApolloSim

## TODO:
- [ ] Complete the documents
- [ ] Upload testing engine

## Setup
### 1. Quick Setup
You can quickly setup this project by running the following commands:
```aiignore
git clone https://github.com/MingfeiCheng/DriveTester.git DriveTester
cd DriveTester
bash scripts/setup_DriveTester.sh
```

### 2. Step by Step
You can also config this project and Apollo step by step.
#### Step 1: Setup Apollo
##### 1. Download Apollo
```aiignore
git clone -b v7.0.0 https://github.com/ApolloAuto/apollo.git Apollo
```
The apollo is saved at `/workspace/DriveTester/Apollo`.

##### 2. Build Apollo

(1) Start docker container
```aiignore
cd /workspace/DriveTester/Apollo
bash docker/scripts/dev_start.sh
```

(2) Compile the Apollo
```aiignore
bash docker/scripts/dev_into.sh
./apollo.sh build
```
Note that there may has an issue during building Apollo v7.0.0, should be fixed by
```aiignore
vim WORKSPACE
# paste the following patch to line 60
http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://github.com/madler/zlib/archive/v1.2.11.tar.gz"],
)
```
Please refer to https://github.com/ApolloAuto/apollo/issues/14374 and https://github.com/ApolloAuto/apollo/pull/14387/files


#### Step 2: Setup ApolloSim

##### 1. Download ApolloSim
```aiignore
git clone -b v7.0 https://github.com/MingfeiCheng/ApolloSim.git v7.0
```
The ApolloSim is saved at `/workspace/DriveTester`

##### 2. Config Python Environment  
The ApolloSim is totally developed by Python. Please ensure you have installed Anaconda before the following steps.
```aiignore
conda env create -f environment.yml
```

##### 3. Convert the Apollo proto files
```aiignore
cd /workspace/DriveTester
bash scripts/proto_generate.sh
```

##### 4. Generate maps from Apollo Map
```aiignore
python tools/preprocess_ApolloSim_map.py
```

### Quick Check
You can quickly check the steup by running the following command:
```aiignore
python test/ApolloSim/test_traffic_generator.py
```

### Acknowledgement
We thanks for the following open-source projects:
[DoppelTest](https://github.com/Software-Aurora-Lab/DoppelTest)   
[scenoRITA-7.0
](https://github.com/Software-Aurora-Lab/scenoRITA-7.0)  
[ScenarioRunner](https://github.com/carla-simulator/scenario_runner)