<div align="center">
  <!-- <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center">DriveTester</h1>
  <p align="center">
    <b>A Unified Platform for Simulation-Based Autonomous Driving Testing</b>
    <!-- <br /> -->
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br /> -->
  </p>
</div>

## About
This project aims to provide a convenient and unified testing platform for Autonomous Driving Systems (ADSs). 
It integrates various ADSs, such as [Baidu Apollo](https://github.com/ApolloAuto/apollo), along with different ADS testing tools like [AVFuzzer](https://github.com/MingfeiCheng/AV-Fuzzer-Reimplement), [BehAVExplor](https://github.com/MingfeiCheng/BehAVExplor).
This project is research-driven and will continue to be maintained for further development. We welcome any contributions!

## TODO List
- [ ] Release full ADS testing techniques
- [ ] Release Carla version (i.e., support different ADSs)
- [ ] Extend different Apollo version
- [ ] Release detailed documentation

[//]: # (## Sections)

[//]: # (We offer multiple testing platforms. Please refer to the detailed section for usage instructions.)

[//]: # (1. [ApolloSim]&#40;&#41;)

[//]: # (2. [Carla]&#40;&#41;)

## Setup
### Install Apollo
You can also config this project and Apollo step by step.
#### Step 1: Setup Apollo
##### 1. Download Apollo
```aiignore
git clone -b v7.0.0 https://github.com/ApolloAuto/apollo.git Apollo
```
Example: The apollo is saved at `/workspace/DriveTester/Apollo`.

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
git clone -b v7.0.0 https://github.com/MingfeiCheng/ApolloSim.git
```

##### 2. Config Python Environment  
The ApolloSim is fully developed by Python. You can install ApolloSim by running the following command:
```aiignore
cd ApolloSim
pip install -e .
```

##### 3. Quick Check ApolloSim Installation
```
python -c "import apollo_sim"
```

#### Step 3: Setup DriveTester & Testing
##### 1. Download DriveTester
```aiignore
git clone -b v7.0.0 https://github.com/MingfeiCheng/DriveTester.git
cd DriveTester
pip install -r requirements.txt
```
##### 2. Download Map resources
Please download the map resources from the following link: [ApolloSim Map Resources](https://drive.google.com/file/d/1cla1GERt4urK3MpWAwAoXAyBU8VVGcL4/view?usp=drive_link)
Then, unzip the resources to the `drive_tester/scenario_runner/ApolloSim/data` folder.

##### 3. Quick Check DriveTester Installation
You can quickly check the setup by running the following command:
```aiignore
python main.py
```
This will start the DriveTester with ApolloSim and Apollo under RANDOM testing.
Please see `drive_tester/configs` for more details. 

Because recently I am busy with my work, I will release the full version of DriveTester soon. Please stay tuned. Thank you!

## Cite & Contact
```aiignore
@article{cheng2024drivetester,
  title={DriveTester: A Unified Platform for Simulation-Based Autonomous Driving Testing},
  author={Cheng, Mingfei and Zhou, Yuan and Xie, Xiaofei},
  journal={arXiv preprint arXiv:2412.12656},
  year={2024}
}
@article{cheng2024evaluating,
  title={Decictor: Towards Evaluating the Robustness of Decision-Making in Autonomous Driving Systems},
  author={Cheng, Mingfei and Zhou, Yuan and Xie, Xiaofei and Wang, Junjie and Meng, Guozhu and Yang, Kairui},
  journal={arXiv preprint arXiv:2402.18393},
  year={2024}
}
```
Contact: [Mingfei Cheng](snowbirds.mf@gamil.com)

## Contribution
We warmly welcome contributions and suggestions to enhance this project and promote research in industrial-grade ADS testing. Your contributions are highly valued and greatly appreciated.

If you encounter any issues, please don't hesitate to report them by opening an issue. Before submitting a pull request, kindly open an issue for discussion to ensure alignment. Thank you for your support!
## License
Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Acknowledgements
We thanks for the following open-source projects:  
* [DoppelTest](https://github.com/Software-Aurora-Lab/DoppelTest)   
* [scenoRITA-7.0
](https://github.com/Software-Aurora-Lab/scenoRITA-7.0)  
* [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
