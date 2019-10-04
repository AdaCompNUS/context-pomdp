#
# LeTS-Drive with SUMMIT simulator integration

Watch the driving video: [![Watch the driving video](https://img.youtube.com/vi/bQjcd-NBdIg/0.jpg)](https://www.youtube.com/watch?v=bQjcd-NBdIg)

## Overview
This repository contains all algorithmic elements for reproducing LeTS-Drive [(paper)](https://arxiv.org/abs/1905.12197) in heterogenous traffic simulated by the SUMMIT simulator [(paper)](https://www.dropbox.com/s/fs0e9j4o0r80e82/SUMMIT.pdf?dl=0).
* Even though this repository implements the full LeTS-Drive pipeline, you can easily down-grade it to perform stand-alone imitation learning or POMDP planning.
* The current repository is in progress of migrating from the Unity pedestrian simulator to SUMMIT. Some modules like the neural networks learner package is still for the old Unity simulator. For now, one can use the [existing pedestrain datasets](https://www.dropbox.com/s/bl093sneceu40fe/ped_datasets.zip?dl=0) to test imitation learning.

Here is the list of ROS packages marked with the simulator they support:
* (SUMMIT) __summit_connector__: A python package for communicating with SUMMIT, constructing the scene, controlling the traffic, and processing state and context information. summit_connector publishes the following ROS topics to driving algorithms: 
    * /odom: Odometry of the exo-vehicle;
    * /ego_state: State of the exo_vehicle;
    * /plan: Reference path of the ego-vehicle;
    * /agent_array: State and intended paths of exo-agents.
* (SUMMIT) __car_hyp_despot__: The core POMDP planner performing guided tree search using HyP-DESPOT and policy/value networks. It contains the following source folders:
    * HypDespot: The HyP-DESPOT POMDP solver.
    * planner: The POMDP model for the driving problem.
    * Porca: The PORCA motion model for predicting agent's motion.
* (SUMMIT) __crowd_pomdp_planner__: A wrapping over the POMDP planner. It receives information from the simulator and run belief tracking and POMDP planning. Key files in the package include:
    * PedPomdpNode.cpp: A ROS node host which also receives ROS parameters.
    * controller.cpp: A class that launches the planning loop.
    * world_simulator.cpp: A class that maintains world state.
    * VelPublisher.cpp: A velocity controller that converts accelaration output by POMDP planning to command velocities. It publishes the following ROS topics:
        * \cmd_vel: command velocity for the ego-vehicle converted from accelaration and the current velocity.
        * \cmd_accel: command accelaration given by the POMDP planner if one wish to directly apply it.
* (Unity) __il_controller__: The neural network learner for imitation learning. The key files include:
    * data_processing.sh: Script for executing the data_processing pipeline given a folder path of recorded rosbags;
    * policy_value_network.py: The neural network architectures for the policy and the value networks.
    * train.py: Script for training the neural networks using the processed dataset in hdf5 (.h5) format;
    * test.py: Script for using the learned neural network to directly drive a car in the simulator.

The repository also contains two utility folders:
* __setup__: Bash scripts for setting up the environment.
* __scripts__: Scripts for performing driving in simulators. Main files include:
   * run_data_collection.py: launch the simulator and driving algorithms for data collection or evaluation purposes. One can use the following modes to be set via the `--baseline` argument:
      * imitation (Unity): drive a car directly using a trained policy network.
      * pomdp (?): drive a car using Decoupled-Action POMDP and Hybrid A*.
      * joint_pomdp (SUMMIT): drive a car using Joint-Action POMDP.
      * lets-drive (?): drive a car using guided belief tree search.
      
      This script will automatically compile the entire project.
   * experiment_summit.sh: Script for executing repetitive experiments by calling run_data_collection.py.

## 1. Setup
### 1.1 Pre-requisites
1. Install [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Note: you need to follow the [official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for a successful installation.)
2. Install [CUDNN 7](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

### 1.2 Setup Dependencies
Download all bash scripts in the [setup](./setup) folder (note: not the full repo). Then, go through the following steps to set up the dependencies:
#### 1.2.1 Install ros-melodic
Run
```
bash install_ros_melodic.sh
```
The script might prompt for sudo privilege.
Then add the following line to the end of `~/.bashrc`:
```
source /opt/ros/melodic/setup.bash
```
Logout and login again into your account. Then type `roscd` to validate the installation. The command should navigate you to `/opt/ros/melodic/`.
#### 1.2.2 Build and install the lastest libtorch (the CPP frontend of Pytorch)
Run
```
bash install_torch.sh
```
The script might prompt for sudo privilege.

Run 
```ls -alh ~/libtorch```
to validate the installation. You are expected to see the following directories:
```include  lib  share```

An additional step here is to validate the CMake installation:
```
cmake --version
```
The output version should be `3.14.1`. If not, logout and login again to your account. 

If the output version is still not correct. Redo the CMake installation lines in `install_torch.sh`, then re-login to your account.
#### 1.2.3 Build and install OpenCV 4.1.0
Run
```
bash install_opencv4.sh 
```
The script might prompt for sudo privilege.
#### 1.2.4 Prepare the catkin workspace
Run
```
cd && mkdir -p catkin_ws/src
cd catkin_ws
catkin config --merge-devel
catkin build
```
#### 1.2.5 Fetch the full repository
Run
```
cd ~/catkin_ws/src
git clone https://github.com/cindycia/LeTS-Drive-SUMMIT.git    
mv LeTS-Drive-SUMMIT/* .
mv LeTS-Drive-SUMMIT/.git .
```
Now all ROS packages should be in `~/catkin_ws/src`.

Compile packages in the workspace:
```
cd ~/catkin_ws
catkin config --merge-devel
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Then, add the following line to the end of `~/.bashrc`:
```
source ~/catkin_ws/devel/setup.bash
```
#### 1.2.6 Install dependent python packages
(Optional) Set up a python virtualenv for an isolated setup of python packages which do not interfere with global python packages.
```
cd
virtualenv --system-site-packages lets_drive 
source ~/lets_drive/bin/activate
```
You can append the source ~/lets_drive... line to ~/.bashrc. Also, to deactivate the current virtual environment, just type.
```
deactivate
```
(Compulsory) To install python dependencies, run
```
roscd il_controller/src && pip install -r requirements.txt
```
### 1.3 Setup the SUMMIT simulator
Download the [SUMMIT simultator release package](https://www.dropbox.com/s/3cnjktij8vtfn56/summit.zip?dl=0), and unzip it to `~/summit`. 
Or you can download the [source code](https://github.com/AdaCompNUS/carla.git) from github and compile from source.
For now the code explicitly uses this `~/summit` to find the simulator. So stick to the path for SUMMIT installation.

## 2. Run the System
### 2.1 Launch the SUMMIT Simulator
```
export SDL_VIDEODRIVER=offscreen
LinuxNoEditor/CarlaUE4.sh -carla-rpc-port=2000 -carla-streaming-port=2001
```
You can change 2000 and 2001 to any unused port you like.

Note: this is only required when launching the simulator and the planning in different containers like docker. For this, you also need to block the cooresponding launching code in experiment_summit.sh.
### 2.2 Launch the Planner
```
cd ~/catkin_ws/src/scripts
./experiment_summmit.sh [gpu_id] [start_round] [end_round(inclusive)] [carla_portal, e.g. 2000 as set before]
```
Step 2.1 is not required if experiment_summit.sh launches the summit simulator. 

experiment_summit.sh does not record bags by default. To enable rosbag recording, change the following variable to 1 in the script:
```
record_bags=0
```
## 3. Process ROS Bags to HDF5 Datasets
Convert bags into h5 files using multiple threads:
```
roscd il_controller && cd src
python3 Data_processing/parallel_parse_pool.py --bagspath [rosbag/path/] --peds_goal_path Maps/
```
or using a single thread:
```
python3 Data_processing/bag_to_hdf5.py --bagspath [rosbag/path/] --peds_goal_path Maps/
```
combine bag_h5 files into training, validation, and test sets:
```
python3 Data_processing/combine.py --bagspath [rosbag/path/]
```
## 4. IL Training
Start training and open tensorboard port
```
python3 train.py --batch_size 128 --lr 0.0001 --train train.h5 --val val.h5
tensorboard --logdir runs --port=6001
```
[Optional] If you are running a learning section on the server, to check the learning curve, run the following line on your local machine to bind the log port:
```
ssh -4 -N -f -L localhost:6001:localhost:6001 [remote address, e.g. panpan@unicorn4.d2.comp.nus.edu.sg]
```
Then, on your local browser, visit http://localhost:6001/

**Main Arguments**:
- `train`: The path to the train dataset.
- `val`: The path to the validation dataset.
- `lr`: Learning rate.
- `batch_size`: Batch size. 
- `epochs`: Number of epochs to train. Default: 50.
- `modelfile`: Name of the model file to be saved. Time flag will be appended for the actual save name. To specify the exact name, set `exactname` to be True.
- `resume`: The model file you want to load and retrain.
- `k`: Number of Value Iterations in GPPN. Default: 5.
- `f`: GPPN Conv kernel size. Default: 7
- `l_h`: Number of hidden layers in GPPN. Default: 2, as used in LeTS-Drive.
- `nres`: Number of resnet layers.
- `w`: Width of the resnet (image size).
- `do_p`: Probability of dropping out features output by the resnet.
- `ssm`: Sigma smoothing factor for Guassian Mixture heads. If set, the learned variance will be at least the ssm value. Default: 0.03.
- `v_scale`: Scaling factor of the value loss when added to the composite loss.
