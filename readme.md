### change the vehicle model, and collision model
WorldModel.cpp
peds_simulator_no_car.cpp

collision.cpp 
GPU_Car_Drive.cu
ped_pathplan/launch/param.yaml

Controller.cpp changed the use straight line or not  fixed_path_ = true;

### change goal of the car
change goal_x, goal_y in /home/yuanfu/workspace/catkin_ws/src/ped_is_despot/is_despot_param.yaml
or
change the goal_x, goal_y parameters in run_despot_and_unity_and_record.sh

### change ped goals
1. In Unity project, file PedsSystem.cs:  change goals_ in InitPed function; 
	Rebuild executable after change to /home/yuanfu/Unity/DESPOT-Unity/;
	Modify the executable name in /home/yuanfu/Unity/DESPOT-Unity/run_despot_and_unity_and_record.sh
2. 
	In /home/yuanfu/workspace/catkin_ws/src/car_hyp_despot/src/planner/WorldModel.cpp, change goals in WorldModel().
	In /home/yuanfu/workspace/catkin_ws/src/peds_unity_system/src/peds_simulator_no_car.cpp, change goals in PedsSystem()
	or:
	update Maps/indian_cross_goals_*.txt and modify the goal_file in run_despot_and_unity_and_record.sh


### change initial position of the car
Translate the car in Unity and rebuild the exe. Update this line below: rosrun tf static_transform_publisher -16.0 0 0.0 0.0 0.0 0.0 /map /odom 10


### change initial position of new pedestrians
In Unity project, file PedsSystem.cs:  change the parameters of GenOneRandPed in InitPed() and Update() function; 

### change map
Generate a map and its .yaml file in ~
update line: cd ~ && rosrun map_server map_server <map_name>.yaml
update Maps/indian_cross_obs_*.txt, and modify the obstacle_file_name parameter in run_despot_and_unity_and_record.sh

### to change cost map option
In /home/yuanfu/workspace/catkin_ws/src/ped_pathplan/launch/params.yaml, change peds: topic:



### run the whole system
```
roscore

## for airport:
#rosrun tf static_transform_publisher -205 -142.5 0.0 0.0 0.0 0.0 /map /odom 10 # -207.26 -143.595

## for indian cross:
#rosrun tf static_transform_publisher 0 -8.0 0.0 1.57 0.0 0.0 /map /odom 10
rosrun tf static_transform_publisher -16.0 0 0.0 0.0 0.0 0.0 /map /odom 10

rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 /base_link /laser_frame 10

## for airport:
#cd ~ && rosrun map_server map_server airport_departure.yaml

## for indian cross: don't use this line during data collection
#cd ~ && rosrun map_server map_server indian_cross2.yaml

cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python OdometryROS.py
# cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python LidarROS.py
# run unity
#roslaunch ped_pathplan pathplan.launch
#rosrun peds_unity_system peds_simulator_no_car ## seems that peds_simulator_no_car performs better than peds_simulator
roscd peds_unity_system/src && python unity_connector.py
cd ~/workspace/catkin_ws/src/purepursuit_combined/ && python purepursuit_NavPath_unity.py
#rviz

## to run with unity open
roslaunch ped_is_despot is_despot.launch

## to run data collection
cd /home/yuanfu/Unity/DESPOT-Unity && ./run_despot_and_unity_and_record.sh
```


### test pedestrian system:
```
roscore
rosrun peds_unity_system peds_simulator_no_car
roscd peds_unity_system/src && python unity_connector.py
# run unity with use_ped_model set to true
```

### test path planning system:
```
roscore
#change the following x,y to robot start pos in unity (x,z)
rosrun tf static_transform_publisher -205 -143 0.0 0.0 0.0 0.0 /map /odom 1000 # -207.26 -143.595
cd ~ && rosrun map_server map_server airport_departure.yaml
cd Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python OdometryROS.py
# run unity
roslaunch ped_pathplan pathplan.launch
rviz
# optional (to make pedestrian system work at the same time): 
rosrun peds_unity_system peds_simulator_no_car
roscd peds_unity_system/src && python unity_connector.py
```

### test speed controller:
```
roscore
rosrun tf static_transform_publisher -205 -143 0.0 0.0 0.0 0.0 /map /odom 1000 # -207.26 -143.595
cd ~ && rosrun map_server map_server airport_departure.yaml
cd Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python OdometryROS.py
# run unity
roslaunch ped_pathplan pathplan.launch
rviz
cd ~/workspace/catkin_ws/src/purepursuit_combined/ && python purepursuit_NavPath_unity.py
```

### test path planning with ped_path_prediction
```
roscore
rosrun tf static_transform_publisher -205 -143 0.0 0.0 0.0 0.0 /map /odom 1000 # -207.26 -143.595
rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 /base_link /laser_frame 1000
cd ~ && rosrun map_server map_server airport_departure.yaml
cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python OdometryROS.py
# cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python LidarROS.py
# run unity
roslaunch ped_pathplan pathplan.launch
rviz
rosrun peds_unity_system peds_simulator_no_car
roscd peds_unity_system/src && python unity_connector.py
 rosrun ped_path_predictor ped_path_predictor_node 
```


