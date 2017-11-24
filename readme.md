### run the whole system
```
roscore
rosrun tf static_transform_publisher -205 -141 0.0 0.0 0.0 0.0 /map /odom 10 # -207.26 -143.595
rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 /base_link /laser_frame 10
cd ~ && rosrun map_server map_server airport_departure.yaml
cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python OdometryROS.py
# cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/ && python LidarROS.py
# run unity
roslaunch ped_pathplan pathplan.launch
rosrun peds_unity_system peds_simulator_no_car ## seems that peds_simulator_no_car performs better than peds_simulator
roscd peds_unity_system/src && python unity_connector.py
roslaunch ped_is_despot is_despot.launch
cd ~/workspace/catkin_ws/src/purepursuit_combined/ && python purepursuit_NavPath_unity.py
rviz
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


