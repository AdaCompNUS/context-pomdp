#!/bin/bash
### 219
roscore &
sleep 3
echo "roscore restarted"

subfolder=''
obstacle_file=''
goal_file=''
executable='cross.x86_64'
root_path='/home/yuanfu/Unity/DESPOT-Unity'
problem_flag=''
scenario=0
doshift=false

problem_flag="16_8_16"
scenario=3
road=8.0

total_batch=11

goal_x=0.0
goal_y=0.0
start_x=-18.0
start_y=0.0
rand_shift=$(python -c "import random;print('{0:0.1f}'.format(random.uniform(-0.5, 0.5)))")
echo 'rand_shift: '$rand_shift

num_cases=3

i=0

#for j in $(seq 3 $num_cases)
for j in $(seq 3 3)
do
	goal_flag=''
	goal_value=19.9
	start_value=18.0
	goal_value_flag="19d9"

	rot_angle=0.0
	start_x=-$start_value
	#start_x=$start_value
	start_y=$rand_shift
	if (($j == 1)); then
		goal_x=$goal_value
		goal_y=0.0
		rot_angle=0.0
		goal_flag=$goal_value_flag'_0'
	elif (($j == 2)); then
		goal_x=0.0
		goal_y=-$goal_value
		rot_angle=0.0
		goal_flag='0_n'$goal_value_flag
	elif (($j == 3)); then
		goal_x=0.0
		goal_y=$goal_value
		goal_flag='0_'$goal_value_flag
		rot_angle=0.0
	elif (($j == 4)); then
		goal_x=0.0
		goal_y=-$goal_value
		goal_flag='0_n'$goal_value_flag
		rot_angle=3.1415926
	fi

	subfolder='map_'$problem_flag'_goal_'$goal_flag'_single'
	mkdir -p $root_path/result/$subfolder
	mkdir -p $root_path/result/$subfolder'_debug'
	obstacle_file="/home/yuanfu/Maps/indian_cross_obs_"$problem_flag".txt"
	goal_file="/home/yuanfu/Maps/indian_cross_goals_"$problem_flag".txt" 
	executable="cross_bts_single"

	echo "Running problem indian_cross_"$problem_flag
	echo "executable: "$executable
	echo "Obstacle file: "$obstacle_file
	echo "Pedestrian goal file: "$goal_file
	echo "Car start: "$start_x","$start_y
	echo "Car goal: "$goal_x","$goal_y
	echo "Scenaro, doshift, and roadedge: "$scenario","$doshift","$road
	# to make all launches wait for roscore
	#timeout 1 roslaunch --wait ped_is_despot is_despot.launch goal_x:=$goal_x goal_y:=$goal_y obstacle_file_name:=$obstacle_file goal_file_name:=$goal_file &> result/dummy
	rosrun tf static_transform_publisher $start_x $start_y 0.0 $rot_angle 0.0 0.0 /map /odom 10 &> $root_path/result/$subfolder'_debug'/trans_map_log_single'_'$i'_'$j.txt &
	rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 /base_link /laser_frame 10 &> $root_path/result/$subfolder'_debug'/trans_laser_log_single'_'$i'_'$j.txt &
	echo "Tranforms setup"
	sleep 1
	cd ~/Maps
	rosrun map_server map_server indian_cross_$problem_flag.yaml &> $root_path/result/$subfolder'_debug'/map_server_log_single'_'$i'_'$j.txt &
	echo "Map setup"
	sleep 1
	cd ~/Unity/DESPOT-Unity/Assets/Sensors/Scripts/ROS/  
	python OdometryROS.py &> $root_path/result/$subfolder'_debug'/odom_log_single'_'$i'_'$j.txt & export Odom_ID=$! 
	cd ~/workspace/catkin_ws/src/ped_is_despot/src
	python unity_connector.py &> $root_path/result/$subfolder'_debug'/connector_log_single'_'$i'_'$j.txt & export Connector_ID=$!
	cd ~/workspace/catkin_ws/src/purepursuit_combined/
	python purepursuit_NavPath_unity.py &> $root_path/result/$subfolder'_debug'/pursuit_log_single'_'$i'_'$j.txt & export Pursuit_ID=$!
	echo "Python scripts setup"
	sleep 1
	roslaunch ped_is_despot ped_sim.launch obstacle_file_name:=$obstacle_file goal_file_name:=$goal_file &> $root_path/result/$subfolder'_debug'/ped_sim_log_single'_'$i'_'$j &
	roslaunch porca_planner ped_sim.launch obstacle_file_name:=$obstacle_file goal_file_name:=$goal_file &> $root_path/result/$subfolder'_debug'/ped_sim_log_single'_'$i'_'$j &
	rosrun ped_is_despot vel_publisher &

	$root_path/$executable -scenario $scenario -startx $start_x -starty $start_y -doshift $doshift -road $road &> $root_path/result/$subfolder'_debug'/unity_log_single'_'$i'_'$j.txt &
	echo "Simulator setup"
	# sleep 1
	# roslaunch --wait ped_pathplan pathplan.launch &> $root_path/result/$subfolder'_debug'/path_planner_log_single'_'$i'_'$j &
	
	
	# timeout 120 roslaunch --wait ped_is_despot is_despot.launch goal_x:=$goal_x goal_y:=$goal_y obstacle_file_name:=$obstacle_file goal_file_name:=$goal_file &> $root_path/result/$subfolder/$problem_flag'_goal_'$goal_flag'_sample-'$run'-'$i'-'$j.txt	
	# sleep 7
	# echo "Finish data: sample_"$run'_'$i'_'$j

	#pkill -2 record

	sleep 120

	executable_process="cross_bts_singl"
	sleep 2
	pkill map_server
	pkill $executable_process
	pkill path_planner
	pkill peds_simulator_
	pkill static_transfor
	pkill ped_pomdp
	pkill local_frame
	pkill vel_publisher
	pkill roslaunch
	kill -9 $Odom_ID
	kill -9 $Connector_ID
	kill -9 $Pursuit_ID
	sleep 3
	pkill -9 $executable_process
	yes | rosclean purge
done


pkill rosmaster
pkill roscore
pkill rosout


