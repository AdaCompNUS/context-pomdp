#pragma once
#include <interface/world.h>
#include <string>
#include <ros/ros.h>
#include "ped_pomdp.h"
#include "param.h"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>
//#include <sensing_on_road/pedestrian_laser_batch.h>
//#include <dataAssoc_experimental/PedDataAssoc_vector.h>
//#include <dataAssoc_experimental/PedDataAssoc.h>

//#include <ped_is_despot/peds_believes.h>


#include <rosgraph_msgs/Clock.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
//#include "executer.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
//#include "pedestrian_changelane.h"
//#include "mcts.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "param.h"
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
//#include <pomdp_path_planner/GetPomdpPath.h>
//#include <pomdp_path_planner/PomdpPath.h>
#include <nav_msgs/GetPlan.h>



#include <ped_is_despot/ped_local_frame.h>
#include <ped_is_despot/ped_local_frame_vector.h>
#include <ped_is_despot/imitation_data.h>
#include <ped_is_despot/car_info.h>
#include <ped_is_despot/peds_info.h>

#include "std_msgs/Float32.h"



using namespace despot;

class SimulatorBase  {

public:

	SimulatorBase(ros::NodeHandle& _nh, std::string obstacle_file_name):
		nh(_nh), obstacle_file_name_(obstacle_file_name){
		;
	}

	double real_speed_;
	double target_speed_;

	double time_scale_;

    double steering_;


   	static WorldStateTracker* stateTracker;
   	WorldBeliefTracker* beliefTracker;

    static WorldModel worldModel;

	std::string global_frame_id;
    std::string obstacle_file_name_;

public:
	// for imitation learning
	ros::NodeHandle& nh;

	ros::Subscriber carSub_;

	ros::Publisher IL_pub; 
	bool b_update_il;
	ped_is_despot::imitation_data p_IL_data; 


	virtual void publishImitationData(PomdpStateWorld& planning_state, ACT_TYPE safeAction, float reward, float vel) =0;


public:
	static bool ped_data_ready;
};



static double marker_colors[20][3] = {
    	{0.0,1.0,0.0},  //green
		{1.0,0.0,0.0},  //red
		{0.0,0.0,1.0},  //blue
		{1.0,1.0,0.0},  //sky blue
		{0.0,1.0,1.0},  //yellow
		{0.5,0.25,0.0}, //brown
		{1.0,0.0,1.0}   //pink
};

static int action_map[3]={2,0,1};
