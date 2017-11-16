/*
 * PedPomdpNode.cpp
 *
 */

#include <fenv.h>
#include "PedPomdpNode.h"
#include <time.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Path.h>
//#include <pomdp_path_planner/GetPomdpPath.h>
//#include <pomdp_path_planner/PomdpPath.h>
//#include <ped_navfn/MakeNavPlan.h>

PedPomdpNode::PedPomdpNode()
{
    ROS_INFO("Starting Pedestrian Avoidance ... ");

    

	cerr << "DEBUG: Setting up subscription..." << endl;
    ros::NodeHandle nh;
    /// Setting up subsciption
    speedSub_ = nh.subscribe("odom", 1, &PedPomdpNode::speedCallback, this);
    pedSub_ = nh.subscribe("ped_local_frame_vector", 1, &PedPomdpNode::pedPoseCallback, this); 
	//pathSub_=nh.subscribe("global_plan", 1, &PedPomdpNode::pathCallback,this);
	pc_pub=nh.advertise<sensor_msgs::PointCloud>("confident_objects_momdp",1);
	path_pub=nh.advertise<nav_msgs::Path>("momdp_path",10);
    ros::NodeHandle n("~");

	pathPublished=false;
    bool simulation;
    n.param("simulation", simulation, false);
    ModelParams::init_params(simulation);
	cout << "simulation = " << simulation << endl;
	cout << "rosns = " << ModelParams::rosns << endl;

	bool fixed_path;
	double pruning_constant, pathplan_ahead;
    n.param("pruning_constant", pruning_constant, 0.0);
	n.param("fixed_path", fixed_path, false);
	n.param("pathplan_ahead", pathplan_ahead, 4.0);

    n.param("crash_penalty", ModelParams::CRASH_PENALTY, -1000.0);
    n.param("reward_base_crash_vel", ModelParams::REWARD_BASE_CRASH_VEL, 0.8);
    n.param("reward_factor_vel", ModelParams::REWARD_FACTOR_VEL, 1.0);
    n.param("belief_smoothing", ModelParams::BELIEF_SMOOTHING, 0.05);
    n.param("noise_robvel", ModelParams::NOISE_ROBVEL, 0.2);
    n.param("collision_distance", ModelParams::COLLISION_DISTANCE, 1.0);
    n.param("infront_angle_deg", ModelParams::IN_FRONT_ANGLE_DEG, 90.0);
    n.param("driving_place", ModelParams::DRIVING_PLACE, 0);


    double noise_goal_angle_deg;
    n.param("noise_goal_angle_deg", noise_goal_angle_deg, 45.0);
    ModelParams::NOISE_GOAL_ANGLE = noise_goal_angle_deg / 180.0 * M_PI;

    n.param("max_vel", ModelParams::VEL_MAX, 2.0);

    move_base_speed_=nh.subscribe("momdp_speed_dummy",1, &PedPomdpNode::moveSpeedCallback, this);
    //goalPub_ = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal",1);

	cerr << "DEBUG: Creating ped_momdp instance" << endl;
	controller = new Controller(nh, fixed_path, pruning_constant, pathplan_ahead);

    // default goal: after create door
    n.param("goalx", controller->goalx_, 19.5);
    n.param("goaly", controller->goaly_, 55.5);

	controller->window_pub=nh.advertise<geometry_msgs::PolygonStamped>("/my_window",1000);
	controller->pa_pub=nh.advertise<geometry_msgs::PoseArray>("my_poses",1000);
	controller->car_pub=nh.advertise<geometry_msgs::PoseStamped>("car_pose",1000);

	//controller->path_client=nh.serviceClient<pomdp_path_planner::GetPomdpPath>("get_pomdp_paths");
	controller->path_client=nh.serviceClient<nav_msgs::GetPlan>(ModelParams::rosns + "/ped_path_planner/planner/make_plan");



    ros::spin();
}
extern double marker_colors[20][3];
void PedPomdpNode::publishPath()
{
	cerr << "DEBUG: Call publishPath() " << endl;
	nav_msgs::Path msg;

	msg.header.frame_id=ModelParams::rosns+"/map";
	msg.header.stamp=ros::Time::now();
	vector<COORD> path=controller->worldModel.path;
	msg.poses.resize(path.size());
	for(int i=0;i<path.size();i++)
	{
		msg.poses[i].pose.position.x=controller->worldModel.path[i].x;
		msg.poses[i].pose.position.y=controller->worldModel.path[i].y;
		msg.poses[i].pose.position.z=0;
		msg.poses[i].header.frame_id=msg.header.frame_id;
		msg.poses[i].header.stamp=ros::Time::now();
	}
	path_pub.publish(msg);
//	cout<<"path with length "<<length<<" published"<<endl;

	cerr << "DEBUG: Done publishPath() " << endl;
}
PedPomdpNode::~PedPomdpNode()
{
    //controller->~ped_momdp();
}

void PedPomdpNode::speedCallback(nav_msgs::Odometry odo)
{
	//cout<<"update real speed "<<odo.twist.twist.linear.x<<endl;
	controller->real_speed_=odo.twist.twist.linear.x;
}

void PedPomdpNode::moveSpeedCallback(geometry_msgs::Twist speed)
{
    controller->updateSteerAnglePublishSpeed(speed);
}

bool sortFn(Pedestrian p1,Pedestrian p2)
{
	return p1.id<p2.id;
}

void PedPomdpNode::pedPoseCallback(ped_is_despot::ped_local_frame_vector lPedLocal)
{
   // cout<<"pedestrians received size = "<<lPedLocal.ped_local.size()<<endl;
	if(lPedLocal.ped_local.size()==0) return;

	sensor_msgs::PointCloud pc;

	pc.header.frame_id=ModelParams::rosns+"/map";
	pc.header.stamp=lPedLocal.ped_local[0].header.stamp;
	//RealWorld.ped_list.clear();
	vector<Pedestrian> ped_list;
    for(int ii=0; ii< lPedLocal.ped_local.size(); ii++)
    {
		geometry_msgs::Point32 p;
		Pedestrian world_ped;
		ped_is_despot::ped_local_frame ped=lPedLocal.ped_local[ii];
		world_ped.id=ped.ped_id;
		world_ped.w = ped.ped_pose.x;
		world_ped.h = ped.ped_pose.y;
		p.x=ped.ped_pose.x;
		p.y=ped.ped_pose.y;
		p.z=1.0;
		pc.points.push_back(p);

		//cout<<"ped pose "<<ped.ped_pose.x<<" "<<ped.ped_pose.y<<" "<<world_ped.id<<endl;
		ped_list.push_back(world_ped);
    }
	//std::sort(ped_list.begin(),ped_list.end(),sortFn);
	for(int i=0;i<ped_list.size();i++)
	{
		controller->worldStateTracker.updatePed(ped_list[i]);
		//cout<<ped_list[i].id<<" ";
	}
	//cout<<endl;
	pc_pub.publish(pc);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "mdp");

    // raise error for floating point exceptions rather than silently return nan
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    srand(unsigned(time(0)));
    PedPomdpNode mdp_node;
}
