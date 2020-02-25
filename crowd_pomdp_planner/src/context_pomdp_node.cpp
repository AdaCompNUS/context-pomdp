/*
 * ContextPomdpNode.cpp
 *
 */

#include "context_pomdp_node.h"

ContextPomdpNode::ContextPomdpNode(int argc, char** argv) {
	ROS_INFO("Starting Crowd-POMDP ...");

	cerr << "DEBUG: Setting up subscription..." << endl;
	ros::NodeHandle nh;
	ros::NodeHandle n("~");

	pathPublished = false;
	bool simulation;
	n.param("simulation", simulation, false);
	ModelParams::InitParams(simulation);
	cout << "simulation = " << simulation << endl;
	cout << "rosns = " << ModelParams::ROS_NS << endl;

	bool fixed_path;
	n.param("fixed_path", fixed_path, false);
	n.param("crash_penalty", ModelParams::CRASH_PENALTY, -1000.0);
	n.param("reward_base_crash_vel", ModelParams::REWARD_BASE_CRASH_VEL, 0.8);
	n.param("reward_factor_vel", ModelParams::REWARD_FACTOR_VEL, 1.0);
	n.param("belief_smoothing", ModelParams::BELIEF_SMOOTHING, 0.05);
	n.param("noise_robvel", ModelParams::NOISE_ROBVEL, 0.2);
	n.param("collision_distance", ModelParams::COLLISION_DISTANCE, 1.0);
	n.param("infront_angle_deg", ModelParams::IN_FRONT_ANGLE_DEG, 90.0);
	n.param("driving_place", ModelParams::DRIVING_PLACE, 0);

	std::cout << "before obstacle" << std::endl;
	std::cout << "before obstacle" << std::endl;

	double noise_goal_angle_deg;
	n.param("noise_goal_angle_deg", noise_goal_angle_deg, 45.0);
	ModelParams::NOISE_GOAL_ANGLE = noise_goal_angle_deg / 180.0 * M_PI;

	n.param("max_vel", ModelParams::VEL_MAX, 2.0);
	n.param("drive_mode", Controller::b_drive_mode, 0);
	n.param("gpu_id", Controller::gpu_id, 0);
	n.param<int>("summit_port", Controller::summit_port, 0);
	n.param<float>("time_scale", Controller::time_scale, 1.0);
	n.param<std::string>("map_location", Controller::map_location, "");

	cerr << "DEBUG: Params list: " << endl;
	cerr << "-drive_mode " << Controller::b_drive_mode << endl;
	cerr << "-time_scale " << Controller::time_scale << endl;
	cerr << "-summit_port " << Controller::summit_port << endl;
	cerr << "-map_location " << Controller::map_location << endl;

	controller = new Controller(nh, fixed_path);

	ModelParams::PrintParams();

	logi << " ContextPomdpNode constructed at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	controller->RunPlanning(argc, argv);
}

int main(int argc, char** argv) {
	Globals::RecordStartTime();
	srand(unsigned(time(0)));

	ros::init(argc, argv, "context_pomdp");
	ContextPomdpNode context_pomdp_node(argc, argv);
}
