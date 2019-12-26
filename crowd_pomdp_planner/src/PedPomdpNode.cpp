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
bool b_load_goal=true; // !!!set to false only for data collection purpose!!! 

PedPomdpNode::PedPomdpNode(int argc, char** argv)
{
    ROS_INFO("Starting Pedestrian Avoidance ... ");

	cerr << "DEBUG: Setting up subscription..." << endl;
    ros::NodeHandle nh;
    /// Setting up subsciption
    
    ros::NodeHandle n("~");

	pathPublished=false;
    bool simulation;
    n.param("simulation", simulation, false);
    ModelParams::init_params(simulation);
	cout << "simulation = " << simulation << endl;
	cout << "rosns = " << ModelParams::rosns << endl;

	bool fixed_path;
	double pruning_constant, pathplan_ahead;
    n.param("pruning_constant", pruning_constant, 0.001);
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


    string obstacle_file_name;
    std::cout<<"before obstacle"<<std::endl;
    n.param<std::string>("obstacle_file_name", obstacle_file_name, "null");
    std::cout<<"before obstacle"<<std::endl;
    std::cout<<obstacle_file_name<<std::endl;

    double noise_goal_angle_deg;
    n.param("noise_goal_angle_deg", noise_goal_angle_deg, 45.0);
    ModelParams::NOISE_GOAL_ANGLE = noise_goal_angle_deg / 180.0 * M_PI;

    n.param("max_vel", ModelParams::VEL_MAX, 2.0);

    n.param("drive_mode", Controller::b_drive_mode_, 0);
    n.param("gpu_id", Controller::gpu_id_, 0);
    n.param<int>("summit_port", Controller::summit_port_, 0);
    n.param<std::string>("model", Controller::model_file_, "");
    n.param<std::string>("val_model", Controller::value_model_file_, "");

    n.param<float>("time_scale", Controller::time_scale_, 1.0);

    // Carla related params
    n.param<std::string>("map_location", Controller::map_location_, "");


    cerr << "DEBUG: Params list: " << endl;
    cerr << "-drive_mode " << Controller::b_drive_mode_ << endl;
    cerr << "-model " << Controller::model_file_ << endl;
    cerr << "-time_scale " << Controller::time_scale_ << endl;
    cerr << "-summit_port " << Controller::summit_port_ << endl;
    cerr << "-map_location " << Controller::map_location_ << endl;

	cerr << "DEBUG: Creating ped_momdp instance" << endl;
	controller = new Controller(nh, fixed_path, pruning_constant, pathplan_ahead, obstacle_file_name);

	ModelParams::print_params();

    // default goal: after create door
    if(b_load_goal){// load goal from crowd_pomdp_planner.yaml file
	    n.param("goalx", controller->goalx_, 19.5);
	    n.param("goaly", controller->goaly_, 55.5);

        cout << "car goal: " << controller->goalx_ << " " << controller->goaly_ << endl;
	}
	else{// to use a list of possible goals
		srand (time(NULL));
		int which_goal =rand() % 3;
		switch(which_goal){
			case 0: controller->goalx_=0.0; controller->goaly_=18.0; break;
			case 1: controller->goalx_=18.0; controller->goaly_=0.0; break;
			case 2: controller->goalx_=0.0; controller->goaly_=-18.0; break;
		}

		cout << "No goal availabel from ros params. Using default goals\n";
		cout << "car goal: " << controller->goalx_ << " " << controller->goaly_ << endl;
	}

    logi << " PedPomdpNode constructed at the " <<  Globals::ElapsedTime() << "th second" << endl;

    controller->RunPlanning(argc, argv);

    //ros::spin();
}


PedPomdpNode::~PedPomdpNode()
{
    //controller->~ped_momdp();
}

int main(int argc, char** argv)
{
    Globals::RecordStartTime();

    // cout<< __FUNCTION__ <<"@" << __LINE__<< endl;

    ros::init(argc, argv, "mdp");
    // cout<< __FUNCTION__ <<"@" << __LINE__<< endl;

    // raise error for floating point exceptions rather than silently return nan
//    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
//    feenableexcept(FE_DIVBYZERO | FE_OVERFLOW);

    // cout<< __FUNCTION__ <<"@" << __LINE__<< endl;

    srand(unsigned(time(0)));
        cout<< __FUNCTION__ <<"@" << __LINE__<< endl;

	PedPomdpNode mdp_node(argc, argv);
}
