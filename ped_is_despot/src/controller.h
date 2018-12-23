/*
 * momdp.h
 *
 *  Created on: Mar 4, 2012
 *      Author: golfcar
 */

#ifndef MOMDP_H_
#define MOMDP_H_

#include <RVO.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>
//#include <sensing_on_road/pedestrian_laser_batch.h>
//#include <dataAssoc_experimental/PedDataAssoc_vector.h>
//#include <dataAssoc_experimental/PedDataAssoc.h>
#include <rosgraph_msgs/Clock.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <ped_is_despot/peds_believes.h>
#include <ped_is_despot/ped_local_frame.h>
#include <ped_is_despot/ped_local_frame_vector.h>
#include <ped_is_despot/imitation_data.h>
#include <std_msgs/Float32.h>
#include <pnc_msgs/speed_contribute.h>
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
#include "WorldModel.h"
#include "ped_pomdp.h"
#include "core/particle_belief.h"
#include "solver/despot.h"
#include <ped_pathplan/StartGoal.h>
#include "planner.h"

class WorldSimulator;
class POMDPSimulator;

class PedPomdpBelief;

using namespace std;

class Controller: public Planner
{
	enum SIM_MODE{
		POMDP,
		UNITY,
	};
public:

	PomdpStateWorld world_state;

    Controller(ros::NodeHandle& nh, bool fixed_path, double pruning_constant, double pathplan_ahead, string obstacle_file_name);

    ~Controller();

	void publishBelief();
	void publishPath(const string& frame_id, const Path& path);
	bool getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const;

	//for despot
	void initSimulator();
	void momdpInit();
	void sendPathPlanStart(const tf::Stamped<tf::Pose>& carpose);
	void RetrievePathCallBack(const nav_msgs::Path::ConstPtr path); 
    void setGoal(const geometry_msgs::PoseStamped::ConstPtr goal);
	geometry_msgs::PoseStamped getPoseAhead(const tf::Stamped<tf::Pose>& carpose);

	ros::Publisher pathPub_;
	ros::Subscriber pathSub_;
    ros::Subscriber navGoalSub_;
	ros::Publisher start_goal_pub;

	ros::NodeHandle& nh;
	//ros::Publisher markers_pubs[ModelParams::N_PED_IN];

    
	//PedPomdp * despot;
	//DESPOT* solver;
	//Simulator* gpu_handler;
	
	string global_frame_id;
	

    double goalx_, goaly_;
    std::string obstacle_file_name_;

    ACT_TYPE last_action;
    OBS_TYPE last_obs;

private:
    bool fixed_path_;
	double pathplan_ahead_;
	double control_freq;
    int X_SIZE, Y_SIZE;
    double dX, dY;
    double momdp_problem_timeout;
    bool robot_pose_available;
    double robotx_, roboty_, robotspeedx_;
    ros::Timer timer_;
    ros::Publisher plannerPedsPub_;
    ros::Publisher pedStatePub_;

    void controlLoop(const ros::TimerEvent &e);
    double StepReward(PomdpStateWorld& state, int action);

	void publishPlannerPeds(const State &);
	bool getUnityPos();
public:
	DSPOMDP* InitializeModel(option::Option* options);

	World* InitializeWorld(std::string& world_type, DSPOMDP* model, option::Option* options);

	void InitializeDefaultParameters();

	std::string ChooseSolver();

public:
	bool RunStep(Solver* solver, World* world, Logger* logger);
	int RunPlanning(int argc, char* argv[]);
	void PlanningLoop(Solver*& solver, World* world, Logger* logger);

public:
	WorldSimulator* unity_driving_simulator_;
	POMDPSimulator* pomdp_driving_simulator_;
	PedPomdpBelief* ped_belief_;
	//World* world_;
	Logger *logger_;
	DSPOMDP* model_;
	Solver* solver_;

	SIM_MODE simulation_mode_;
	
public: // lets_drive
	static int b_use_drive_net_;
	static int gpu_id_;
    static float time_scale_; // scale down the speed of time, value < 1.0
    static std::string model_file_;
    static std::string value_model_file_;
private:

	SolverPrior* prior_;

	void CreateNNPriors(DSPOMDP* model);
	bool RunPreStep(Solver* solver, World* world, Logger* logger);
private:
	Path path_from_topic;
};
#endif /* MOMDP_H_ */
