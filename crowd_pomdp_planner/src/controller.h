
#ifndef CONTROLLER_H_
#define CONTROLLER_H_

#include <RVO.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>
#include <rosgraph_msgs/Clock.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <msg_builder/peds_believes.h>
#include <msg_builder/ped_local_frame.h>
#include <msg_builder/ped_local_frame_vector.h>
#include <msg_builder/imitation_data.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "param.h"
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <nav_msgs/GetPlan.h>
#include "world_model.h"
#include "ped_pomdp.h"
#include "core/particle_belief.h"
#include "solver/despot.h"
#include <msg_builder/StartGoal.h>
#include "planner.h"

class WorldSimulator;
class POMDPSimulator;

class PedPomdpBelief;

using namespace std;

class Controller: public Planner
{
private:
  ros::NodeHandle& nh_;

  ros::Subscriber pathSub_;
  ros::Subscriber navGoalSub_;

  ros::Publisher pedStatePub_;
  ros::Publisher plannerPedsPub_;
  ros::Publisher start_goal_pub_;
  ros::Publisher pathPub_;

  ros::Timer timer_;

  string global_frame_id_;
  ACT_TYPE last_action_;
  OBS_TYPE last_obs_;
  bool fixed_path_;
  double control_freq_;

  WorldSimulator* summit_driving_simulator_;
  PedPomdpBelief* ped_belief_;
  DSPOMDP* model_;
  SolverPrior* prior_;
  Path path_from_topic_;

public:

  Controller(ros::NodeHandle& nh, bool fixed_path);
  ~Controller();

private:

  void ControlLoop(const ros::TimerEvent &e);
  double StepReward(PomdpStateWorld& state, int action);

	void PublishPath(const string& frame_id, const Path& path);
	bool GetEgoPosFromSummit();
	void RetrievePathCallBack(const nav_msgs::Path::ConstPtr path);

	void PredictPedsForSearch(State* search_state);
	void UpdatePriors(const State* cur_state, State* search_state);
	void TruncPriors(int cur_search_hist_len);
	void CreateDefaultPriors(DSPOMDP* model);

	DSPOMDP* InitializeModel(option::Option* options);
	World* InitializeWorld(std::string& world_type, DSPOMDP* model, option::Option* options);
	void InitializeDefaultParameters();
	std::string ChooseSolver();

	bool RunStep(despot::Solver* solver, World* world, Logger* logger);
	void PlanningLoop(despot::Solver*& solver, World* world, Logger* logger);

public:
	int RunPlanning(int argc, char* argv[]);

public:
	static int b_drive_mode;
	static int gpu_id;
  static int summit_port;
  static float time_scale; // scale down the speed of time, value < 1.0
  static std::string map_location;
};
#endif /* CONTROLLER_H_ */
