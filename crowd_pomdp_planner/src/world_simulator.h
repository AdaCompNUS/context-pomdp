#include <interface/world.h>
#include <string>
#include <ros/ros.h>
#include "context_pomdp.h"
#include "param.h"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>

#include <rosgraph_msgs/Clock.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
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

#include <msg_builder/car_info.h>
#include <msg_builder/peds_info.h>
#include <msg_builder/TrafficAgentArray.h>
#include <msg_builder/AgentPathArray.h>

#include "std_msgs/Float32.h"
#include <std_msgs/Bool.h>

#include "simulator_base.h"


using namespace despot;

class WorldSimulator: public SimulatorBase, public World {
private:
	DSPOMDP* model_;
    WorldModel& worldModel;

	double car_time_stamp_;
	double agents_time_stamp_;
	double paths_time_stamp_;
	CarStruct car_;
	std::map<int, AgentStruct> exo_agents_;

	PomdpStateWorld current_state_;
	Path path_from_topic_;

	int safe_action_;
	bool goal_reached_;
	double last_acc_;

	ros::Publisher cmdPub_;
	ros::Subscriber ego_sub_, ego_dead_sub_, pathSub_, agent_sub_, agent_path_sub_;

	std::string map_location_;
	int summit_port_;

public:
	double time_scale;

public:
	WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed,
			std::string map_location, int summit_port);
	~WorldSimulator();

public:

	bool Connect();
	void Connect_Carla();
	State* Initialize();
	std::map<double, AgentStruct&> GetSortedAgents();
	State* GetCurrentState();
	bool ExecuteAction(ACT_TYPE action, OBS_TYPE& obs);
	double StepReward(PomdpStateWorld& state, ACT_TYPE action);
	bool Emergency(PomdpStateWorld* curr_state);

	void UpdateCmds(ACT_TYPE action, bool emergency = false);
	void PublishCmdAction(const ros::TimerEvent &e);
	void PublishCmdAction(ACT_TYPE);

	void EgoDeadCallBack(const std_msgs::Bool ego_dead);
	void EgoStateCallBack(const msg_builder::car_info::ConstPtr car);
	void RetrievePathCallBack(const nav_msgs::Path::ConstPtr path);

	void AgentArrayCallback(msg_builder::TrafficAgentArray data);
	void AgentPathArrayCallback(msg_builder::AgentPathArray data);

};

