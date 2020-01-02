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

#include <msg_builder/ped_local_frame.h>
#include <msg_builder/ped_local_frame_vector.h>
#include <msg_builder/imitation_data.h>
#include <msg_builder/car_info.h>
#include <msg_builder/peds_info.h>

#include "std_msgs/Float32.h"
#include <std_msgs/Bool.h>

#include "simulator_base.h"


using namespace despot;
class WorldBeliefTracker;

class WorldSimulator: public SimulatorBase, public World {
private:
	DSPOMDP* model_;

	PomdpStateWorld current_state_;

	int safe_action_;
	bool goal_reached_;
	double last_acc_;

	ros::Publisher cmdPub_, actionPub_, actionPubPlot_;
	ros::Publisher goal_pub, car_pub, pa_pub;
    tf::TransformListener tf_;

	std::string map_location_;
	int summit_port_;

public:
	double time_scale;
	COORD odom_vel;
	double odom_heading;
	double baselink_heading;

public:
	WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed,
			std::string map_location, int summit_port);
	~WorldSimulator();

public:

	bool Connect();
	void Connect_Carla();
	State* Initialize();
	State* GetCurrentState();
	bool ExecuteAction(ACT_TYPE action, OBS_TYPE& obs);
	double StepReward(PomdpStateWorld& state, ACT_TYPE action);

	void PublishPedsPrediciton();
	void PublishAction(int action, double reward);
	void PublishCmdAction(const ros::TimerEvent &e);
	void PublishCmdAction(ACT_TYPE);
	void PublishROSState();
    void PublishPath();

	void RobotPoseCallback(geometry_msgs::PoseWithCovarianceStamped odo);
    void SpeedCallback(nav_msgs::Odometry odo);
    void MoveSpeedCallback(geometry_msgs::Twist speed);
    void CmdSteerCallback(const std_msgs::Float32::ConstPtr steer);
    void LaneChangeCallback(const std_msgs::Int32::ConstPtr data);
    void EgoDeadCallBack(const std_msgs::Bool ego_dead);

	bool GetObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const;
	tf::Stamped<tf::Pose>GetBaseLinkPose();

    void UpdateEgoCar(const msg_builder::car_info::ConstPtr car) ;
	void UpdateCmdsNaive(ACT_TYPE action, bool buffered = false);
	void UpdateCmdsBuffered(const ros::TimerEvent &e);
};

