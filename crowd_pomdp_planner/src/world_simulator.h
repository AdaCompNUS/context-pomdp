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

#include "simulator_base.h"


using namespace despot;
class WorldBeliefTracker;

class WorldSimulator: public SimulatorBase, public World {
	DSPOMDP* model_;


public:
	WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed, bool pathplan_ahead,
			std::string obstacle_file_name, std::string map_location, int carla_port, COORD car_goal);
	~WorldSimulator();

public:
	/**
	 * [Essential]
	 * Establish connection to simulator or system
	 */
	bool Connect();

	// for carla
	void Connect_Carla();

	/**
	 * [Essential]
	 * Initialize or reset the (simulation) environment, return the start state if applicable
	 */
	State* Initialize();

	/**
	 * [Optional]
	 * To help construct initial belief to print debug informations in Logger
	 */
	State* GetCurrentState();

	/**
	 * [Essential]
	 * send action, receive reward, obs, and terminal
	 * @param action Action to be executed in the real-world system
	 * @param obs    Observation sent back from the real-world system
	 */
	bool ExecuteAction(ACT_TYPE action, OBS_TYPE& obs);

	void AddObstacle();

	void publishPedsPrediciton();
	void publishAction(int action, double reward);
	void publishCmdAction(const ros::TimerEvent &e);
	void publishCmdAction(ACT_TYPE);
	void publishROSState();

	void robotPoseCallback(geometry_msgs::PoseWithCovarianceStamped odo);
    void speedCallback(nav_msgs::Odometry odo);
    void moveSpeedCallback(geometry_msgs::Twist speed);
    void cmdSteerCallback(const std_msgs::Float32::ConstPtr steer);
    void lane_change_Callback(const std_msgs::Int32::ConstPtr data);
	void publishPath();


	bool getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const;
	tf::Stamped<tf::Pose>GetBaseLinkPose();
	geometry_msgs::PoseStamped getPoseAhead(const tf::Stamped<tf::Pose>& carpose);
	double StepReward(PomdpStateWorld& state, ACT_TYPE action);

public:
	
	int safeAction;
	bool goal_reached;
	double last_acc_;

	//double real_speed_;
	COORD odom_vel_;
	double odom_heading_;
	double baselink_heading_;
	//double target_speed_;

    //double steering_;


    bool pathplan_ahead_;

   	//static WorldStateTracker* stateTracker;
   	//WorldBeliefTracker* beliefTracker;

    //static WorldModel worldModel;

	//std::string global_frame_id;
    //std::string obstacle_file_name_;


	ros::Publisher goal_pub;
	ros::Publisher car_pub;
	ros::Publisher pa_pub;
	ros::Publisher cmdPub_, actionPub_, actionPubPlot_;

    ros::Subscriber speedSub_, laneSub_, obsSub_, agentSub_, agentpathSub_,
    	mapSub_, scanSub_, move_base_speed_, steerSub_, lane_change_Sub_;

    ros::Timer timer_speed;
    ros::Timer timer_cmd_update;

    tf::TransformListener tf_;

	PomdpStateWorld current_state;

private:
	std::string map_location_;
	int carla_port_;

public:

    void update_il_car(const msg_builder::car_info::ConstPtr car) ;
    void update_il_steering(const std_msgs::Float32::ConstPtr steer);
    
	void publishImitationData(PomdpStateWorld& planning_state, ACT_TYPE safeAction, float reward, float vel);

	void update_cmds_fix_latency(ACT_TYPE action, bool buffered = false);

	void update_cmds_naive(ACT_TYPE action, bool buffered = false);

	void update_cmds_buffered(const ros::TimerEvent &e);

public:
	void Debug_action();

	void setCarGoal(COORD);

public:
	void updateLanes(COORD car_pos);
	void updateObs(COORD car_pos);
};

