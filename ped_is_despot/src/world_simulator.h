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

#include "simulator_base.h"


using namespace despot;
class WorldBeliefTracker;

class WorldSimulator: public SimulatorBase, public World {
	DSPOMDP* model_;


public:
	WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed, bool pathplan_ahead, std::string obstacle_file_name, COORD car_goal);
	~WorldSimulator();

public:
	/**
	 * [Essential]
	 * Establish connection to simulator or system
	 */
	bool Connect();

	/**
	 * [Essential]
	 * Initialize or reset the (simulation) environment, return the start state if applicable
	 */
	State* Initialize();

	/**
	 * [Optional]
	 * To help construct initial belief to print debug informations in Logger
	 */
	State* GetCurrentState() const;

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
	void publishROSState();

	void robotPoseCallback(geometry_msgs::PoseWithCovarianceStamped odo);
    void speedCallback(nav_msgs::Odometry odo);
    void moveSpeedCallback(geometry_msgs::Twist speed);
	void publishPath();


	bool getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const;

	geometry_msgs::PoseStamped getPoseAhead(const tf::Stamped<tf::Pose>& carpose);
	double StepReward(PomdpStateWorld& state, ACT_TYPE action);

public:
	
	int safeAction;
	bool goal_reached;
	double last_acc_;

	//double real_speed_;
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

    ros::Subscriber speedSub_, pedSub_, mapSub_, scanSub_, move_base_speed_;

    ros::Timer timer_speed;

    tf::TransformListener tf_;

    visualization_msgs::MarkerArray markers;


public:
	// for imitation learning
	//ros::Subscriber carSub_;
	ros::Subscriber steerSub_;

	//ros::Publisher IL_pub; 
	//bool b_update_il;
	//ped_is_despot::imitation_data p_IL_data; 


    void update_il_car(const ped_is_despot::car_info::ConstPtr car) ;
    void update_il_steering(const std_msgs::Float32::ConstPtr steer);
    
	void publishImitationData(PomdpStateWorld& planning_state, ACT_TYPE safeAction, float reward, float vel);

};

