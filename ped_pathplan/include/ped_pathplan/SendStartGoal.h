#ifndef SEND_START_GOAL_H_
#define SEND_START_GOAL_H_

#include <vector>
#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
// #include <nav_core/base_global_planner.h>
// #include <nav_msgs/GetPlan.h>
#include <ped_pathplan/pathplan.h>
#include <ped_pathplan/StartGoal.h>

using namespace std;

class SendStartGoal{
public:
	SendStartGoal(ros::NodeHandle& nh);

	void sendPathPlanStart(const tf::Stamped<tf::Pose>& carpose);
	void setGoal(const geometry_msgs::PoseStamped::ConstPtr goal);
	bool getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const;

	void controlLoop(const ros::TimerEvent &e);

	
	ros::Timer timer_;
	ros::Publisher start_goal_pub_;
	ros::Subscriber nav_goal_sub_;
	tf::TransformListener tf_;
	string global_frame_id;

public:
	double goalx_, goaly_;
};


#endif