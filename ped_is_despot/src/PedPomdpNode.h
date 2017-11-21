/*
 * PedPomdpNode.h
 *
 */

#ifndef PEDESTRIAN_MOMDP_H_
#define PEDESTRIAN_MOMDP_H_

#include "controller.h"

class PedPomdpNode
{
public:
    PedPomdpNode();
    ~PedPomdpNode();
    void robotPoseCallback(geometry_msgs::PoseWithCovarianceStamped odo);
    void speedCallback(nav_msgs::Odometry odo);
    void pedPoseCallback(ped_is_despot::ped_local_frame_vector);
    void moveSpeedCallback(geometry_msgs::Twist speed);
	void publishPath();
    ros::Subscriber speedSub_, pedSub_, scanSub_, move_base_speed_;
	ros::Publisher pc_pub,path_pub;
    Controller* controller;

	bool pathPublished;
};

#endif /* PEDESTRIAN_MOMDP_H_ */
