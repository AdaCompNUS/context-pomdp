#include <ped_pathplan/SendStartGoal.h>

SendStartGoal::SendStartGoal(ros::NodeHandle& nh){
	//global_frame_id = ModelParams::rosns + "/map";
	global_frame_id = "/map";
	start_goal_pub_ = nh.advertise<ped_pathplan::StartGoal> ("ped_path_planner/planner/start_goal", 1);
    nav_goal_sub_ = nh.subscribe("navgoal", 1, &SendStartGoal::setGoal, this);
	timer_ = nh.createTimer(ros::Duration(0.2), &SendStartGoal::controlLoop, this);
}


bool SendStartGoal::getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const
{
    out_pose.setIdentity();

    try {
        tf_.transformPose(target_frame, in_pose, out_pose);
    }
    catch(tf::LookupException& ex) {
        ROS_ERROR("No Transform available Error: %s\n", ex.what());
        return false;
    }
    catch(tf::ConnectivityException& ex) {
        ROS_ERROR("Connectivity Error: %s\n", ex.what());
        return false;
    }
    catch(tf::ExtrapolationException& ex) {
        ROS_ERROR("Extrapolation Error: %s\n", ex.what());
        return false;
    }
    return true;
}

void SendStartGoal::controlLoop(const ros::TimerEvent &e){

	tf::Stamped<tf::Pose> in_pose, out_pose;
  
    /****** update world state ******/
    ros::Rate err_retry_rate(10);

	// transpose to base link for path planing
	in_pose.setIdentity();
	//in_pose.frame_id_ = ModelParams::rosns + "/base_link";
	in_pose.frame_id_ = "/base_link";
	if(!getObjectPose(global_frame_id, in_pose, out_pose)) {
		cerr<<"transform error within control loop"<<endl;
		cout<<"laser frame "<<in_pose.frame_id_<<endl;
        err_retry_rate.sleep();
        return;
	}

	sendPathPlanStart(out_pose);
}


void SendStartGoal::setGoal(const geometry_msgs::PoseStamped::ConstPtr goal) {
    goalx_ = goal->pose.position.x;
    goaly_ = goal->pose.position.y;
}


void SendStartGoal::sendPathPlanStart(const tf::Stamped<tf::Pose>& carpose) {

    ped_pathplan::StartGoal startGoal;
    geometry_msgs::PoseStamped pose;
    tf::poseStampedTFToMsg(carpose, pose);

    // should plan the path with the start in front of the car, because computing the path takes time; after getting the path, 
    // the robot already moved to its front place.
/*    if(pathplan_ahead_ > 0 && worldModel.path.size()>0) {
        startGoal.start = getPoseAhead(carpose);
    } else {
        startGoal.start=pose;
    }*/ 
    startGoal.start=pose;

    pose.pose.position.x=goalx_;
    pose.pose.position.y=goaly_;
    
    startGoal.goal=pose;
    start_goal_pub_.publish(startGoal);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "send_start_goal");

    srand(unsigned(time(0)));

	ros::NodeHandle nh;
	SendStartGoal* controller = new SendStartGoal(nh);

    nh.param("goalx", controller->goalx_, -170.5);
    nh.param("goaly", controller->goaly_, -146.68);

    ros::spin();
}