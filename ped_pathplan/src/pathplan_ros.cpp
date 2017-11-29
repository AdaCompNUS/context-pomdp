#include <ped_pathplan/pathplan_ros.h>

namespace ped_pathplan {
    using namespace std;
    using namespace costmap_2d;
    PathPlanROS::PathPlanROS()
        : costmap_ros(NULL), initialized(false), allow_unknown(false)
    {}
    PathPlanROS::PathPlanROS(string name, Costmap2DROS* costmap)
        : costmap_ros(costmap), initialized(false), allow_unknown(false) {
        initialize(name, costmap_ros);
    }

    void PathPlanROS::initialize(string name, Costmap2DROS* cmros) {
        if(!initialized) {
            ros::NodeHandle private_nh("~/" + name);
            double steering_limit_deg, yaw_res_deg, cost_steering_deg;
            double discretize_ratio;
            double discount;
            int step;
			int num_search;
			double carlen;

            private_nh.param("steering_limit_deg", steering_limit_deg, 5.0);
            private_nh.param("yaw_res_deg", yaw_res_deg, 2.5);
            private_nh.param("cost_steering_deg", cost_steering_deg, 100.0);
            private_nh.param("step", step, 3);
            private_nh.param("num_search", num_search, 500000);
            private_nh.param("discretize_ratio", discretize_ratio, 0.3);
            private_nh.param("discount", discount, 1.0);
            private_nh.param("car_len", carlen, 1.0);

            costmap_ros = cmros;

            int nx = cmros->getSizeInCellsX();
            int ny = cmros->getSizeInCellsY();
            float map_res = cmros->getResolution(); //map resolution 
            planner = boost::shared_ptr<PathPlan>(new PathPlan(nx, ny, steering_limit_deg, yaw_res_deg, cost_steering_deg, step, num_search, discretize_ratio, discount, map_res, carlen));

            updateCostmap();

			plan_pub = private_nh.advertise<nav_msgs::Path>("plan", 1, false);

            //string global_frame = cmros->getGlobalFrameID();
            make_plan_srv =  private_nh.advertiseService("make_plan", &PathPlanROS::makePlanService, this);
			make_plan_sub =  private_nh.subscribe("start_goal",1,&PathPlanROS::makePlanAsync, this);
            //nav_goal_sub = nh.subscribe("navgoal", 1, &Controller::setGoal, this);


            ros::NodeHandle prefix_nh;
            tf_prefix = tf::getPrefixParam(prefix_nh);

            initialized = true;
        } else {
          ROS_WARN("This planner has already been initialized, you can't call it twice, doing nothing");
        }

    }

	void PathPlanROS::makePlanAsync(const ped_pathplan::StartGoal::ConstPtr & startGoal) {
		std::vector<geometry_msgs::PoseStamped> plan;
		//cout<<"================ makePlanAsync"<<endl;
		makePlan(startGoal->start, startGoal->goal, plan);
	}

/*    void PathPlanROS::makePlanAsync(const ped_pathplan::StartGoal startGoal) {
        std::vector<geometry_msgs::PoseStamped> plan;
        makePlan(startGoal.start, startGoal.goal, plan);
    }*/

    bool PathPlanROS::makePlanService(nav_msgs::GetPlan::Request& req, nav_msgs::GetPlan::Response& resp) {
        bool ret = makePlan(req.start, req.goal, resp.plan.poses);
        resp.plan.header.stamp = ros::Time::now();
        resp.plan.header.frame_id = costmap_ros->getGlobalFrameID();

        return ret;
    }

    bool PathPlanROS::makePlan(const geometry_msgs::PoseStamped& start, 
            const geometry_msgs::PoseStamped& goal, std::vector<geometry_msgs::PoseStamped>& plan) {
        boost::mutex::scoped_lock lock(mutex);
		//cout<<"***************** makeplan"<<endl;
        if(!initialized){
            ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
            return false;
        }

        plan.clear();
        updateCostmap();

        ros::NodeHandle n;

        if(tf::resolve(tf_prefix, goal.header.frame_id) != tf::resolve(tf_prefix, costmap_ros->getGlobalFrameID())){
            ROS_ERROR("The goal pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", 
                    tf::resolve(tf_prefix, costmap_ros->getGlobalFrameID()).c_str(), tf::resolve(tf_prefix, goal.header.frame_id).c_str());
            return false;
        }

        if(tf::resolve(tf_prefix, start.header.frame_id) != tf::resolve(tf_prefix, costmap_ros->getGlobalFrameID())){
            ROS_ERROR("The start pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", 
                    tf::resolve(tf_prefix, costmap_ros->getGlobalFrameID()).c_str(), tf::resolve(tf_prefix, start.header.frame_id).c_str());
            return false;
        }

        double wx = start.pose.position.x;
        double wy = start.pose.position.y;
        double yaw = tf::getYaw(start.pose.orientation);

        unsigned int mx, my;
        if(!costmap.worldToMap(wx, wy, mx, my)){
            ROS_WARN("The robot's start position is off the global costmap. Planning will always fail, are you sure the robot has been properly localized?");
            return false;
        }

        tf::Stamped<tf::Pose> start_pose;
        tf::poseStampedMsgToTF(start, start_pose);

        planner->setCostmap(costmap.getCharMap(), true, allow_unknown);

        State start_state = {float(mx),float(my), float(yaw)};

        wx = goal.pose.position.x;
        wy = goal.pose.position.y;

        if(!costmap.worldToMap(wx, wy, mx, my)){
            ROS_WARN("The goal sent to the navfn planner is off the global costmap. Planning will always fail to this goal.");
            return false;
        }

        State goal_state = {float(mx), float(my), 0};
        planner->setStart(start_state);
        planner->setGoal(goal_state);

        auto path_states = planner->calcPath();

        // extract the plan
        ros::Time plan_time = ros::Time::now();
        std::string global_frame = costmap_ros->getGlobalFrameID();
        for(const auto& s: path_states) {
            double world_x, world_y;
            mapToWorld(s[0], s[1], world_x, world_y);

            geometry_msgs::PoseStamped pose;
            pose.header.stamp = plan_time;
            pose.header.frame_id = global_frame;
            pose.pose.position.x = world_x;
            pose.pose.position.y = world_y;
            pose.pose.position.z = 0.0;
            pose.pose.orientation.x = 0.0;
            pose.pose.orientation.y = 0.0;
            pose.pose.orientation.z = 0.0;
            pose.pose.orientation.w = 1.0;
            plan.push_back(pose);
        }

/*        double world_x_start = -205, world_y_start = 0,
        world_x_end = , world_y_end = 

        for(int i = 0; i < ) {
            double world_x, world_y;

            geometry_msgs::PoseStamped pose;
            pose.header.stamp = plan_time;
            pose.header.frame_id = global_frame;
            pose.pose.position.x = world_x;
            pose.pose.position.y = world_y;
            pose.pose.position.z = 0.0;
            pose.pose.orientation.x = 0.0;
            pose.pose.orientation.y = 0.0;
            pose.pose.orientation.z = 0.0;
            pose.pose.orientation.w = 1.0;
            plan.push_back(pose);
        }*/

		publishPlan(plan);

        return !plan.empty();
    }

/*    void PathPlanROS::setGoal(const geometry_msgs::PoseStamped::ConstPtr goal) {
        goalx_ = goal->pose.position.x;
        goaly_ = goal->pose.position.y;
    }*/

/*    void PathPlanROS::sendPathPlanStart(const tf::Stamped<tf::Pose>& carpose) {
        if(fixed_path_ && worldModel.path.size()>0)  return;

        ped_pathplan::StartGoal startGoal;
        geometry_msgs::PoseStamped pose;
        tf::poseStampedTFToMsg(carpose, pose);

        // set start
        if(pathplan_ahead_ > 0 && worldModel.path.size()>0) {
            startGoal.start = getPoseAhead(carpose);
        } else {
            startGoal.start=pose;
        }

        pose.pose.position.x=goalx_;
        pose.pose.position.y=goaly_;
        
        startGoal.goal=pose;
        start_goal_pub.publish(startGoal);  
        //makePlanAsync(startGoal);
    }*/

    void PathPlanROS::updateCostmap() {
        costmap_ros->getCostmapCopy(costmap);
    }

    void PathPlanROS::mapToWorld(double mx, double my, double& wx, double& wy) {
        wx = costmap.getOriginX() + mx * costmap.getResolution();
        wy = costmap.getOriginY() + my * costmap.getResolution();
    }

	void PathPlanROS::publishPlan(const std::vector<geometry_msgs::PoseStamped>& path){
		if(!initialized){
			ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
			return;
		}

		//create a message for the plan 
		nav_msgs::Path gui_path;
		gui_path.poses.resize(path.size());

		if(!path.empty())
		{
			gui_path.header.frame_id = path[0].header.frame_id;
			gui_path.header.stamp = path[0].header.stamp;
		}

		// Extract the plan in world co-ordinates, we assume the path is all in the same frame
		for(unsigned int i=0; i < path.size(); i++){
			gui_path.poses[i] = path[i];
		}

		plan_pub.publish(gui_path);
	}

}
