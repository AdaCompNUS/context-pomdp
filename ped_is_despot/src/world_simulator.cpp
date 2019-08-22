
#include "world_simulator.h"
#include "WorldModel.h"
#include "ped_pomdp.h"

#include <ped_pathplan/StartGoal.h>
#include <ped_is_despot/car_info.h>
#include <ped_is_despot/peds_info.h>
#include <ped_is_despot/ped_info.h>
#include <ped_is_despot/peds_believes.h>
#include "ros/ros.h"

#include <nav_msgs/OccupancyGrid.h>

#define CONNECTOR2

#include <carla_connector/agent_array.h>

#ifdef CONNECTOR2
#include <carla_connector2/TrafficAgentArray.h>
#endif

#include "neural_prior.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

WorldStateTracker* SimulatorBase::stateTracker;

WorldModel SimulatorBase::worldModel;
bool SimulatorBase::agents_data_ready = false;
double pub_frequency = 9.0;

void pedPoseCallback(ped_is_despot::ped_local_frame_vector);
#ifdef CONNECTOR2
	void agentArrayCallback(carla_connector2::TrafficAgentArray);
#else
	void agentArrayCallback(carla_connector::agent_array);
#endif
void receive_map_callback(nav_msgs::OccupancyGrid map);

COORD poseToCoord(const tf::Stamped<tf::Pose>& pose) {
	COORD coord;
	coord.x=pose.getOrigin().getX();
	coord.y=pose.getOrigin().getY();
	return coord;
}


double poseToHeadingDir(const tf::Stamped<tf::Pose>& pose) {
	/* get yaw angle [-pi, pi)*/
	double yaw;
	yaw=tf::getYaw(pose.getRotation());
	if (yaw<0) yaw+=2* 3.1415926;
	return yaw;
}

double navposeToHeadingDir(const geometry_msgs::Pose & msg) {
	/* get yaw angle [-pi, pi)*/
	tf::Pose pose;
	tf::poseMsgToTF(msg, pose);

	double yaw;
	yaw=tf::getYaw(pose.getRotation());
	if (yaw<0) yaw+=2* 3.1415926;
	return yaw;
}


WorldSimulator::WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed, bool pathplan_ahead, std::string obstacle_file_name, COORD car_goal): 
	SimulatorBase(_nh, obstacle_file_name), pathplan_ahead_(pathplan_ahead),
	model_(model),
	World(){
		
	global_frame_id = ModelParams::rosns + "/map";

	logi<<__FUNCTION__<<" global_frame_id: "<<global_frame_id <<" "<< endl;


	ros::NodeHandle n("~");
    n.param<std::string>("goal_file_name", worldModel.goal_file_name_, "null");

	worldModel.InitPedGoals();
	worldModel.InitRVO();
	worldModel.car_goal=car_goal;
    AddObstacle();

    stateTracker=new WorldStateTracker(worldModel);
}

WorldSimulator::~WorldSimulator(){
    geometry_msgs::Twist cmd;
    cmd.angular.z = 0;
    cmd.linear.x = 0;
    cmdPub_.publish(cmd);
}
/**
 * [Essential]
 * Establish connection to simulator or system
 */
bool WorldSimulator::Connect(){
	cerr << "DEBUG: Connecting with world" << endl;

    cmdPub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel_pomdp",1);
    actionPub_ = nh.advertise<visualization_msgs::Marker>("pomdp_action",1);
    actionPubPlot_= nh.advertise<geometry_msgs::Twist>("pomdp_action_plot",1);
    
    pa_pub=nh.advertise<geometry_msgs::PoseArray>("my_poses",1000);
	car_pub=nh.advertise<geometry_msgs::PoseStamped>("car_pose",1000);
	goal_pub=nh.advertise<visualization_msgs::MarkerArray> ("pomdp_goals",1);
    IL_pub = nh.advertise<ped_is_despot::imitation_data>("il_data", 1);

	speedSub_ = nh.subscribe("odom", 1, &WorldSimulator::speedCallback, this);
    // pedSub_ = nh.subscribe("ped_local_frame_vector", 1, pedPoseCallback); 
    mapSub_ = nh.subscribe("map", 1, receive_map_callback); // nav_msgs::OccupancyGrid

  	carSub_ = nh.subscribe("IL_car_info", 1, &WorldSimulator::update_il_car, this);

    agentSub_ = nh.subscribe("agent_array", 1, agentArrayCallback); 

	clock_t begin = clock();

    auto odom_data = ros::topic::waitForMessage<nav_msgs::Odometry>("odom");
    auto car_data = ros::topic::waitForMessage<ped_is_despot::car_info>("IL_car_info");
#ifdef CONNECTOR2
	auto agent_data = ros::topic::waitForMessage<carla_connector2::TrafficAgentArray>("agent_array");
#else
	auto agent_data = ros::topic::waitForMessage<carla_connector::agent_array>("agent_array");
#endif   

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	logi << "[WorldSimulator] Simulator connected after waiting for " 
		<< elapsed_secs << " s" << endl;

    return true;
}

/**
 * [Essential]
 * Initialize or reset the (simulation) environment, return the start state if applicable
 */
State* WorldSimulator::Initialize(){

	cerr << "DEBUG: Initializing world" << endl;

	safeAction=2;
	target_speed_=0.0;
	steering_ = 0.0;
	goal_reached=false;
	last_acc_=0;
	b_update_il = true;

	return NULL;
}

/**
 * [Optional]
 * To help construct initial belief to print debug informations in Logger
 */
State* WorldSimulator::GetCurrentState(){
	
	/* Get current car coordinates */
    tf::Stamped<tf::Pose> in_pose, out_pose;
	// transpose to laser frame for ped avoidance
	in_pose.setIdentity();
	in_pose.frame_id_ = ModelParams::rosns + ModelParams::laser_frame;

	// in_pose.frame_id_ = ModelParams::rosns + "odom";

	// cout<<__FUNCTION__<<" global_frame_id: "<<global_frame_id<<" "<<endl;

	/*if(!getObjectPose(global_frame_id, in_pose, out_pose)) {
		cerr<<"transform error within GetCurrentState"<<endl;
		cout<<"laser frame "<<in_pose.frame_id_<<endl;
		ros::Rate err_retry_rate(10);
        err_retry_rate.sleep();
        return NULL; // no up-to-date state
	}*/

	while(!getObjectPose(global_frame_id, in_pose, out_pose)) {
		cerr<<"transform error within GetCurrentState"<<endl;
		cout<<"laser frame "<<in_pose.frame_id_<<endl;
		ros::Rate err_retry_rate(10);
        err_retry_rate.sleep();
        //return NULL; // no up-to-date state
	}

	CarStruct updated_car;
	COORD coord;
	//ped_pathplan::StartGoal startGoal;

	/*if(pathplan_ahead_ > 0 && worldModel.path.size()>0) {
		startGoal.start = getPoseAhead(out_pose);
		coord.x = startGoal.start.pose.position.x;
		coord.y = startGoal.start.pose.position.y;
	}else{*/
	coord = poseToCoord(out_pose);
	//}

	cout << "======transformed pose = "<< coord.x << " " <<coord.y << endl;

	updated_car.pos=coord;
	updated_car.vel=real_speed_;
	updated_car.heading_dir = poseToHeadingDir(out_pose);

	// DEBUG("update");
	stateTracker->updateCar(updated_car);
	stateTracker->cleanAgents();

	// DEBUG("state");
	PomdpStateWorld state = stateTracker->getPomdpWorldState();

	current_state.assign(state);

	if (logging::level()>=4)
		static_cast<PedPomdp*>(model_)->PrintWorldState(current_state);

	current_state.time_stamp = SolverPrior::get_timestamp();

	logi << " current state time stamp " <<  current_state.time_stamp << endl;

	// logi << "&current_state " << &current_state << endl;
	return static_cast<State*>(&current_state);
	// return NULL;
}

double WorldSimulator::StepReward(PomdpStateWorld& state, ACT_TYPE action){
	double reward = 0.0;

	if (worldModel.isGlobalGoal(state.car)) {
		reward = ModelParams::GOAL_REWARD;
		return reward;
	}

	PedPomdp* pedpomdp_model = static_cast<PedPomdp*>(model_);

	if(state.car.vel > 0.001 && worldModel.inRealCollision(state) ) { /// collision occurs only when car is moving
		reward = pedpomdp_model->CrashPenalty(state);
		return reward;
	}

	// Smoothness control
	reward += pedpomdp_model->ActionPenalty(action);

	// Speed control: Encourage higher speed
  if( Globals::config.use_prior)	
    reward += pedpomdp_model->MovementPenalty(state);
  else
    reward += pedpomdp_model->MovementPenalty(state, pedpomdp_model->GetSteering(action));
	return reward;
}

/**
 * [Essential]
 * send action, receive reward, obs, and terminal
 * @param action Action to be executed in the real-world system
 * @param obs    Observation sent back from the real-world system
 */
bool WorldSimulator::ExecuteAction(ACT_TYPE action, OBS_TYPE& obs){

	if(action ==-1) return false;

	cout << "[ExecuteAction]" << endl;

	ros::spinOnce();

	cout << "[ExecuteAction] after spin" << endl;

	/* Update state */
	PomdpStateWorld* curr_state = static_cast<PomdpStateWorld*>(GetCurrentState());
	double acc;
	double steer;

	/* Reach goal: no more action sent */
	if(worldModel.isGlobalGoal(curr_state->car)) {
		cout << "--------------------------- goal reached ----------------------------" << endl;
		cout << "" << endl;
		// Stop the car after reaching goal
		acc = static_cast<PedPomdp*>(model_)->GetAccfromAccID(PedPomdp::ACT_DEC);
		steer = 0;
		action=static_cast<PedPomdp*>(model_)->GetActionID(steer, acc);
		goal_reached=true;
		//return true;
	}

	/* Collision: slow down the car */
	int collision_peds_id;
	if( curr_state->car.vel > 0.001 * time_scale_ && worldModel.inRealCollision(*curr_state,collision_peds_id) ) {
		cout << "--------------------------- collision = 1 ----------------------------" << endl;
		cout << "collision ped: " << collision_peds_id<<endl;
		
		acc = static_cast<PedPomdp*>(model_)->GetAccfromAccID(PedPomdp::ACT_DEC);
		steer = 0;
		action=static_cast<PedPomdp*>(model_)->GetActionID(steer, acc);
	}


	/* Publish action and step reward */
	double step_reward=StepReward(*curr_state,action);
	cout<<"action **= "<<action<<endl;
	cout<<"reward **= "<<step_reward<<endl;
	safeAction=action;

//	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Publish action (for data collection)"<<endl;
//	publishAction(action, step_reward);

	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Update steering and target speed"<<endl;

	/* Publish steering and target velocity to Unity */
	if (goal_reached == true){
		//ros::shutdown();
		steering_ = 0;
		target_speed_=real_speed_;

	    target_speed_ -= ModelParams::AccSpeed;
		if(target_speed_<=0.0) target_speed_ = 0.0;

		buffered_action_ = static_cast<PedPomdp*>(model_)->GetActionID(0.0, -ModelParams::AccSpeed);

        // shutdown the node after reaching goal
        // TODO consider do this in simulaiton only for safety
        if(real_speed_ <= 0.01 * time_scale_) {
        	cout<<"After reaching goal: real_spead is already zero"<<endl;
            ros::shutdown();
        }
		//return;
	}
	else if(stateTracker->emergency()){
		steering_ = 0;
		target_speed_=-1;

		buffered_action_ = static_cast<PedPomdp*>(model_)->GetActionID(0.0, -ModelParams::AccSpeed);

		cout<<"--------------------------- emergency ----------------------------" <<endl;
	}
	else if (worldModel.path.size()>0 && COORD::EuclideanDistance(stateTracker->carpos, worldModel.path[0])>4.0){
		cerr << "=================== Path offset too high !!! Node shutting down" << endl;
		ros::shutdown();
	}
	else{
//		get_action_fix_latency(action);
//		update_cmds_naive(action);
		buffered_action_ = action;
	}

	Debug_action();

	// Updating cmd steering and speed. They will be send to vel_pulisher by timer
	update_cmds_naive(buffered_action_, false);

	publishCmdAction(buffered_action_);

	publishImitationData(*curr_state, action, step_reward, target_speed_);

	/* Receive observation.
	 * Caution: obs info is not up-to-date here. Don't use it
	 */

	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Generate obs"<<endl;
	obs=static_cast<PedPomdp*>(model_)->StateToIndex(GetCurrentState());

	//worldModel.traffic_agent_sim_[0] -> OutputTime();
	return goal_reached;
}

//void WorldSimulator::update_cmds_fix_latency(ACT_TYPE action){
//	buffered_action_ = action;
//	cout<<"current steering = "<<steering_<<endl;
//	cout<<"current target_speed = "<<target_speed_<<endl;
//	cout<<"last_acc_ = "<<last_acc_<<endl;
//	cout<<"real_speed_ = "<<real_speed_<<endl;
//
//	/* Do look-ahead to adress the latency in real_speed_ */
//	double predicted_speed = min(max(real_speed_+last_acc_ , 0.0), ModelParams::VEL_MAX);
//	double last_speed = predicted_speed;
//	cout<<"predicted real speed: "<<real_speed_+last_acc_<<endl;
//
//	float acc=static_cast<PedPomdp*>(model_)->GetAcceleration(action);
//	last_acc_=acc/ModelParams::control_freq;
//
//	cout<<"applying step acc: "<< last_acc_<<endl;
//
//	cout<<"trunc to max car vel: "<< ModelParams::VEL_MAX<<endl;
//
//	target_speed_ = min(predicted_speed + last_acc_, ModelParams::VEL_MAX);
//	target_speed_ = max(target_speed_, 0.0);
//
//	last_acc_=target_speed_-last_speed;
//
//	steering_=static_cast<PedPomdp*>(model_)->GetSteering(action);
//
//	cerr << "DEBUG: Executing action:" << action << " steer/acc = " << steering_ << "/" << acc << endl;
//	cerr << "action IDs steer/acc = " << static_cast<PedPomdp*>(model_)->GetSteeringID(action)
//			<< "/" << static_cast<PedPomdp*>(model_)->GetAccfromAccID(action) << endl;
//
//	cout<<"cmd steering = "<<steering_<<endl;
//	cout<<"cmd target_speed = "<<target_speed_<<endl;
//}

void WorldSimulator::update_cmds_fix_latency(ACT_TYPE action, bool buffered){
//	buffered_action_ = action;

	cout<<"real_speed_ = "<<real_speed_<<endl;
	cout<<"speed_in_search_state_ = "<<speed_in_search_state_<<endl;

	float acc=static_cast<PedPomdp*>(model_)->GetAcceleration(action);

	float accelaration;
	if (!buffered)
		accelaration=acc/ModelParams::control_freq;
	else if (buffered)
		accelaration=acc/pub_frequency;

	cout<<"applying step acc: "<< accelaration<<endl;
	cout<<"trunc to max car vel: "<< ModelParams::VEL_MAX<<endl;

	target_speed_ = min(speed_in_search_state_ + accelaration, ModelParams::VEL_MAX);
	target_speed_ = max(target_speed_, 0.0);

	steering_=static_cast<PedPomdp*>(model_)->GetSteering(action);

	cerr << "DEBUG: Executing action:" << action << " steer/acc = " << steering_ << "/" << acc << endl;
	cerr << "action IDs steer/acc = " << static_cast<PedPomdp*>(model_)->GetSteeringID(action)
				<< "/" << static_cast<PedPomdp*>(model_)->GetAccelerationID(action) << endl;
	cout<<"cmd steering = "<<steering_<<endl;
	cout<<"cmd target_speed = "<<target_speed_<<endl;
}

void WorldSimulator::update_cmds_naive(ACT_TYPE action, bool buffered){
//	buffered_action_ = action;

	SolverPrior::nn_priors[0]->Check_force_steer(stateTracker->car_heading_dir, 
		worldModel.path.getCurDir(), stateTracker->carvel);

	worldModel.path.text();

	cout << "[update_cmds_naive] Buffering action " << action << endl;

//	cout<<"real_speed_ = "<<real_speed_<<endl;

	float acc=static_cast<PedPomdp*>(model_)->GetAcceleration(action);

//	float accelaration;
//	if (!buffered)
//		accelaration=acc/ModelParams::control_freq;
//	else if (buffered)
//		accelaration=acc/pub_frequency;
//
//	target_speed_ = min(real_speed_ + accelaration, ModelParams::VEL_MAX);
//	target_speed_ = max(target_speed_, 0.0);


	double speed_step = ModelParams::AccSpeed / ModelParams::control_freq;
	int level = real_speed_/speed_step;
	if (std::abs(real_speed_ - (level+1) * speed_step) < speed_step * 0.3){
		level = level + 1;
	}

	int next_level = level;
	if (acc > 0.0)
		next_level = level + 1;
	else if (acc < 0.0)
		next_level = max(0, level - 1);

	target_speed_ = min(next_level * speed_step, ModelParams::VEL_MAX);

	steering_=static_cast<PedPomdp*>(model_)->GetSteering(action);

	cerr << "DEBUG: Executing action:" << action << " steer/acc = " << steering_ << "/" << acc << endl;

//	target_speed_ = 0; // Freezing robot motion. Debugging !!!!!!!!
}

//void WorldSimulator::update_cmds_buffered(const ros::TimerEvent &e){
//	update_cmds_naive(buffered_action_, true);
////	update_cmds_fix_latency(buffered_action_, true);
//
//	cout<<"[update_cmds_buffered] time stamp" << SolverPrior::nn_priors[0]->get_timestamp() <<endl;
//}

void WorldSimulator::AddObstacle(){


	if(obstacle_file_name_ == "null"){
		/// for indian_cross
		cout << "Using default obstacles " << endl;

		std::vector<RVO::Vector2> obstacle[4];

	    obstacle[0].push_back(RVO::Vector2(-4,-4));
	    obstacle[0].push_back(RVO::Vector2(-30,-4));
	    obstacle[0].push_back(RVO::Vector2(-30,-30));
	    obstacle[0].push_back(RVO::Vector2(-4,-30));

	    obstacle[1].push_back(RVO::Vector2(4,-4));
	    obstacle[1].push_back(RVO::Vector2(4,-30));
	    obstacle[1].push_back(RVO::Vector2(30,-30));
	    obstacle[1].push_back(RVO::Vector2(30,-4));

		obstacle[2].push_back(RVO::Vector2(4,4));
	    obstacle[2].push_back(RVO::Vector2(30,4));
	    obstacle[2].push_back(RVO::Vector2(30,30));
	    obstacle[2].push_back(RVO::Vector2(4,30));

		obstacle[3].push_back(RVO::Vector2(-4,4));
	    obstacle[3].push_back(RVO::Vector2(-4,30));
	    obstacle[3].push_back(RVO::Vector2(-30,30));
	    obstacle[3].push_back(RVO::Vector2(-30,4));


	    int NumThreads=1/*Globals::config.NUM_THREADS*/;

	    for(int tid=0; tid<NumThreads;tid++){
		    for (int i=0; i<4; i++){
		 	   worldModel.traffic_agent_sim_[tid]->addObstacle(obstacle[i]);
			}

		    /* Process the obstacles so that they are accounted for in the simulation. */
		    worldModel.traffic_agent_sim_[tid]->processObstacles();
		}
	} else{
		cout << "Skipping obstacle file " << obstacle_file_name_ << endl;
		return;

		cout << "Using obstacle file " << obstacle_file_name_ << endl;

		int NumThreads=1/*Globals::config.NUM_THREADS*/;

		ifstream file;
	    file.open(obstacle_file_name_, std::ifstream::in);

	    if(file.fail()){
	        cout<<"open obstacle file failed !!!!!!"<<endl;
	        return;
	    }

	    std::vector<RVO::Vector2> obstacles[100];

	    std::string line;
	    int obst_num = 0;
	    while (std::getline(file, line))
	    {
	        std::istringstream iss(line);
	        
	        double x;
	        double y;
	        cout << "obstacle shape: ";
	        while (iss >> x >> y){
	            cout << x <<" "<< y << " ";
	            obstacles[obst_num].push_back(RVO::Vector2(x, y));
	        }
	        cout << endl;

	        for(int tid=0; tid<NumThreads;tid++){
			 	worldModel.traffic_agent_sim_[tid]->addObstacle(obstacles[obst_num]);			
			}

	        worldModel.AddObstacle(obstacles[obst_num]);

	        obst_num++;
	        if(obst_num > 99) break;
	    }

	    for(int tid=0; tid<NumThreads;tid++){
			worldModel.traffic_agent_sim_[tid]->processObstacles();			    
		}
	    
	    file.close();
	}

}


void WorldSimulator::publishCmdAction(const ros::TimerEvent &e)
{
	// for timer
	publishCmdAction(buffered_action_);
}

void WorldSimulator::publishCmdAction(ACT_TYPE action)
{
	geometry_msgs::Twist cmd;
	cmd.linear.x = target_speed_;
	cmd.linear.y = real_speed_;
	cmd.linear.z = static_cast<PedPomdp*>(model_)->GetAcceleration(action);
	cmd.angular.z = steering_; // GetSteering returns radii value
	cout<<"[publishCmdAction] time stamp" << SolverPrior::nn_priors[0]->get_timestamp() <<endl;
    cout << "[publishCmdAction] target speed " << target_speed_ << " steering " << steering_ << endl;
	cmdPub_.publish(cmd);
}


void WorldSimulator::publishROSState()
{
	geometry_msgs::Point32 pnt;
	
	geometry_msgs::Pose pose;
	
	geometry_msgs::PoseStamped pose_stamped;
	pose_stamped.header.stamp=ros::Time::now();

	pose_stamped.header.frame_id=global_frame_id;

	pose_stamped.pose.position.x=(stateTracker->carpos.x+0.0);
	pose_stamped.pose.position.y=(stateTracker->carpos.y+0.0);
	pose_stamped.pose.orientation.w=1.0;
	car_pub.publish(pose_stamped);
	
	geometry_msgs::PoseArray pA;
	pA.header.stamp=ros::Time::now();

	pA.header.frame_id=global_frame_id;
	for(int i=0;i<stateTracker->ped_list.size();i++)
	{
		//GetCurrentState(ped_list[i]);
		pose.position.x=(stateTracker->ped_list[i].w+0.0);
		pose.position.y=(stateTracker->ped_list[i].h+0.0);
		pose.orientation.w=1.0;
		pA.poses.push_back(pose);
	}
	for(auto& veh : stateTracker->veh_list)
	{
		//GetCurrentState(ped_list[i]);
		pose.position.x=(veh.w+0.0);
		pose.position.y=(veh.h+0.0);
		pose.orientation.w=veh.heading_dir;
		pA.poses.push_back(pose);
	}

	pa_pub.publish(pA);

	uint32_t shape = visualization_msgs::Marker::CYLINDER;

	if (worldModel.goal_mode == "goal"){

	    visualization_msgs::MarkerArray markers;

		for(int i=0;i<worldModel.goals.size();i++)
		{
			visualization_msgs::Marker marker;

			marker.header.frame_id=ModelParams::rosns+"/map";
			marker.header.stamp=ros::Time::now();
			marker.ns="basic_shapes";
			marker.id=i;
			marker.type=shape;
			marker.action = visualization_msgs::Marker::ADD;

			marker.pose.position.x = worldModel.goals[i].x;
			marker.pose.position.y = worldModel.goals[i].y;
			marker.pose.position.z = 0;
			marker.pose.orientation.x = 0.0;
			marker.pose.orientation.y = 0.0;
			marker.pose.orientation.z = 0.0;
			marker.pose.orientation.w = 1.0;

			marker.scale.x = 1;
			marker.scale.y = 1;
			marker.scale.z = 1;
			marker.color.r = marker_colors[i][0];
			marker.color.g = marker_colors[i][1];
			marker.color.b = marker_colors[i][2];
			marker.color.a = 1.0;
			
			markers.markers.push_back(marker);
		}
		goal_pub.publish(markers);
	}
}

extern double marker_colors[20][3];
void WorldSimulator::publishAction(int action, double reward)
{
	if(action ==-1) return;
	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Prepare markers"<<endl;

	uint32_t shape = visualization_msgs::Marker::CUBE;
	visualization_msgs::Marker marker;
    
	
	marker.header.frame_id=global_frame_id;
	marker.header.stamp=ros::Time::now();
	marker.ns="basic_shapes";
	marker.id=0;
	marker.type=shape;
	marker.action = visualization_msgs::Marker::ADD;
	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Prepare markers check 1"<<endl;
	double px,py;
	px=stateTracker->carpos.x;
	py=stateTracker->carpos.y;
	marker.pose.position.x = px+1;
	marker.pose.position.y = py+1;
	marker.pose.position.z = 0;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;
	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Prepare markers check 2"<<endl;

	// Set the scale of the marker in meters
	marker.scale.x = 0.6;
	marker.scale.y = 3;
	marker.scale.z = 0.6;

	int aid = action;
	aid=max(aid, 0)%3;
	logd << "[WorldSimulator::"<<__FUNCTION__<<"] color id"<< aid << " action "<< action<<endl;

	marker.color.r = marker_colors[action_map[aid]][0];
    marker.color.g = marker_colors[action_map[aid]][1];
	marker.color.b = marker_colors[action_map[aid]][2];
	marker.color.a = 1.0;

	ros::Duration d(1/ModelParams::control_freq);
	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Publish marker"<<endl;
	marker.lifetime=d;
	actionPub_.publish(marker);
	logd << "[WorldSimulator::"<<__FUNCTION__<<"] Publish action comand"<<endl;
    geometry_msgs::Twist action_cmd;
    action_cmd.linear.x=action;
    action_cmd.linear.y=reward;
    actionPubPlot_.publish(action_cmd);
}


bool WorldSimulator::getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const
{
    out_pose.setIdentity();

	cout<<"laser frame "<<in_pose.frame_id_<<endl;
		
    cout << "getting transform of frame " << target_frame << endl;

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


//=============== from pedpomdpnode ==============

void WorldSimulator::speedCallback(nav_msgs::Odometry odo)
{
//	cout<<"update real speed "<<odo.twist.twist.linear.x<<endl;
	// real_speed_=odo.twist.twist.linear.x;
	real_speed_=sqrt(odo.twist.twist.linear.x * odo.twist.twist.linear.x + 
		odo.twist.twist.linear.y * odo.twist.twist.linear.y);
	if (real_speed_ > ModelParams::VEL_MAX*2.0){
		cerr << "ERROR: Unusual car vel (too large): " << real_speed_ << endl;
		raise(SIGABRT);
	}
}


bool sortFn(Pedestrian p1,Pedestrian p2)
{
	return p1.id<p2.id;
}

#ifdef CONNECTOR2
void agentArrayCallback(carla_connector2::TrafficAgentArray data){
#else 
void agentArrayCallback(carla_connector::agent_array data){
#endif

	double data_sec = data.header.stamp.sec;  // std_msgs::time
	double data_nsec = data.header.stamp.nsec;

	double data_time_sec = data_sec + data_nsec * 1e-9;

	WorldSimulator::stateTracker->latest_time_stamp = data_time_sec;

	DEBUG(string_sprintf("receive agent num %d at time %f \n", 
		data.agents.size(), data_time_sec));

	vector<Pedestrian> ped_list;
	vector<Vehicle> veh_list;
	
	#ifdef CONNECTOR2
	for (carla_connector2::TrafficAgent& agent : data.agents){
	#else
	for (carla_connector::traffic_agent& agent : data.agents){
	#endif

		#ifdef CONNECTOR2
		std::string agent_type = agent.type;
		#else
		std::string agent_type = agent.type.data;
		#endif

		cout << "get agent: " << agent.id <<" "<< agent_type << endl;

		if (agent_type == "car"){
			Vehicle world_veh;

			world_veh.time_stamp = data_time_sec;

			world_veh.id = agent.id;
			world_veh.w = agent.pose.position.x;
			world_veh.h = agent.pose.position.y;
			world_veh.heading_dir = navposeToHeadingDir(agent.pose);

			for (auto& corner : agent.bbox.points){
				world_veh.bb.emplace_back(corner.x, corner.y);	
			}
		
			world_veh.reset_intention = agent.reset_intention;

			for (auto& nav_path : agent.path_candidates){
				Path new_path;
				world_veh.paths.emplace_back(new_path);
				for (auto& pose : nav_path.poses){
					world_veh.paths.back().emplace_back(
						pose.pose.position.x, pose.pose.position.y);
				}
			}

			cout << "vehicle has "<< agent.path_candidates.size() << " path_candidates" << endl;
			veh_list.push_back(world_veh);

		} else if (agent_type == "ped"){
			Pedestrian world_ped;

			world_ped.time_stamp = data_time_sec;

			world_ped.id = agent.id;
			world_ped.w = agent.pose.position.x;
			world_ped.h = agent.pose.position.y;

			world_ped.reset_intention = agent.reset_intention;

			world_ped.cross_dir = agent.cross_dirs[0];
			for (auto& nav_path : agent.path_candidates){
				Path new_path;
				world_ped.paths.emplace_back(new_path);
				for (auto& pose : nav_path.poses){
					world_ped.paths.back().emplace_back(
						pose.pose.position.x, pose.pose.position.y);
				}
			}

			cout << "ped has "<< agent.path_candidates.size() << " path_candidates" << endl;
			
			ped_list.push_back(world_ped);
		}
		else{
			ERR(string_sprintf("Unsupported type %s", agent_type));
		}
	}

	for(auto& ped: ped_list)
	{
		WorldSimulator::stateTracker->updatePed(ped);
	}

	for(auto& veh: veh_list)
	{
		WorldSimulator::stateTracker->updateVeh(veh);
	}

	WorldSimulator::stateTracker->model.print_path_map();

	WorldSimulator::stateTracker->text(ped_list);
	WorldSimulator::stateTracker->text(veh_list);

	// DEBUG(string_sprintf("ped_list len %d", ped_list.size()));
	// DEBUG(string_sprintf("veh_list len %d", veh_list.size()));
    logd << "====================[ agentArrayCallback end ]=================" << endl;

	SimulatorBase::agents_data_ready = true;
}

void pedPoseCallback(ped_is_despot::ped_local_frame_vector lPedLocal)
{
    logd << "======================[ pedPoseCallback ]= ts "<<
    		Globals::ElapsedTime()<< " ==================" << endl;

  	DEBUG(string_sprintf("receive peds num %d", lPedLocal.ped_local.size()));


    if(lPedLocal.ped_local.size()==0) return;

	vector<Pedestrian> ped_list;
    for(int ii=0; ii< lPedLocal.ped_local.size(); ii++)
    {
		Pedestrian world_ped;
		ped_is_despot::ped_local_frame ped=lPedLocal.ped_local[ii];
		world_ped.id=ped.ped_id;
		world_ped.w = ped.ped_pose.x;
		world_ped.h = ped.ped_pose.y;
		ped_list.push_back(world_ped);
    }
	
	for(int i=0;i<ped_list.size();i++)
	{
		WorldSimulator::stateTracker->updatePed(ped_list[i]);
	}

    logd << "====================[ pedPoseCallback end ]=================" << endl;

	SimulatorBase::agents_data_ready = true;
}

void receive_map_callback(nav_msgs::OccupancyGrid map){
	logi << "[receive_map_callback] " << endl;

	for (int i=0;i< SolverPrior::nn_priors.size();i++){
		PedNeuralSolverPrior * nn_prior = static_cast<PedNeuralSolverPrior *>(SolverPrior::nn_priors[i]);
		nn_prior->raw_map_ = map;
		nn_prior->map_received = true;
		nn_prior->Init();
	}

	logi << "[receive_map_callback] end " << endl;
}

geometry_msgs::PoseStamped WorldSimulator::getPoseAhead(const tf::Stamped<tf::Pose>& carpose) {
    static int last_i = -1;
    static double last_yaw = 0;
    static COORD last_ahead;
	auto& path = worldModel.path;
	COORD coord = poseToCoord(carpose);

	int i = path.nearest(coord);
    COORD ahead;
    double yaw;

    if (i == last_i) {
        yaw = last_yaw;
        ahead = last_ahead;
    } else {
        int j = path.forward(i, pathplan_ahead_);
        //yaw = (path.getYaw(j) + path.getYaw(j-1) + path.getYaw(j-2)) / 3;
        yaw = path.getYaw(j);
        ahead = path[j];
        last_yaw = yaw;
		last_ahead = ahead;
        last_i = i;
    }

	auto q = tf::createQuaternionFromYaw(yaw);
	tf::Pose p(q, tf::Vector3(ahead.x, ahead.y, 0));
	tf::Stamped<tf::Pose> pose_ahead(p, carpose.stamp_, carpose.frame_id_);
	geometry_msgs::PoseStamped posemsg;
	tf::poseStampedTFToMsg(pose_ahead, posemsg);
	return posemsg;
}


void WorldSimulator::update_il_car(const ped_is_despot::car_info::ConstPtr car) {
    if (b_update_il == true){
    	p_IL_data.past_car = p_IL_data.cur_car;
    	p_IL_data.cur_car = *car;
    }
}

//void WorldSimulator::update_il_steering(const std_msgs::Float32::ConstPtr steer){
//	if (b_update_il == true){
//    	p_IL_data.action_reward.angular.x = float(steer->data);
//    	cout<< "receive steer "<<steer->data <<endl;
//    }
//}


void WorldSimulator::publishImitationData(PomdpStateWorld& planning_state, ACT_TYPE safeAction, float reward, float cmd_vel)
{

	// peds for publish
	ped_is_despot::peds_info p_ped;
	// only publish information for N_PED_IN peds for imitation learning
	for (int i = 0; i < ModelParams::N_PED_IN; i++){
		ped_is_despot::ped_info ped;
        ped.ped_id = planning_state.agents[i].id;
        ped.ped_goal_id = planning_state.agents[i].intention;
        ped.ped_speed = 1.2;
        ped.ped_pos.x = planning_state.agents[i].pos.x;
        ped.ped_pos.y = planning_state.agents[i].pos.y;
        ped.ped_pos.z = 0;
        p_ped.peds.push_back(ped);
    }

    p_IL_data.past_peds = p_IL_data.cur_peds;
    p_IL_data.cur_peds = p_ped;

	// ped belief for pushlish
	int i=0;
	ped_is_despot::peds_believes pbs;	
	for(auto & kv: beliefTracker->agent_beliefs)
	{
		ped_is_despot::ped_belief pb;
		AgentBelief belief = kv.second;
		pb.ped_x=belief.pos.x;
		pb.ped_y=belief.pos.y;
		pb.ped_id=belief.id;
		for(auto & goal_probs : belief.prob_modes_goals)
			for (auto v: goal_probs)
				pb.belief_value.push_back(v);
		pbs.believes.push_back(pb);
	}
	pbs.cmd_vel=stateTracker->carvel;
	pbs.robotx=stateTracker->carpos.x;
	pbs.roboty=stateTracker->carpos.y;

	p_IL_data.believes = pbs.believes;


	// action for publish
	geometry_msgs::Twist p_action_reward;

    p_IL_data.action_reward.linear.y=reward;
    p_IL_data.action_reward.linear.z=cmd_vel;

	p_IL_data.action_reward.linear.x = static_cast<PedPomdp*>(model_)->GetAccelerationID(safeAction);
	p_IL_data.action_reward.angular.x = static_cast<PedPomdp*>(model_)->GetSteering(safeAction);

    IL_pub.publish(p_IL_data);

}

void WorldSimulator::Debug_action(){
//	if (SolverPrior::nn_priors[0]->default_action != SolverPrior::nn_priors[0]->searched_action){
//		cerr << "ERROR: Default prior action modified!!!!" << endl;
//		raise(SIGABRT);
//	}
//
//	if (buffered_action_ != SolverPrior::nn_priors[0]->searched_action){
//		cerr << "ERROR: Searched action modified!!!!" << endl;
//		raise(SIGABRT);
//	}
}

void WorldSimulator::setCarGoal(COORD car_goal){
	worldModel.car_goal=car_goal;
}
