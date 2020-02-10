#include "world_simulator.h"
#include "world_model.h"
#include "context_pomdp.h"

#include <std_msgs/Int32.h>

#include <msg_builder/StartGoal.h>
#include <msg_builder/car_info.h>
#include <msg_builder/peds_info.h>
#include <msg_builder/ped_info.h>
#include <msg_builder/peds_believes.h>
#include "ros/ros.h"

#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Bool.h>
#include <msg_builder/TrafficAgentArray.h>
#include <msg_builder/AgentPathArray.h>
#include <msg_builder/Lanes.h>
#include <msg_builder/Obstacles.h>
#include <msg_builder/PomdpCmd.h>

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

WorldModel SimulatorBase::world_model;
bool SimulatorBase::agents_data_ready = false;
bool SimulatorBase::agents_path_data_ready = false;

double pub_frequency = 9.0;

double navposeToHeadingDir(const geometry_msgs::Pose & msg) {
	/* get yaw angle [-pi, pi)*/
	tf::Pose pose;
	tf::poseMsgToTF(msg, pose);

	double yaw;
	yaw = tf::getYaw(pose.getRotation());
	if (yaw < 0)
		yaw += 2 * 3.1415926;
	return yaw;
}

WorldSimulator::WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model,
		unsigned seed, std::string map_location, int summit_port) :
		SimulatorBase(_nh), worldModel(SimulatorBase::world_model), model_(
				model), last_acc_(-1), goal_reached_(false), paths_time_stamp_(
				-1), car_time_stamp_(0), agents_time_stamp_(0), safe_action_(0), time_scale(
				1.0), World() {

	map_location_ = map_location;
	summit_port_ = summit_port;

	worldModel.InitGamma();
}

WorldSimulator::~WorldSimulator() {
	msg_builder::PomdpCmd cmd;
	cmd.target_speed = 0.0;
	cmd.cur_speed = real_speed;
	cmd.acc = -ModelParams::ACC_SPEED;
	cmd.steer = 0;
	cmdPub_.publish(cmd);
}

/**
 * [Essential]
 * Establish connection to simulator or system
 */
bool WorldSimulator::Connect() {
	cerr << "DEBUG: Connecting with world" << endl;

	cmdPub_ = nh.advertise<msg_builder::PomdpCmd>("cmd_action_pomdp", 1);

	ego_sub_ = nh.subscribe("ego_state", 1, &WorldSimulator::EgoStateCallBack,
			this);
	ego_dead_sub_ = nh.subscribe("ego_dead", 1,
			&WorldSimulator::EgoDeadCallBack, this);
	pathSub_ = nh.subscribe("plan", 1, &WorldSimulator::RetrievePathCallBack,
			this);

	agent_sub_ = nh.subscribe("agent_array", 1,
			&WorldSimulator::AgentArrayCallback, this);
	agent_path_sub_ = nh.subscribe("agent_path_array", 1,
			&WorldSimulator::AgentPathArrayCallback, this);
	logi << "Subscribers and Publishers created at the "
			<< Globals::ElapsedTime() << "th second" << endl;

	auto path_data = ros::topic::waitForMessage<nav_msgs::Path>("plan",
			ros::Duration(300));
	logi << "plan get at the " << Globals::ElapsedTime() << "th second" << endl;

	ros::spinOnce();

	auto start = Time::now();
	bool agent_data_ok = false, car_data_ok = false, agent_flag_ok = false;
	while (Globals::ElapsedTime(start) < 30.0) {
		auto agent_data = ros::topic::waitForMessage<
				msg_builder::TrafficAgentArray>("agent_array",
				ros::Duration(1));
		if (agent_data && !agent_data_ok) {
			logi << "agent_array get at the " << Globals::ElapsedTime()
					<< "th second" << endl;
			agent_data_ok = true;
		}
		ros::spinOnce();

		auto car_data = ros::topic::waitForMessage<msg_builder::car_info>(
				"ego_state", ros::Duration(1));
		if (car_data && !car_data_ok) {
			logi << "ego_state get at the " << Globals::ElapsedTime()
					<< "th second" << endl;
			car_data_ok = true;
		}
		ros::spinOnce();

		auto agent_ready_bool = ros::topic::waitForMessage<std_msgs::Bool>(
				"agents_ready", ros::Duration(1));
		if (agent_ready_bool && !agent_flag_ok) {
			logi << "agents ready at get at the " << Globals::ElapsedTime()
					<< "th second" << endl;
			agent_flag_ok = true;
		}
		ros::spinOnce();
	}

	if (Globals::ElapsedTime(start) >= 30.0)
		ERR("No agent array messages received after 30 seconds.");

	return true;
}

/**
 * [Essential]
 * Initialize or reset the (simulation) environment, return the start state if applicable
 */
State* WorldSimulator::Initialize() {

	cerr << "DEBUG: Initializing world in WorldSimulator::Initialize" << endl;

	safe_action_ = 2;
	cmd_speed_ = 0.0;
	cmd_steer_ = 0.0;
	goal_reached_ = false;
	last_acc_ = 0;

	if (SolverPrior::nn_priors.size() == 0) {
		ERR("No nn_prior exist");
	}

	return NULL;
}

/**
 * [Optional]
 * To help construct initial belief to print debug informations in Logger
 */
State* WorldSimulator::GetCurrentState() {

	if (path_from_topic_.size() == 0) {
		logi << "[GetCurrentState] path topic not ready yet..." << endl;
		return NULL;
	}

	PomdpStateWorld state;
	current_state_.car = car_;
	int n = 0;
	for (std::map<int, AgentStruct>::iterator it = exo_agents_.begin();
			it != exo_agents_.end(); ++it) {
		if (n < ModelParams::N_PED_WORLD)
			current_state_.agents[n] = it->second;
		n++;
	}
	current_state_.num = n;
	current_state_.time_stamp = min(car_time_stamp_, agents_time_stamp_);

	if (logging::level() >= logging::DEBUG) {
		logi << "current world state:" << endl;
		static_cast<ContextPomdp*>(model_)->PrintWorldState(current_state_);
	}
	logi << " current state time stamp " << current_state_.time_stamp << endl;

	return static_cast<State*>(&current_state_);
}

double WorldSimulator::StepReward(PomdpStateWorld& state, ACT_TYPE action) {
	double reward = 0.0;

	if (worldModel.IsGlobalGoal(state.car)) {
		reward = ModelParams::GOAL_REWARD;
		return reward;
	}

	ContextPomdp* ContextPomdp_model = static_cast<ContextPomdp*>(model_);

	if (state.car.vel > 0.001 && worldModel.InRealCollision(state, 120.0)) { /// collision occurs only when car is moving
		reward = ContextPomdp_model->CrashPenalty(state);
		return reward;
	}
	// Smoothness control
	reward += ContextPomdp_model->ActionPenalty(action);

	// Speed control: Encourage higher speed
	reward += ContextPomdp_model->MovementPenalty(state,
			ContextPomdp_model->GetSteering(action));
	return reward;
}

bool WorldSimulator::Emergency() {
	double mindist = numeric_limits<double>::infinity();
	for (std::map<int, AgentStruct>::iterator it = exo_agents_.begin();
			it != exo_agents_.end(); ++it) {
		AgentStruct& agent = it->second;
		double d = COORD::EuclideanDistance(car_.pos, agent.pos);
		if (d < mindist)
			mindist = d;
	}
	cout << "Emergency mindist = " << mindist << endl;
	return (mindist < 1.5);
}

/**
 * [Essential]
 * send action, receive reward, obs, and terminal
 * @param action Action to be executed in the real-world system
 * @param obs    Observation sent back from the real-world system
 */
bool WorldSimulator::ExecuteAction(ACT_TYPE action, OBS_TYPE& obs) {

	if (action == -1)
		return false;
	logi << "ExecuteAction at the " << Globals::ElapsedTime() << "th second"
			<< endl;
	ros::spinOnce();

	/* Update state */
	PomdpStateWorld* curr_state =
			static_cast<PomdpStateWorld*>(GetCurrentState());

	if (logging::level() >= logging::INFO) {
		logi << "Executing action for state:" << endl;
		static_cast<ContextPomdp*>(model_)->PrintWorldState(*curr_state);
	}

	double acc, steer;

	/* Reach goal: no more action sent */
	if (worldModel.IsGlobalGoal(curr_state->car)) {
		cout
				<< "--------------------------- goal reached ----------------------------"
				<< endl;
		cout << "" << endl;
		acc = static_cast<ContextPomdp*>(model_)->GetAccfromAccID(
				ContextPomdp::ACT_DEC);
		steer = 0;
		action = static_cast<ContextPomdp*>(model_)->GetActionID(steer, acc);
		goal_reached_ = true;
	}

	int collision_peds_id;
	if (curr_state->car.vel > 0.5 * time_scale
			&& worldModel.InRealCollision(*curr_state, collision_peds_id,
					120.0)) {
		cout
				<< "--------------------------- collision = 1 ----------------------------"
				<< endl;
		cout << "collision ped: " << collision_peds_id << endl;

		acc = static_cast<ContextPomdp*>(model_)->GetAccfromAccID(
				ContextPomdp::ACT_DEC);
		steer = 0;
		action = static_cast<ContextPomdp*>(model_)->GetActionID(steer, acc);

		ERR("Termination of episode due to coll.");
	}

	logv << "[WorldSimulator::" << __FUNCTION__
			<< "] Update steering and target speed" << endl;

	if (goal_reached_ == true) {
		cmd_steer_ = 0;
		cmd_speed_ = real_speed;

		cmd_speed_ -= ModelParams::ACC_SPEED;
		if (cmd_speed_ <= 0.0)
			cmd_speed_ = 0.0;

		buffered_action_ = static_cast<ContextPomdp*>(model_)->GetActionID(0.0,
				-ModelParams::ACC_SPEED);

		if (real_speed <= 0.01 * time_scale) {
			cout << "After reaching goal: real_spead is already zero" << endl;
			ros::shutdown();
		}
	} else if (Emergency()) {
		cmd_steer_ = 0;
		cmd_speed_ = -1;

		buffered_action_ = static_cast<ContextPomdp*>(model_)->GetActionID(0.0,
				-ModelParams::ACC_SPEED);
		cout
				<< "--------------------------- emergency ----------------------------"
				<< endl;
	} else if (worldModel.path.size() > 0
			&& COORD::EuclideanDistance(car_.pos, worldModel.path[0]) > 4.0) {
		cerr
				<< "=================== Path offset too high !!! Node shutting down"
				<< endl;
		ros::shutdown();
	} else {
		buffered_action_ = action;
	}

	// Updating cmd steering and speed. They will be send to vel_pulisher by timer
	UpdateCmds(buffered_action_, false);

	PublishCmdAction(buffered_action_);

	double step_reward = StepReward(*curr_state, buffered_action_);
	cout << "action **= " << buffered_action_ << endl;
	cout << "reward **= " << step_reward << endl;
	safe_action_ = action;

	/* Receive observation.
	 * Caution: obs info is not up-to-date here. Don't use it
	 */

	logv << "[WorldSimulator::" << __FUNCTION__ << "] Generate obs" << endl;
	obs = static_cast<ContextPomdp*>(model_)->StateToIndex(GetCurrentState());

	return goal_reached_;
}

void WorldSimulator::UpdateCmds(ACT_TYPE action, bool buffered) {
	if (logging::level() >= logging::INFO)
		worldModel.path.Text();

	logi << "[update_cmds_naive] Buffering action " << action << endl;

	float acc = static_cast<ContextPomdp*>(model_)->GetAcceleration(action);
	double speed_step = ModelParams::ACC_SPEED / ModelParams::CONTROL_FREQ;

	cmd_speed_ = real_speed;
	if (acc > 0.0)
		cmd_speed_ = real_speed + speed_step;
	else if (acc < 0.0)
		cmd_speed_ = real_speed - speed_step;
	cmd_speed_ = max(min(cmd_speed_, ModelParams::VEL_MAX), 0.0);

	cmd_steer_ = static_cast<ContextPomdp*>(model_)->GetSteering(action);

	logi << "Executing action:" << action << " steer/acc = " << cmd_steer_
			<< "/" << acc << endl;
}

void WorldSimulator::PublishCmdAction(const ros::TimerEvent &e) {
	// for timer
	PublishCmdAction(buffered_action_);
}

void WorldSimulator::PublishCmdAction(ACT_TYPE action) {
	msg_builder::PomdpCmd cmd;
	cmd.target_speed = cmd_speed_;
	cmd.cur_speed = real_speed;
	cmd.acc = static_cast<ContextPomdp*>(model_)->GetAcceleration(action);
	cmd.steer = cmd_steer_; // GetSteering returns radii value
	cout << "[PublishCmdAction] time stamp" << Globals::ElapsedTime() << endl;
	cout << "[PublishCmdAction] target speed " << cmd_speed_ << " steering "
			<< cmd_steer_ << endl;
	cmdPub_.publish(cmd);
}

void WorldSimulator::RetrievePathCallBack(const nav_msgs::Path::ConstPtr path) {

	logi << "receive path from navfn " << path->poses.size() << " at the "
			<< Globals::ElapsedTime() << "th second" << endl;

	if (path->poses.size() == 0) {
		DEBUG("Path missing from topic");
		return;
	}

	Path p;
	for (int i = 0; i < path->poses.size(); i++) {
		COORD coord;
		coord.x = path->poses[i].pose.position.x;
		coord.y = path->poses[i].pose.position.y;
		p.push_back(coord);
	}
	if (p.GetLength() < 3)
		ERR("Path length shorter than 3 meters.");
	path_from_topic_ = p.Interpolate();

	world_model.SetPath(path_from_topic_);
}

void CalBBExtents(COORD pos, double heading_dir, vector<COORD>& bb,
		double& extent_x, double& extent_y) {
	COORD forward_vec = COORD(cos(heading_dir), sin(heading_dir));
	COORD sideward_vec = COORD(-sin(heading_dir), cos(heading_dir));

	for (auto& point : bb) {
		extent_x = max((point - pos).dot(sideward_vec), extent_x);
		extent_y = max((point - pos).dot(forward_vec), extent_y);
	}
}

void CalBBExtents(AgentStruct& agent, std::vector<COORD>& bb,
		double heading_dir) {
	if (agent.type == AgentType::ped) {
		agent.bb_extent_x = 0.3;
		agent.bb_extent_y = 0.3;
	} else {
		agent.bb_extent_x = 0.0;
		agent.bb_extent_y = 0.0;
	}
	CalBBExtents(agent.pos, heading_dir, bb, agent.bb_extent_x,
			agent.bb_extent_y);
}

void WorldSimulator::AgentArrayCallback(msg_builder::TrafficAgentArray data) {
	double data_sec = data.header.stamp.sec;  // std_msgs::time
	double data_nsec = data.header.stamp.nsec;
	double data_time_sec = data_sec + data_nsec * 1e-9;
	agents_time_stamp_ = data_time_sec;
	DEBUG(
			string_sprintf("receive %d agents at time %f", data.agents.size(),
					Globals::ElapsedTime()));

	exo_agents_.clear();
	for (msg_builder::TrafficAgent& agent : data.agents) {
		std::string agent_type = agent.type;
		int id = agent.id;
		exo_agents_[id] = AgentStruct();
		exo_agents_[id].id = id;
		if (agent_type == "car")
			exo_agents_[id].type = AgentType::car;
		else if (agent_type == "bike")
			exo_agents_[id].type = AgentType::car;
		else if (agent_type == "ped")
			exo_agents_[id].type = AgentType::ped;
		else
			ERR(string_sprintf("Unsupported type %s", agent_type));

		exo_agents_[id].pos = COORD(agent.pose.position.x,
				agent.pose.position.y);
		exo_agents_[id].vel = COORD(agent.vel.x, agent.vel.y);
		exo_agents_[id].heading_dir = navposeToHeadingDir(agent.pose);

		std::vector<COORD> bb;
		for (auto& corner : agent.bbox.points) {
			bb.emplace_back(corner.x, corner.y);
		}
		CalBBExtents(exo_agents_[id], bb, exo_agents_[id].heading_dir);
	}

	if (logging::level() >= logging::DEBUG)
		worldModel.PrintPathMap();

	SimulatorBase::agents_data_ready = true;
}

void WorldSimulator::AgentPathArrayCallback(msg_builder::AgentPathArray data) {
	double data_sec = data.header.stamp.sec;  // std_msgs::time
	double data_nsec = data.header.stamp.nsec;
	double data_time_sec = data_sec + data_nsec * 1e-9;
	paths_time_stamp_ = data_time_sec;
	DEBUG(
			string_sprintf("receive %d agent paths at time %f",
					data.agents.size(), Globals::ElapsedTime()));

	worldModel.id_map_belief_reset.clear();
	worldModel.id_map_paths.clear();
	worldModel.id_map_num_paths.clear();

	for (msg_builder::AgentPaths& agent : data.agents) {
		std::string agent_type = agent.type;
		int id = agent.id;

		auto it = exo_agents_.find(id);
		if (it != exo_agents_.end()) {
			exo_agents_[id].cross_dir = agent.cross_dirs[0];

			worldModel.id_map_belief_reset[id] = agent.reset_intention;
			worldModel.id_map_num_paths[id] = agent.path_candidates.size();
			worldModel.id_map_paths[id] = std::vector<Path>();
			for (auto& nav_path : agent.path_candidates) {
				Path new_path;
				for (auto& pose : nav_path.poses) {
					new_path.emplace_back(pose.pose.position.x,
							pose.pose.position.y);
				}
				worldModel.id_map_paths[id].emplace_back(new_path);
			}
		}
	}

	SimulatorBase::agents_path_data_ready = true;
}

double xylength(geometry_msgs::Point32 p) {
	return sqrt(p.x * p.x + p.y * p.y);
}
bool car_data_ready = false;

void WorldSimulator::EgoDeadCallBack(const std_msgs::Bool ego_dead) {
	ERR("Ego vehicle killed in ego_vehicle.py");
}

void WorldSimulator::EgoStateCallBack(
		const msg_builder::car_info::ConstPtr car) {
	const msg_builder::car_info& ego_car = *car;
	car_.pos = COORD(ego_car.car_pos.x, ego_car.car_pos.y);
	car_.heading_dir = ego_car.car_steer;
	car_.vel = ego_car.car_speed;

	real_speed = COORD(ego_car.car_vel.x, ego_car.car_vel.y).Length();
	if (real_speed > ModelParams::VEL_MAX * 1.3) {
		ERR(
				string_sprintf(
						"Unusual car vel (too large): %f. Check the speed controller for possible problems (VelPublisher.cpp)",
						real_speed));
	}

	ModelParams::CAR_FRONT = COORD(
			ego_car.front_axle_center.x - ego_car.car_pos.x,
			ego_car.front_axle_center.y - ego_car.car_pos.y).Length();
	ModelParams::CAR_REAR = COORD(
			ego_car.rear_axle_center.y - ego_car.car_pos.y,
			ego_car.rear_axle_center.y - ego_car.car_pos.y).Length();
	ModelParams::CAR_WHEEL_DIST = ModelParams::CAR_FRONT
			+ ModelParams::CAR_REAR;

	ModelParams::MAX_STEER_ANGLE = ego_car.max_steer_angle / 180.0 * M_PI;

	ModelParams::CAR_WIDTH = 0;
	ModelParams::CAR_LENGTH = 0;

	double car_yaw = car_.heading_dir;
	COORD tan_dir(-sin(car_yaw), cos(car_yaw));
	COORD along_dir(cos(car_yaw), sin(car_yaw));
	for (auto& point : ego_car.car_bbox.points) {
		COORD p(point.x - ego_car.car_pos.x, point.y - ego_car.car_pos.y);
		double proj = p.dot(tan_dir);
		ModelParams::CAR_WIDTH = max(ModelParams::CAR_WIDTH, fabs(proj));
		proj = p.dot(along_dir);
		ModelParams::CAR_LENGTH = max(ModelParams::CAR_LENGTH, fabs(proj));
	}
	ModelParams::CAR_WIDTH = ModelParams::CAR_WIDTH * 2;
	ModelParams::CAR_LENGTH = ModelParams::CAR_LENGTH * 2;
	ModelParams::CAR_FRONT = ModelParams::CAR_LENGTH / 2.0;

	car_data_ready = true;
}
