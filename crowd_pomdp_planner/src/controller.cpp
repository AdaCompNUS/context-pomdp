#include "controller.h"
#include "core/node.h"
#include "core/solver.h"
#include "core/globals.h"
#include <csignal>
#include <time.h>
#include "boost/bind.hpp"
#include "world_simulator.h"
#include "custom_particle_belief.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

using namespace std;
using namespace despot;

int Controller::b_drive_mode = 0;
int Controller::gpu_id = 0;
int Controller::summit_port = 2000;
float Controller::time_scale = 1.0;

std::string Controller::map_location = "";
bool path_missing = true;

static DSPOMDP* ped_pomdp_model;
static ACT_TYPE action = (ACT_TYPE) (-1);
static OBS_TYPE obs = (OBS_TYPE) (-1);

bool predict_peds = true;

struct my_sig_action {
	typedef void (*handler_type)(int, siginfo_t*, void*);

	explicit my_sig_action(handler_type handler) {
		memset(&_sa, 0, sizeof(struct sigaction));
		_sa.sa_sigaction = handler;
		_sa.sa_flags = SA_SIGINFO;
	}

	operator struct sigaction const*() const {
		return &_sa;
	}
protected:
	struct sigaction _sa;
};

struct div_0_exception {
};

void handle_div_0(int sig, siginfo_t* info, void*) {
	switch (info->si_code) {
	case FPE_INTDIV:
		cout << "Integer divide by zero." << endl;
		break;
	case FPE_INTOVF:
		cout << "Integer overflow. " << endl;
		break;
	case FPE_FLTUND:
		cout << "Floating-point underflow. " << endl;
		break;
	case FPE_FLTRES:
		cout << "Floating-point inexact result. " << endl;
		break;
	case FPE_FLTINV:
		cout << "Floating-point invalid operation. " << endl;
		break;
	case FPE_FLTSUB:
		cout << "Subscript out of range. " << endl;
		break;
	case FPE_FLTDIV:
		cout << "Floating-point divide by zero. " << endl;
		break;
	case FPE_FLTOVF:
		cout << "Floating-point overflow. " << endl;
		break;
	};
	exit(-1);
}

Controller::Controller(ros::NodeHandle& _nh, bool fixed_path) :
		nh_(_nh), fixed_path_(fixed_path), last_action_(-1), last_obs_(
				-1), model_(NULL), prior_(NULL), ped_belief_(NULL), summit_driving_simulator_(
				NULL) {
	my_sig_action sa(handle_div_0);
	if (0 != sigaction(SIGFPE, sa, NULL)) {
		std::cerr << "!!!!!!!! fail to setup segfault handler !!!!!!!!"
				<< std::endl;
	}

	global_frame_id_ = ModelParams::ROS_NS + "/map";
	control_freq_ = ModelParams::CONTROL_FREQ;

	cerr << "DEBUG: Initializing publishers..." << endl;
	pathPub_ = nh_.advertise<nav_msgs::Path>("pomdp_path_repub", 1, true); // for visualization
	pathSub_ = nh_.subscribe("plan", 1, &Controller::RetrievePathCallBack,
			this); // receive path from path planner

	logi << " Controller constructed at the " << Globals::ElapsedTime()
			<< "th second" << endl;
}

DSPOMDP* Controller::InitializeModel(option::Option* options) {
	cerr << "DEBUG: Initializing model" << endl;
	DSPOMDP* model = new PedPomdp();
	static_cast<PedPomdp*>(model)->world_model = &SimulatorBase::worldModel;
	model_ = model;
	ped_pomdp_model = model;

	return model;
}

void Controller::CreateDefaultPriors(DSPOMDP* model) {
	logv << "DEBUG: Creating solver prior " << endl;

	if (Globals::config.use_multi_thread_) {
		SolverPrior::nn_priors.resize(Globals::config.NUM_THREADS);
	} else
		SolverPrior::nn_priors.resize(1);

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		logv << "DEBUG: Creating prior " << i << endl;
		SolverPrior::nn_priors[i] =
				static_cast<PedPomdp*>(model)->CreateSolverPrior(
						summit_driving_simulator_, "DEFAULT", false);
		SolverPrior::nn_priors[i]->prior_id(i);
	}

	prior_ = SolverPrior::nn_priors[0];
	logv << "DEBUG: Created solver prior " << typeid(*prior_).name() << "at ts "
			<< Globals::ElapsedTime() << endl;
}

World* Controller::InitializeWorld(std::string& world_type, DSPOMDP* model,
		option::Option* options) {
	cerr << "DEBUG: Initializing world" << endl;

	World* world = new WorldSimulator(nh_, static_cast<DSPOMDP*>(model),
			Globals::config.root_seed, map_location, summit_port);
	logi << "WorldSimulator constructed at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	if (Globals::config.useGPU) {
		model->InitGPUModel();
		logi << "InitGPUModel finished at the " << Globals::ElapsedTime()
				<< "th second" << endl;
	}

	static_cast<PedPomdp*>(model)->world_model = &(WorldSimulator::worldModel);
	summit_driving_simulator_ = static_cast<WorldSimulator*>(world);
	summit_driving_simulator_->time_scale = time_scale;

	world->Connect();
	logi << "Connect finished at the " << Globals::ElapsedTime() << "th second"
			<< endl;

	CreateDefaultPriors(model);
	logi << "CreateNNPriors finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	world->Initialize();
	logi << "Initialize finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	return world;
}

void Controller::InitializeDefaultParameters() {
	cerr << "DEBUG: Initializing parameters" << endl;
	Globals::config.root_seed = time(NULL);
	Globals::config.time_per_move = (1.0 / ModelParams::CONTROL_FREQ) * 0.9
			/ time_scale;
	Globals::config.time_scale = time_scale;
	Globals::config.sim_len = 600;
	Globals::config.xi = 0.97;
	Globals::config.GPUid = gpu_id;
	Globals::config.use_multi_thread_ = true;
	Globals::config.exploration_mode = UCT;
	Globals::config.exploration_constant_o = 1.0;
	Globals::config.experiment_mode = true;

	Obs_type = OBS_INT_ARRAY;
	DESPOT::num_Obs_element_in_GPU = 1 + ModelParams::N_PED_IN * 2 + 3;

	if (b_drive_mode == JOINT_POMDP || b_drive_mode == ROLL_OUT) {
		Globals::config.useGPU = false;
		Globals::config.num_scenarios = 5;
		Globals::config.NUM_THREADS = 10;
		Globals::config.discount = 0.95;
		Globals::config.search_depth = 20;
		Globals::config.max_policy_sim_len = 20;
		if (b_drive_mode == JOINT_POMDP)
			Globals::config.pruning_constant = 0.001;
		else if (b_drive_mode == ROLL_OUT)
			Globals::config.pruning_constant = 100000000.0;
		Globals::config.exploration_constant = 0.1;
		Globals::config.silence = true;
	} else
		ERR("Unsupported drive mode");

	logging::level(3);

	logi << "Planner default parameters:" << endl;
	Globals::config.text();
}

std::string Controller::ChooseSolver() {
	return "DESPOT";
}

Controller::~Controller() {

}

void Controller::RetrievePathCallBack(const nav_msgs::Path::ConstPtr path) {

	logi << "receive path from navfn " << path->poses.size() << " at the "
			<< Globals::ElapsedTime() << "th second" << endl;

	if (fixed_path_ && path_from_topic_.size() > 0)
		return;

	if (path->poses.size() == 0) {
		path_missing = true;
		DEBUG("Path missing from topic");
		return;
	} else
		path_missing = false;

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
	WorldSimulator::worldModel.SetPath(path_from_topic_);
	PublishPath(path->header.frame_id, path_from_topic_);
}

void Controller::PublishPath(const string& frame_id, const Path& path) {
	nav_msgs::Path navpath;
	ros::Time plan_time = ros::Time::now();

	navpath.header.frame_id = frame_id;
	navpath.header.stamp = plan_time;

	for (const auto& s : path) {
		geometry_msgs::PoseStamped pose;
		pose.header.stamp = plan_time;
		pose.header.frame_id = frame_id;
		pose.pose.position.x = s.x;
		pose.pose.position.y = s.y;
		pose.pose.position.z = 0.0;
		pose.pose.orientation.x = 0.0;
		pose.pose.orientation.y = 0.0;
		pose.pose.orientation.z = 0.0;
		pose.pose.orientation.w = 1.0;
		navpath.poses.push_back(pose);
	}

	pathPub_.publish(navpath);
}

bool Controller::GetEgoPosFromSummit() {
	logi << "[GetEgoPosFromSummit] Getting car pos from summit..." << endl;

	tf::Stamped<tf::Pose> in_pose, out_pose;
	in_pose.setIdentity();
	in_pose.frame_id_ = ModelParams::ROS_NS + "/base_link";
	assert(summit_driving_simulator_);
	logv << "global_frame_id_: " << global_frame_id_ << " " << endl;
	if (!summit_driving_simulator_->GetObjectPose(global_frame_id_, in_pose,
			out_pose)) {
		cerr << "transform error within Controller::RunStep" << endl;
		logv << "laser frame " << in_pose.frame_id_ << endl;
		ros::Rate err_retry_rate(10);
		err_retry_rate.sleep();
		return false; // skip the current step
	}

	return true && SimulatorBase::agents_data_ready;
}

void Controller::PredictPedsForSearch(State* search_state) {
	if (predict_peds) {
		// predict state using last action
		if (last_action_ < 0 || last_action_ > model_->NumActions()) {
			logi << "Skip state prediction for the initial step, last action=" << last_action_ << endl;
			return;
		} else {

			summit_driving_simulator_->beliefTracker->cur_acc =
					static_cast<const PedPomdp*>(ped_pomdp_model)->GetAcceleration(
							last_action_);
			summit_driving_simulator_->beliefTracker->cur_steering =
					static_cast<const PedPomdp*>(ped_pomdp_model)->GetSteering(
							last_action_);

			cerr << "DEBUG: Prediction with last action:" << last_action_
					<< " steer/acc = "
					<< summit_driving_simulator_->beliefTracker->cur_steering
					<< "/" << summit_driving_simulator_->beliefTracker->cur_acc
					<< endl;

			auto predicted =
					summit_driving_simulator_->beliefTracker->PredictPedsCurVel(
							static_cast<PomdpState*>(search_state),
							summit_driving_simulator_->beliefTracker->cur_acc,
							summit_driving_simulator_->beliefTracker->cur_steering);

			PomdpState* predicted_state =
					static_cast<PomdpState*>(static_cast<const PedPomdp*>(ped_pomdp_model)->Copy(
							&predicted));

			static_cast<const PedPomdp*>(ped_pomdp_model)->PrintStateAgents(
					*predicted_state, string("predicted_agents"));

			for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
				SolverPrior::nn_priors[i]->Add_in_search(-1, predicted_state);

				logv << __FUNCTION__ << " add predicted search state of ts "
						<< predicted_state->time_stamp
						<< " predicted from search state of ts "
						<< static_cast<PomdpState*>(search_state)->time_stamp
						<< " hist len " << SolverPrior::nn_priors[i]->Size(true)
						<< endl;
			}
		}
	}
}

void Controller::UpdatePriors(const State* cur_state, State* search_state) {
	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		// make sure the history has not corrupted
		SolverPrior::nn_priors[i]->CompareHistoryWithRecorded();
		SolverPrior::nn_priors[i]->Add(last_action_, cur_state);
		SolverPrior::nn_priors[i]->Add_in_search(-1, search_state);

		logv << __FUNCTION__ << " add history search state of ts "
				<< static_cast<PomdpState*>(search_state)->time_stamp
				<< " hist len " << SolverPrior::nn_priors[i]->Size(true)
				<< endl;

		if (SolverPrior::nn_priors[i]->Size(true) == 10)
			Record_debug_state(search_state);

		SolverPrior::nn_priors[i]->RecordCurHistory();
	}
	logi << "history len = " << SolverPrior::nn_priors[0]->Size(false) << endl;
	logi << "history_in_search len = " << SolverPrior::nn_priors[0]->Size(true)
			<< endl;
}

bool Controller::RunStep(despot::Solver* solver, World* world, Logger* logger) {

	cerr << "DEBUG: Running step" << endl;

	logger->CheckTargetTime();

	double step_start_t = get_time_second();

	bool summit_ready = GetEgoPosFromSummit();
	if (!summit_ready)
		return false;

	if (path_from_topic_.size() == 0) {
		logi << "[RunStep] path topic not ready yet..." << endl;
		return false;
	}

	cerr << "DEBUG: Updating belief" << endl;

	double start_t = get_time_second();
	const State* cur_state = world->GetCurrentState();
	if(cur_state == NULL)
		ERR("cur_state == NULL");

	cout << "current state address" << cur_state << endl;

	State* search_state =
			static_cast<const PedPomdp*>(ped_pomdp_model)->CopyForSearch(
					cur_state);

	static_cast<PedPomdpBelief*>(solver->belief())->DeepUpdate(
			SolverPrior::nn_priors[0]->history_states(),
			SolverPrior::nn_priors[0]->history_states_for_search(), cur_state,
			search_state, last_action_);

	UpdatePriors(cur_state, search_state);

	double end_t = get_time_second();
	double update_time = (end_t - start_t);
	logi << "[RunStep] Time spent in Update(): " << update_time << endl;

	summit_driving_simulator_->PublishROSState();
	summit_driving_simulator_->beliefTracker->Text();

	int cur_search_hist_len = 0;
	cur_search_hist_len = SolverPrior::nn_priors[0]->Size(true);

	PredictPedsForSearch(search_state);

	start_t = get_time_second();
	ACT_TYPE action =
			static_cast<const PedPomdp*>(ped_pomdp_model)->GetActionID(0.0,
					0.0);
	double step_reward;
	if (b_drive_mode == NO || b_drive_mode == JOINT_POMDP
			|| b_drive_mode == ROLL_OUT) {
		cerr << "DEBUG: Search for action using " << typeid(*solver).name()
				<< endl;
		static_cast<PedPomdpBelief*>(solver->belief())->ResampleParticles(
				static_cast<const PedPomdp*>(ped_pomdp_model), predict_peds);

		const State& sample =
				*static_cast<PedPomdpBelief*>(solver->belief())->GetParticle(0);

		cout << "Car odom velocity " << summit_driving_simulator_->odom_vel.x
				<< " " << summit_driving_simulator_->odom_vel.y << endl;
		cout << "Car odom heading " << summit_driving_simulator_->odom_heading
				<< endl;
		cout << "Car base_link heading "
				<< summit_driving_simulator_->baselink_heading << endl;

		static_cast<PedPomdp*>(ped_pomdp_model)->PrintStateIDs(sample);
		static_cast<PedPomdp*>(ped_pomdp_model)->CheckPreCollision(&sample);
//		static_cast<const PedPomdp*>(ped_pomdp_model)->PrintState(sample);
//		static_cast<const PedPomdp*>(ped_pomdp_model)->ForwardAndVisualize(
//				sample, 10);				// 3 steps

		action = solver->Search().action;
	} else
		throw("drive mode not supported!");

	end_t = get_time_second();
	double search_time = (end_t - start_t);
	logi << "[RunStep] Time spent in " << typeid(*solver).name()
			<< "::Search(): " << search_time << endl;

	TruncPriors(cur_search_hist_len);

	OBS_TYPE obs;
	start_t = get_time_second();
	bool terminal = world->ExecuteAction(action, obs);
	end_t = get_time_second();
	double execute_time = (end_t - start_t);
	logi << "[RunStep] Time spent in ExecuteAction(): " << execute_time << endl;

	last_action_ = action;
	last_obs_ = obs;

//	SolverPrior::nn_priors[0]->DebugHistory("After execute action");

	cerr << "DEBUG: Ending step" << endl;

	return logger->SummarizeStep(step_++, round_, terminal, action, obs,
			step_start_t);
}

void Controller::TruncPriors(int cur_search_hist_len) {
	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		SolverPrior::nn_priors[i]->Truncate(cur_search_hist_len, true);
		logv << __FUNCTION__ << " truncating search history length to "
				<< cur_search_hist_len << endl;
		SolverPrior::nn_priors[i]->CompareHistoryWithRecorded();
	}
}

static int wait_count = 0;

void Controller::PlanningLoop(despot::Solver*& solver, World* world,
		Logger* logger) {

	logi << "Planning loop started at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	summit_driving_simulator_->stateTracker->detect_time = true;

	ros::spinOnce();

	logi << "First ROS spin finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	while (path_from_topic_.size() == 0) {
		cout << "Waiting for path, ts: " << Globals::ElapsedTime() << endl;
		ros::spinOnce();
		Globals::sleep_ms(100.0 / control_freq_ / time_scale);
		wait_count++;
		if (wait_count == 50) {
			ros::shutdown();
		}
	}

	logi << "path_from_topic_ received at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	logi << "Executing first step" << endl;

	RunStep(solver, world, logger);
	logi << "First step end at the " << Globals::ElapsedTime() << "th second"
			<< endl;

	cerr << "DEBUG: before entering controlloop" << endl;
	timer_ = nh_.createTimer(ros::Duration(1.0 / control_freq_ / time_scale),
			(boost::bind(&Controller::RunStep, this, solver, world, logger)));

}

int Controller::RunPlanning(int argc, char *argv[]) {
	cerr << "DEBUG: Starting planning" << endl;

	/* =========================
	 * initialize parameters
	 * =========================*/
	string solver_type = "DESPOT";
	bool search_solver;
	int num_runs = 1;
	string world_type = "pomdp";
	string belief_type = "DEFAULT";
	int time_limit = -1;

	option::Option *options = InitializeParamers(argc, argv, solver_type,
			search_solver, num_runs, world_type, belief_type, time_limit);
	if (options == NULL)
		return 0;
	logi << "InitializeParamers finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	if (Globals::config.useGPU)
		PrepareGPU();

	clock_t main_clock_start = clock();

	/* =========================
	 * initialize model
	 * =========================*/
	DSPOMDP *model = InitializeModel(options);
	assert(model != NULL);
	logi << "InitializeModel finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize world
	 * =========================*/
	World *world = InitializeWorld(world_type, model, options);

	cerr << "DEBUG: End initializing world" << endl;
	assert(world != NULL);
	logi << "InitializeWorld finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize belief
	 * =========================*/

	cerr << "DEBUG: Initializing belief" << endl;
	Belief* belief = model->InitialBelief(world->GetCurrentState(),
			belief_type);
	assert(belief != NULL);
	ped_belief_ = static_cast<PedPomdpBelief*>(belief);

	summit_driving_simulator_->beliefTracker = ped_belief_->beliefTracker;

	logi << "InitialBelief finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize solver
	 * =========================*/
	cerr << "DEBUG: Initializing solver" << endl;

	solver_type = ChooseSolver();
	Solver *solver = InitializeSolver(model, belief, solver_type, options);

	logi << "InitializeSolver finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize logger
	 * =========================*/
	Logger *logger = NULL;
	InitializeLogger(logger, options, model, belief, solver, num_runs,
			main_clock_start, world, world_type, time_limit, solver_type);
	//world->world_seed(world_seed);

	/* =========================
	 * Display parameters
	 * =========================*/
	DisplayParameters(options, model);

	/* =========================
	 * run planning
	 * =========================*/
	cerr << "DEBUG: Starting rounds" << endl;
	logger->InitRound(world->GetCurrentState());
	round_ = 0;
	step_ = 0;
	summit_driving_simulator_->beliefTracker->Text();
	logi << "InitRound finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	PlanningLoop(solver, world, logger);
	ros::spin();

	logger->EndRound();

	PrintResult(1, logger, main_clock_start);

	return 0;
}
