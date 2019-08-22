#include "controller.h"
#include "core/node.h"
#include "core/solver.h"
#include "core/globals.h"
#include <csignal>
#include <time.h>
#include "boost/bind.hpp"
#include "world_simulator.h"
#include "pomdp_simulator.h"

#include "custom_particle_belief.h"
#include "neural_prior.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

using namespace std;
using namespace despot;

int Controller::b_use_drive_net_=0;
int Controller::gpu_id_=0;
float Controller::time_scale_ = 1.0;
std::string Controller::model_file_ = "";
std::string Controller::value_model_file_ = "";

bool path_missing = false;

static DSPOMDP* ped_pomdp_model;
static ACT_TYPE action = (ACT_TYPE)(-1);
static OBS_TYPE obs =(OBS_TYPE)(-1);

bool predict_peds = true;


struct my_sig_action {
    typedef void (* handler_type)(int, siginfo_t*, void*);

    explicit my_sig_action(handler_type handler)
    {
        memset(&_sa, 0, sizeof(struct sigaction));
        _sa.sa_sigaction = handler;
        _sa.sa_flags = SA_SIGINFO;
    }

    operator struct sigaction const*() const
    {
        return &_sa;
    }
protected:
    struct sigaction _sa;
};

struct div_0_exception {};

void handle_div_0(int sig, siginfo_t* info, void*)
{
	switch(info->si_code)
	{
		case FPE_INTDIV:
	        cout<< "Integer divide by zero."<<endl;
			break;
		case FPE_INTOVF:
			cout<< "Integer overflow. "<<endl;
			break;
		case FPE_FLTUND:
			cout<< "Floating-point underflow. "<<endl;	
			break;
		case FPE_FLTRES:
			cout<< "Floating-point inexact result. "<<endl;
			break;
		case FPE_FLTINV:
			cout<< "Floating-point invalid operation. "<<endl;
			break;
		case FPE_FLTSUB:
			cout<< "Subscript out of range. "<<endl;
			break;
		case FPE_FLTDIV:
			cout<< "Floating-point divide by zero. "<<endl;
			break;
		case FPE_FLTOVF:
			cout<< "Floating-point overflow. "<<endl;
			break;
	};
	exit(-1);
}

Controller::Controller(ros::NodeHandle& _nh, bool fixed_path, double pruning_constant, double pathplan_ahead, string obstacle_file_name):  
nh(_nh), fixed_path_(fixed_path), pathplan_ahead_(pathplan_ahead), obstacle_file_name_(obstacle_file_name)
{
	my_sig_action sa(handle_div_0);
    if (0 != sigaction(SIGFPE, sa, NULL)) {
        std::cerr << "!!!!!!!! fail to setup segfault handler !!!!!!!!" << std::endl;
        //return 1;
    }

/// for golfcart
 /*   Path p;
    COORD start = COORD(-205, -142.5);
    COORD goal = COORD(-189, -142.5);
    p.push_back(start);
    p.push_back(goal);
    worldModel.setPath(p.interpolate());
    fixed_path_ = true;
*/

/// for audi r8
	cerr <<"DEBUG: Entering Controller()"<<endl;

    simulation_mode_ = UNITY;
    fixed_path_=false;
	Globals::config.pruning_constant = pruning_constant;
	global_frame_id = ModelParams::rosns + "/map";
	control_freq=ModelParams::control_freq;

	cout << "=> fixed_path = " << fixed_path_ << endl;
	cout << "=> pathplan_ahead (will be reset to 0 for lets-drive mode)= " << pathplan_ahead_ << endl;
	cout << "=> pruning_constant = (will be rewritten by planner::initdefaultparams)" << pruning_constant << endl;

	cerr << "DEBUG: Initializing publishers..." << endl;

	pathPub_= nh.advertise<nav_msgs::Path>("pomdp_path_repub",1, true); // for visualization

	pathSub_= nh.subscribe("plan", 1, &Controller::RetrievePathCallBack, this); // receive path from path planner

    navGoalSub_ = nh.subscribe("navgoal", 1, &Controller::setGoal, this); //receive user input of goal	
	
	start_goal_pub=nh.advertise<ped_pathplan::StartGoal> ("ped_path_planner/planner/start_goal", 1);//send goal to path planner
    
    //imitation learning

	last_action=-1;
	last_obs=-1;

	unity_driving_simulator_ = NULL;
	pomdp_driving_simulator_ = NULL;

}


DSPOMDP* Controller::InitializeModel(option::Option* options) {
	cerr << "DEBUG: Initializing model" << endl;

	DSPOMDP* model = new PedPomdp();
	static_cast<PedPomdp*>(model)->world_model=&SimulatorBase::worldModel;

	model_ = model;
	ped_pomdp_model= model;

	return model;
}


void Controller::CreateNNPriors(DSPOMDP* model) {
	cerr << "DEBUG: Creating solver prior " << endl;

	SolverPrior::record_init_time();

	if (Globals::config.use_multi_thread_) {
		SolverPrior::nn_priors.resize(Globals::config.NUM_THREADS);
	} else
		SolverPrior::nn_priors.resize(1);

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		cerr << "DEBUG: Creating prior "<< i << endl;

		SolverPrior::nn_priors[i] =
				static_cast<PedPomdp*>(model)->CreateSolverPrior(
						unity_driving_simulator_, "NEURAL", false);

		SolverPrior::nn_priors[i]->prior_id(i);

		if (Globals::config.use_prior){
//			assert(model_file_ != "");
//			static_cast<PedNeuralSolverPrior*>(SolverPrior::nn_priors[i])->Load_model(model_file_);
			assert(value_model_file_ != "");
			static_cast<PedNeuralSolverPrior*>(SolverPrior::nn_priors[i])->Load_value_model(value_model_file_);
		}
	}

	prior_ = SolverPrior::nn_priors[0];
	cerr << "DEBUG: Created solver prior " << typeid(*prior_).name() << endl;
//	logi << "Created solver prior " << typeid(*prior_).name() << endl;

}

World* Controller::InitializeWorld(std::string& world_type, DSPOMDP* model, option::Option* options){

	cerr << "DEBUG: Initializing world" << endl;

   //Create a custom world as defined and implemented by the user
	World* world;
	switch(simulation_mode_){
		case POMDP:
			world=new POMDPSimulator(nh, static_cast<DSPOMDP*>(model),
			Globals::config.root_seed/*random seed*/, obstacle_file_name_);
   			break;
		case UNITY:
			world=new WorldSimulator(nh, static_cast<DSPOMDP*>(model),
				Globals::config.root_seed/*random seed*/, 
				pathplan_ahead_, obstacle_file_name_, COORD(goalx_, goaly_));
			break;
	}

	if (Globals::config.useGPU)
		model->InitGPUModel();

	//Establish connection with external system
	world->Connect();
   //Initialize the state of the external system
	world->Initialize();

	switch(simulation_mode_){
	case POMDP:
		static_cast<PedPomdp*>(model)->world_model=&(POMDPSimulator::worldModel);
		pomdp_driving_simulator_ = static_cast<POMDPSimulator*>(world);
		break;
	case UNITY:
		static_cast<PedPomdp*>(model)->world_model=&(WorldSimulator::worldModel);
		unity_driving_simulator_ = static_cast<WorldSimulator*>(world);
		unity_driving_simulator_->time_scale_ = time_scale_;
		break;
	}

   CreateNNPriors(model);

   return world;
}

void Controller::InitializeDefaultParameters() {
	cerr << "DEBUG: Initializing parameters" << endl;

	Globals::config.root_seed=time(NULL);

	Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9 / time_scale_;
	Globals::config.time_scale = time_scale_;
	Globals::config.num_scenarios=5;
	Globals::config.discount=1.0; //0.95;
	Globals::config.sim_len=300/*180*//*10*/; // this is not used
  
    Globals::config.xi = 0.97;
	//Globals::config.pruning_constant= 0.001; // passed as a ROS node param

	Globals::config.useGPU=true;

	if (b_use_drive_net_ == LETS_DRIVE)
		Globals::config.useGPU=false;

	Globals::config.GPUid=1;//default GPU
	Globals::config.use_multi_thread_=true;
	Globals::config.NUM_THREADS=10;

	Globals::config.exploration_mode=UCT;
	Globals::config.exploration_constant = 2.0;
//	Globals::config.exploration_constant=0.0;
	Globals::config.exploration_constant_o = 1.0;

	Globals::config.search_depth=9;
	Globals::config.max_policy_sim_len=/*Globals::config.sim_len+30*/5;

	Globals::config.experiment_mode = true;

	Globals::config.silence=false;
	Obs_type=OBS_INT_ARRAY;
	DESPOT::num_Obs_element_in_GPU=1+ModelParams::N_PED_IN*2+3;

	if (b_use_drive_net_ == LETS_DRIVE)
		Globals::config.use_prior = true;
	else
		Globals::config.use_prior = false;

	if (b_use_drive_net_ == JOINT_POMDP) {
		Globals::config.useGPU=false;
    	Globals::config.num_scenarios=1;
		Globals::config.NUM_THREADS=10;
		Globals::config.discount=0.95;
		Globals::config.search_depth=20;
		Globals::config.max_policy_sim_len=/*Globals::config.sim_len+30*/20;
		Globals::config.pruning_constant=0.001;
		Globals::config.exploration_constant = 0.1;
		Globals::config.silence = true;
  }

//	Globals::config.root_seed=1024;

	logging::level(3);

	logi << "Planner default parameters:" << endl;
	Globals::config.text();

}


std::string Controller::ChooseSolver(){
	return "DESPOT";
}


Controller::~Controller()
{

}


void Controller::sendPathPlanStart(const tf::Stamped<tf::Pose>& carpose) {
	if(fixed_path_ && WorldSimulator::worldModel.path.size()>0)  return;


	ped_pathplan::StartGoal startGoal;
	geometry_msgs::PoseStamped pose;
	tf::poseStampedTFToMsg(carpose, pose);

	// set start

	startGoal.start=pose;

	pose.pose.position.x=goalx_;
	pose.pose.position.y=goaly_;
	
	startGoal.goal=pose;

	logi << "Sending goal to path planner " << endl;

	logi << "start: " << startGoal.start.pose.position.x << startGoal.start.pose.position.y << endl;
	logi << "goal: " << startGoal.goal.pose.position.x << startGoal.goal.pose.position.y << endl;

	start_goal_pub.publish(startGoal);	
}

void Controller::setGoal(const geometry_msgs::PoseStamped::ConstPtr goal) {
    goalx_ = goal->pose.position.x;
    goaly_ = goal->pose.position.y;
}

void Controller::RetrievePathCallBack(const nav_msgs::Path::ConstPtr path)  {

	logi<<"receive path from navfn "<<path->poses.size()<<endl;

	if(fixed_path_ && path_from_topic.size()>0) return;

	if(path->poses.size()==0) {
		path_missing = true;

		DEBUG("Path missing from topic");
		return;
	}else{
		path_missing = false;
	}

	if (simulation_mode_ == UNITY && unity_driving_simulator_->b_update_il == true)
		unity_driving_simulator_->p_IL_data.plan = *path; // record to be further published for imitation learning

	Path p;
	for(int i=0;i<path->poses.size();i++) {
        COORD coord;
		coord.x=path->poses[i].pose.position.x;
		coord.y=path->poses[i].pose.position.y;
        p.push_back(coord);
	}

	// cout << "Path start " << p[0] << " end " << p.back() << endl;

	COORD path_end_from_goal = p.back() - COORD(goalx_, goaly_);

	if (path_end_from_goal.Length() > 2.0f + 1e-3){
		cerr << "Path end mismatch with car goal: path end = " <<
				"(" << p.back().x << "," << p.back().y << ")" <<
				", car goal=(" << goalx_ << "," << goaly_ << ")" << endl;
		// raise(SIGABRT);
		cerr << "reset car goal to path end" << endl;

		setCarGoal(p.back());
	}

	if (b_use_drive_net_ == LETS_DRIVE || b_use_drive_net_ == JOINT_POMDP || b_use_drive_net_ == IMITATION){
		pathplan_ahead_ = 0;
	}

	if(pathplan_ahead_>0 && path_from_topic.size()>0) {
		path_from_topic.cutjoin(p);
		path_from_topic = path_from_topic.interpolate();
        WorldSimulator::worldModel.setPath(path_from_topic);
	} else {
		path_from_topic = p.interpolate();
		WorldSimulator::worldModel.setPath(path_from_topic);
	}

  // if(Globals::config.useGPU)
    // static_cast<PedPomdp*>(ped_pomdp_model)->UpdateGPUPath();

	publishPath(path->header.frame_id, path_from_topic);
}

void Controller::publishPath(const string& frame_id, const Path& path) {
	nav_msgs::Path navpath;
	ros::Time plan_time = ros::Time::now();

	navpath.header.frame_id = frame_id;
	navpath.header.stamp = plan_time;
	
	for(const auto& s: path) {
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

bool Controller::getUnityPos(){
	logi << "[getUnityPos] Getting car pos from unity..." << endl;

    tf::Stamped<tf::Pose> in_pose, out_pose;
	in_pose.setIdentity();
	in_pose.frame_id_ = ModelParams::rosns + "/base_link";
	assert(unity_driving_simulator_);
	cout<<"global_frame_id: "<<global_frame_id<<" "<<endl;
	if(!unity_driving_simulator_->getObjectPose(global_frame_id, in_pose, out_pose)) {
		cerr<<"transform error within Controller::RunStep"<<endl;
		cout<<"laser frame "<<in_pose.frame_id_<<endl;
		ros::Rate err_retry_rate(10);
        err_retry_rate.sleep();
        return false; // skip the current step
	}
	else
		sendPathPlanStart(out_pose);

	return true && SimulatorBase::agents_data_ready;
}

bool Controller::RunPreStep(Solver* solver, World* world, Logger* logger) {

	cerr << "DEBUG: Running pre step" << endl;

	logger->CheckTargetTime();

	double step_start_t = get_time_second();

	if(simulation_mode_ == UNITY){
		bool unity_ready = getUnityPos();

		if (!unity_ready)
			return false;
	}

	cerr << "DEBUG: Pre-updating belief" << endl;

	double start_t = get_time_second();
//	solver->BeliefUpdate(last_action, last_obs);

	State* cur_state = world->GetCurrentState();


	// DEBUG("current state get");

	if(!cur_state)
		ERR(string_sprintf("cur state NULL"));

	// DEBUG("copy for search");
	State* search_state =static_cast<const PedPomdp*>(ped_pomdp_model)->CopyForSearch(cur_state);//create a new state for search

	// DEBUG("DeepUpdate");
	static_cast<PedPomdpBelief*>(solver->belief())->DeepUpdate(
			SolverPrior::nn_priors[0]->history_states(),
			SolverPrior::nn_priors[0]->history_states_for_search(),
			cur_state,
			search_state, last_action);

	// DEBUG("Track");
  	if (Globals::config.use_prior && SolverPrior::history_mode == "track"){
		SolverPrior::nn_priors[0]->Add_tensor_hist(search_state);
		for(int i=0; i<SolverPrior::nn_priors.size();i++){
			if (i > 0){
				SolverPrior::nn_priors[i]->add_car_tensor(SolverPrior::nn_priors[0]->last_car_tensor());
				SolverPrior::nn_priors[i]->add_map_tensor(SolverPrior::nn_priors[0]->last_map_tensor());
			}
		}
	}

	for(int i=0; i<SolverPrior::nn_priors.size();i++){
		SolverPrior::nn_priors[i]->Add(last_action, cur_state);
		SolverPrior::nn_priors[i]->Add_in_search(-1, search_state);

		logd << __FUNCTION__ << " add history search state of ts " <<
				static_cast<PomdpState*>(search_state)->time_stamp << endl;

		SolverPrior::nn_priors[i]->record_cur_history();
	}

	double end_t = get_time_second();
	double update_time = (end_t - start_t);
	logi << "[RunStep] Time spent in Update(): " << update_time << endl;

	if(simulation_mode_ == UNITY){
		unity_driving_simulator_->publishROSState();
		ped_belief_->publishAgentsPrediciton();
	}

	unity_driving_simulator_->beliefTracker->text();

	if(simulation_mode_ == UNITY){
		if (path_from_topic.size()==0){
			logi << "[RunStep] path topic not ready yet..." << endl;
			return false;
		}
		else{
			WorldSimulator::worldModel.path = path_from_topic;
		}
	}

	return true;
}

void Controller::PredictPedsForSearch(State* search_state) {
	if (predict_peds) {
		// predict state using last action
		if (last_action < 0 || last_action > model_->NumActions()) {
			cerr << "ERROR: wrong last action for prediction " << last_action
					<< endl;
		} else {

			unity_driving_simulator_->beliefTracker->cur_acc =
					static_cast<const PedPomdp*>(ped_pomdp_model)->GetAcceleration(
							last_action);
			unity_driving_simulator_->beliefTracker->cur_steering =
					static_cast<const PedPomdp*>(ped_pomdp_model)->GetSteering(
							last_action);

			cerr << "DEBUG: Prediction with last action:" << last_action
					<< " steer/acc = "
					<< unity_driving_simulator_->beliefTracker->cur_steering
					<< "/" << unity_driving_simulator_->beliefTracker->cur_acc
					<< endl;

			auto predicted =
					unity_driving_simulator_->beliefTracker->predictPedsCurVel(
							static_cast<PomdpState*>(search_state),
							unity_driving_simulator_->beliefTracker->cur_acc,
							unity_driving_simulator_->beliefTracker->cur_steering);

			PomdpState* predicted_state =
					static_cast<PomdpState*>(static_cast<const PedPomdp*>(ped_pomdp_model)->Copy(
							&predicted));

			static_cast<const PedPomdp*>(ped_pomdp_model)->PrintStateAgents(*predicted_state, string("predicted_agents"));

      if (Globals::config.use_prior && SolverPrior::history_mode == "track"){
				SolverPrior::nn_priors[0]->Add_tensor_hist(predicted_state);

				for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
					if (i > 0) {
						SolverPrior::nn_priors[i]->add_car_tensor(
								SolverPrior::nn_priors[0]->last_car_tensor());
						SolverPrior::nn_priors[i]->add_map_tensor(
								SolverPrior::nn_priors[0]->last_map_tensor());
					}
				}
			}

			for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
				SolverPrior::nn_priors[i]->Add_in_search(-1, predicted_state);

				logd << __FUNCTION__ << " add predicted search state of ts "
						<< predicted_state->time_stamp
						<< " predicted from search state of ts "
						<< static_cast<PomdpState*>(search_state)->time_stamp
						<< " hist len " << SolverPrior::nn_priors[i]->Size(true)
						<< endl;

				//		SolverPrior::nn_priors[i]->DebugHistory("Add prediction " +
				//			std::to_string(SolverPrior::nn_priors[i]->Size(true)-1));
			}
		}
	}
}

void Controller::UpdatePriors(const State* cur_state, State* search_state) {
	//	SolverPrior::nn_priors[0]->DebugHistory("After Deep update");
  if (Globals::config.use_prior && SolverPrior::history_mode == "track"){
		SolverPrior::nn_priors[0]->Add_tensor_hist(search_state);
		for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
			if (i > 0) {
				SolverPrior::nn_priors[i]->add_car_tensor(
						SolverPrior::nn_priors[0]->last_car_tensor());
				SolverPrior::nn_priors[i]->add_map_tensor(
						SolverPrior::nn_priors[0]->last_map_tensor());
			}
		}
	}
	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		// make sure the history has not corrupted
		SolverPrior::nn_priors[i]->compare_history_with_recorded();

		SolverPrior::nn_priors[i]->Add(last_action, cur_state);
		SolverPrior::nn_priors[i]->Add_in_search(-1, search_state);

		logd << __FUNCTION__ << " add history search state of ts "
				<< static_cast<PomdpState*>(search_state)->time_stamp
				<< " hist len " << SolverPrior::nn_priors[i]->Size(true)
				<< endl;

		if (SolverPrior::nn_priors[i]->Size(true) == 10)
			Record_debug_state(search_state);

		SolverPrior::nn_priors[i]->record_cur_history();
	}
	logi << "history len = " << SolverPrior::nn_priors[0]->Size(false) << endl;
	logi << "history_in_search len = " << SolverPrior::nn_priors[0]->Size(true)
			<< endl;
}

bool Controller::RunStep(despot::Solver* solver, World* world, Logger* logger) {

	cerr << "DEBUG: Running step" << endl;

//	SolverPrior::nn_priors[0]->DebugHistory("Start step");

	logger->CheckTargetTime();

	double step_start_t = get_time_second();

	if(simulation_mode_ == UNITY){
		bool unity_ready = getUnityPos();

		if (!unity_ready)
			return false;

		if (path_from_topic.size()==0){
			logi << "[RunStep] path topic not ready yet..." << endl;
			return false;
		}
		else{
			CheckCurPath();
		}
	}

	// imitation learning: pause update of car info and path info for imitation data
	switch(simulation_mode_){
	case UNITY:
		unity_driving_simulator_->b_update_il = false ;

		break;
	case POMDP:
		pomdp_driving_simulator_->b_update_il = false ;

		break;
	}

	cerr << "DEBUG: Updating belief" << endl;

	double start_t = get_time_second();
//	solver->BeliefUpdate(last_action, last_obs);

	const State* cur_state=world->GetCurrentState();
	unity_driving_simulator_->speed_in_search_state_ = unity_driving_simulator_->real_speed_;

	assert(cur_state);

//	SolverPrior::nn_priors[0]->DebugHistory("After get current step");

	cout << "current state address" << cur_state << endl;

	State* search_state =static_cast<const PedPomdp*>(ped_pomdp_model)->CopyForSearch(cur_state);//create a new state for search

//	SolverPrior::nn_priors[0]->DebugHistory("After copy for search");

	static_cast<PedPomdpBelief*>(solver->belief())->DeepUpdate(
			SolverPrior::nn_priors[0]->history_states(),
			SolverPrior::nn_priors[0]->history_states_for_search(),
			cur_state,
			search_state, last_action);

//	SolverPrior::nn_priors[0]->DebugHistory("After Deep update");

	UpdatePriors(cur_state, search_state);

	double end_t = get_time_second();
	double update_time = (end_t - start_t);
	logi << "[RunStep] Time spent in Update(): " << update_time << endl;

	if(simulation_mode_ == UNITY){
		unity_driving_simulator_->publishROSState();
		ped_belief_->publishAgentsPrediciton();
	}

	unity_driving_simulator_->beliefTracker->text();

	int cur_search_hist_len = 0, cur_tensor_hist_len = 0;
	cur_search_hist_len = SolverPrior::nn_priors[0]->Size(true);
	cur_tensor_hist_len = SolverPrior::nn_priors[0]->Tensor_hist_size();

  if(Globals::config.use_prior && SolverPrior::history_mode=="track" && cur_search_hist_len != cur_tensor_hist_len){
		cerr << "State hist length "<< cur_search_hist_len <<
				" mismatch with tensor hist length " << cur_tensor_hist_len << endl;
		raise(SIGABRT);
	}

	PredictPedsForSearch(search_state);

	start_t = get_time_second();
  ACT_TYPE action = static_cast<const PedPomdp*>(ped_pomdp_model)->GetActionID(0.0, 0.0);
	double step_reward;
  if (b_use_drive_net_ == NO || b_use_drive_net_ == JOINT_POMDP){
		cerr << "DEBUG: Search for action using " <<typeid(*solver).name()<< endl;
		static_cast<PedPomdpBelief*>(solver->belief())->ResampleParticles(
			static_cast<const PedPomdp*>(ped_pomdp_model), predict_peds);
		
		const State& sample = *static_cast<PedPomdpBelief*>(solver->belief())->GetParticle(0);
		static_cast<const PedPomdp*>(ped_pomdp_model)->PrintState(sample);

		static_cast<const PedPomdp*>(ped_pomdp_model)->ForwardAndVisualize(sample, 10);// 3 steps		

		action = solver->Search().action;
	}
	else if (b_use_drive_net_ == LETS_DRIVE){
//	 	int state_code = unity_driving_simulator_->worldModel.hasMinSteerPath(cur_state);
		cerr << "DEBUG: Search for action using " <<typeid(*solver).name()<< " with NN prior." << endl;
		assert(solver->belief());
		cerr << "DEBUG: Sampling particles" << endl;
		static_cast<PedPomdpBelief*>(solver->belief())->ResampleParticles(
			static_cast<const PedPomdp*>(ped_pomdp_model), predict_peds);
		cerr << "DEBUG: Launch search with NN prior" << endl;
		action = solver->Search().action;

		cout << "recording SolverPrior::nn_priors[0]->searched_action" << endl;
		SolverPrior::nn_priors[0]->searched_action = action;
	}
	else if (b_use_drive_net_ == IMITATION){
		// Query the drive_net for actions, do nothing here
	}
	else
		throw("drive net usage mode not supported!");

	end_t = get_time_second();
	double search_time = (end_t - start_t);
	logi << "[RunStep] Time spent in " << typeid(*solver).name()
			<< "::Search(): " << search_time << endl;

	TruncPriors(cur_search_hist_len, cur_tensor_hist_len);

	// imitation learning: renable data update for imitation data
	switch(simulation_mode_){
	case UNITY:
		unity_driving_simulator_->b_update_il = true ;

		break;
	case POMDP:
		pomdp_driving_simulator_->b_update_il = true ;

		break;
	}

	OBS_TYPE obs;
	start_t = get_time_second();
	bool terminal = world->ExecuteAction(action, obs);
	end_t = get_time_second();
	double execute_time = (end_t - start_t);
	logi << "[RunStep] Time spent in ExecuteAction(): " << execute_time << endl;

	last_action=action;
	last_obs=obs;

//	SolverPrior::nn_priors[0]->DebugHistory("After execute action");

	cerr << "DEBUG: Ending step" << endl;

	return logger->SummarizeStep(step_++, round_, terminal, action, obs,
			step_start_t);
}


void Controller::setCarGoal(COORD car_goal){
	goalx_ = car_goal.x;
	goaly_ = car_goal.y;

	unity_driving_simulator_->setCarGoal(car_goal);
}

void Controller::CheckCurPath(){
	if (path_missing){
		cerr << "Path missing, fixing steering" << endl;
		// use default move
		SolverPrior::prior_force_steer = true;
		if (unity_driving_simulator_->stateTracker->carvel>0.01){
			cerr << "Path missing, fixing acc" << endl;
			SolverPrior::prior_force_acc = true;
		}
		else
			SolverPrior::prior_force_acc = false;
	}
	else{
		WorldSimulator::worldModel.path = path_from_topic;

		COORD car_pos_from_goal = unity_driving_simulator_->stateTracker->carpos - COORD(goalx_, goaly_);

	//	if (car_pos_from_goal.Length() < 5.0)
	//	{
	//		SolverPrior::prior_force_steer = true;
	////		SolverPrior::prior_force_acc = true;
	//	}
		if (car_pos_from_goal.Length() < 10.0)
		{
			SolverPrior::prior_force_steer = true;

			if (unity_driving_simulator_->stateTracker->carvel>0.01)
				SolverPrior::prior_force_acc = true;
			else
				SolverPrior::prior_force_acc = false;
		}
		else if (car_pos_from_goal.Length() < 8.0 && unity_driving_simulator_->stateTracker->carvel
				>= ModelParams::AccSpeed/ModelParams::control_freq)
		{
			SolverPrior::prior_discount_optact = 10.0;
		}
	}

}

void Controller::TruncPriors(int cur_search_hist_len, int cur_tensor_hist_len){
	for(int i=0; i<SolverPrior::nn_priors.size();i++){
		SolverPrior::nn_priors[i]->Truncate(cur_search_hist_len, true);
		logd << __FUNCTION__ << " truncating search history length to " <<
				cur_search_hist_len << endl;
		SolverPrior::nn_priors[i]->compare_history_with_recorded();
//		SolverPrior::nn_priors[i]->DebugHistory("Trunc history");
	}

	if (SolverPrior::history_mode == "track"){
		for(int i=0; i<SolverPrior::nn_priors.size();i++){
			SolverPrior::nn_priors[i]->Trunc_tensor_hist(cur_tensor_hist_len);
		}
	}
}

void Controller::PlanningLoop(despot::Solver*& solver, World* world, Logger* logger) {

//	sleep(2);
    while(path_from_topic.size()==0 || SolverPrior::nn_priors[0]->Size(true) < 4){
    	logi << "Executing pre-step" << endl;
    	RunPreStep(solver, world, logger);
		ros::spinOnce();

		logi << "sleeping for "<< 1.0/control_freq/time_scale_ << "s"<< endl;
    	Globals::sleep_ms(1000.0/control_freq/time_scale_);

    	logi << "Pre-step sleep end" << endl;

    }

	logi << "Executing first step" << endl;

	RunStep(solver, world, logger);

    cerr <<"DEBUG: before entering controlloop"<<endl;
    timer_ = nh.createTimer(ros::Duration(1.0/control_freq/time_scale_),
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
	if(options==NULL)
		return 0;

	if(Globals::config.useGPU)
		PrepareGPU();

	clock_t main_clock_start = clock();

	/* =========================
	 * initialize model
	 * =========================*/
	DSPOMDP *model = InitializeModel(options);
	assert(model != NULL);

	/* =========================
	 * initialize world
	 * =========================*/
	World *world = InitializeWorld(world_type, model, options);

	cerr << "DEBUG: End initializing world" << endl;
	assert(world != NULL);

	/* =========================
	 * initialize belief
	 * =========================*/

	cerr << "DEBUG: Initializing belief" << endl;
	Belief* belief = model->InitialBelief(world->GetCurrentState(), belief_type);
	assert(belief != NULL);
	ped_belief_=static_cast<PedPomdpBelief*>(belief);
	switch(simulation_mode_){
	case UNITY:
		unity_driving_simulator_->beliefTracker = ped_belief_->beliefTracker;
		break;
	case POMDP:
		pomdp_driving_simulator_->beliefTracker = ped_belief_->beliefTracker;
		break;
	}

	/* =========================
	 * initialize solver
	 * =========================*/
	cerr << "DEBUG: Initializing solver" << endl;

	solver_type = ChooseSolver();
	Solver *solver = InitializeSolver(model, belief, solver_type, options);

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
	round_=0; step_=0;
	unity_driving_simulator_->beliefTracker->text();
	PlanningLoop(solver, world, logger);
	ros::spin();

	logger->EndRound();

	PrintResult(1, logger, main_clock_start);

	return 0;
}
