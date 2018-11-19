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


using namespace std;

bool Controller::b_use_drive_net_=false;
int Controller::gpu_id_=0;


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
    fixed_path_=false;

    

	cout << "fixed_path = " << fixed_path_ << endl;
	cout << "pathplan_ahead = " << pathplan_ahead_ << endl;
	Globals::config.pruning_constant = pruning_constant;

	global_frame_id = ModelParams::rosns + "/map";
	cerr <<"DEBUG: Entering Controller()"<<endl;
	control_freq=ModelParams::control_freq;

	cerr << "DEBUG: Initializing publishers..." << endl;


    
	pathPub_= nh.advertise<nav_msgs::Path>("pomdp_path_repub",1, true); // for visualization

	pathSub_= nh.subscribe("plan", 1, &Controller::RetrievePathCallBack, this); // receive path from path planner

    navGoalSub_ = nh.subscribe("navgoal", 1, &Controller::setGoal, this); //receive user input of goal	
	

	start_goal_pub=nh.advertise<ped_pathplan::StartGoal> ("ped_path_planner/planner/start_goal", 1);//send goal to path planner
    
    //imitation learning

	b_use_drive_net_ = false;

	last_action=-1;
	last_obs=-1;

	unity_driving_simulator_ = NULL;
	pomdp_driving_simulator_ = NULL;
}


DSPOMDP* Controller::InitializeModel(option::Option* options) {
	cerr << "DEBUG: Initializing model" << endl;

	DSPOMDP* model = new PedPomdp();
	model_ = model;
	return model;
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
		break;
	}
   return world;
}

void Controller::InitializeDefaultParameters() {
	cerr << "DEBUG: Initializing parameters" << endl;

	Globals::config.root_seed=time(NULL);

	Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
	//Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
	Globals::config.num_scenarios=1;
	Globals::config.discount=/*0.983*/0.95/*0.966*/;
	Globals::config.sim_len=200/*180*//*10*/;

	Globals::config.search_depth=10;
	Globals::config.max_policy_sim_len=/*Globals::config.sim_len+30*/5;

	Globals::config.exploration_constant=/*0.095*//*0.1*/0.5;

	Globals::config.silence=false;

	Globals::config.root_seed=1024;

	logging::level(3);
}

std::string Controller::ChooseSolver(){
	return "POMDPLITE";
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
	start_goal_pub.publish(startGoal);	
}

void Controller::setGoal(const geometry_msgs::PoseStamped::ConstPtr goal) {
    goalx_ = goal->pose.position.x;
    goaly_ = goal->pose.position.y;
}

void Controller::RetrievePathCallBack(const nav_msgs::Path::ConstPtr path)  {
//	cout<<"receive path from navfn "<<path->poses.size()<<endl;
	if(fixed_path_ && WorldSimulator::worldModel.path.size()>0) return;

	if(path->poses.size()==0) return;

	if (simulation_mode_ == UNITY && unity_driving_simulator_->b_update_il == true)
		unity_driving_simulator_->p_IL_data.plan = *path; // record to be further published for imitation learning

	Path p;
	for(int i=0;i<path->poses.size();i++) {
        COORD coord;
		coord.x=path->poses[i].pose.position.x;
		coord.y=path->poses[i].pose.position.y;
        p.push_back(coord);
	}

	if(pathplan_ahead_>0 && WorldSimulator::worldModel.path.size()>0) {
        WorldSimulator::worldModel.path.cutjoin(p);
        auto pi = WorldSimulator::worldModel.path.interpolate();
        WorldSimulator::worldModel.setPath(pi);
	} else {
		WorldSimulator::worldModel.setPath(p.interpolate());
	}

	publishPath(path->header.frame_id, WorldSimulator::worldModel.path);
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



bool Controller::RunStep(Solver* solver, World* world, Logger* logger) {


	cerr << "DEBUG: Running step" << endl;

	logger->CheckTargetTime();

	double step_start_t = get_time_second();

	if(simulation_mode_ == UNITY){
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
	solver->BeliefUpdate(last_action, last_obs);
	double end_t = get_time_second();
	double update_time = (end_t - start_t);
	logi << "[RunStep] Time spent in Update(): " << update_time << endl;

	if(simulation_mode_ == UNITY){
		unity_driving_simulator_->publishROSState();
		ped_belief_->publishPedsPrediciton();
	}

	//ped_belief_->publishBelief();// replaced by publishImitationData


	start_t = get_time_second();
	ACT_TYPE action;
	double step_reward;
	if (!b_use_drive_net_){
		cerr << "DEBUG: Search for action using " <<typeid(*solver).name()<< endl;

		action = solver->Search().action;
	}
	else{
		// Query the drive_net for actions


	}


	end_t = get_time_second();
	double search_time = (end_t - start_t);
	logi << "[RunStep] Time spent in " << typeid(*solver).name()
			<< "::Search(): " << search_time << endl;

	// imitation learning: renable data update for imitation data
	switch(simulation_mode_){
	case UNITY:
		unity_driving_simulator_->b_update_il = true ; 

		break;
	case POMDP:
		pomdp_driving_simulator_->b_update_il = true ; 

		break;
	}

	cerr << "DEBUG: Executing action" << endl;

	OBS_TYPE obs;
	start_t = get_time_second();
	bool terminal = world->ExecuteAction(action, obs);
	end_t = get_time_second();
	double execute_time = (end_t - start_t);
	logi << "[RunStep] Time spent in ExecuteAction(): " << execute_time << endl;

	last_action=action;
	last_obs=obs;

	cerr << "DEBUG: Ending step" << endl;

	return logger->SummarizeStep(step_++, round_, terminal, action, obs,
			step_start_t);
}

void Controller::PlanningLoop(Solver*& solver, World* world, Logger* logger) {

	cerr <<"DEBUG: before entering controlloop"<<endl;
    timer_ = nh.createTimer(ros::Duration(1.0/control_freq), 
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

	PlanningLoop(solver, world, logger);
	ros::spin();
	logger->EndRound();

	PrintResult(1, logger, main_clock_start);

	return 0;
}
