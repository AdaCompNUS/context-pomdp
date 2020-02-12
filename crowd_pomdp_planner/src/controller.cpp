#include "controller.h"
#include "core/node.h"
#include "core/solver.h"
#include "core/globals.h"
#include <csignal>
#include <time.h>
#include "boost/bind.hpp"
#include "world_simulator.h"
#include "crowd_belief.h"

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
		nh_(_nh), last_action_(-1), last_obs_(-1), model_(NULL), prior_(NULL), ped_belief_(
				NULL), context_pomdp_(NULL), summit_driving_simulator_(NULL) {
	my_sig_action sa(handle_div_0);
	if (0 != sigaction(SIGFPE, sa, NULL)) {
		std::cerr << "!!!!!!!! fail to setup segfault handler !!!!!!!!"
				<< std::endl;
	}

	control_freq_ = ModelParams::CONTROL_FREQ;

	logi << " Controller constructed at the " << Globals::ElapsedTime()
			<< "th second" << endl;
}

DSPOMDP* Controller::InitializeModel(option::Option* options) {
	cerr << "DEBUG: Initializing model" << endl;
	DSPOMDP* model = new ContextPomdp();
	model_ = model;
	context_pomdp_ = static_cast<ContextPomdp*>(model);

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
				static_cast<ContextPomdp*>(model)->CreateSolverPrior(
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

bool Controller::RunStep(despot::Solver* solver, World* world, Logger* logger) {
	double step_start_t = get_time_second();
	cerr << "DEBUG: Running step" << endl;
	logger->CheckTargetTime();

	cerr << "DEBUG: Getting world state" << endl;
	auto start_t = Time::now();
	const State* cur_state = world->GetCurrentState();
	if (cur_state == NULL)
		ERR("cur_state == NULL");
	cout << "current state address" << cur_state << endl;

	cerr << "DEBUG: Updating belief" << endl;
	ped_belief_->Update(last_action_, cur_state);
	ped_belief_->Text(cout);

	auto particles = ped_belief_->Sample(Globals::config.num_scenarios * 2);
	static_cast<const ContextPomdp*>(model_)->ForwardAndVisualize(
					*(particles[0]), 10);
	ParticleBelief particle_belief(particles, model_);
	solver->belief(&particle_belief);
	logi << "[RunStep] Time spent in Update(): "
			<< Globals::ElapsedTime(start_t) << endl;

	cerr << "DEBUG: Searching for action" << endl;
	start_t = Time::now();
	ACT_TYPE action;
	if (b_drive_mode == NO || b_drive_mode == JOINT_POMDP
			|| b_drive_mode == ROLL_OUT) {
		action = solver->Search().action;
	} else
		ERR("drive mode not supported!");
	logi << "[RunStep] Time spent in " << typeid(*solver).name()
			<< "::Search(): " << Globals::ElapsedTime(start_t) << endl;

	cerr << "DEBUG: Executing action" << endl;
	OBS_TYPE obs;
	start_t = Time::now();
	bool terminal = world->ExecuteAction(action, obs);
	last_action_ = action;
	last_obs_ = obs;
	logi << "[RunStep] Time spent in ExecuteAction(): "
			<< Globals::ElapsedTime(start_t) << endl;

	cerr << "DEBUG: Ending step" << endl;
	return logger->SummarizeStep(step_++, round_, terminal, action, obs,
			step_start_t);
}

static int wait_count = 0;

void Controller::PlanningLoop(despot::Solver*& solver, World* world,
		Logger* logger) {

	logi << "Planning loop started at the " << Globals::ElapsedTime()
			<< "th second" << endl;

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
	ped_belief_ = static_cast<CrowdBelief*>(belief);
	logi << "InitialBelief finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize solver
	 * =========================*/
	cerr << "DEBUG: Initializing solver" << endl;
	solver_type = ChooseSolver();
	Solver *solver = InitializeSolver(model, NULL, solver_type,
			options);
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
	logi << "InitRound finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	PlanningLoop(solver, world, logger);
	ros::spin();

	logger->EndRound();

	PrintResult(1, logger, main_clock_start);

	return 0;
}
