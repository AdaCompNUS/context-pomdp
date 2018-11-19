#include <string>
#include <despot/planner.h>


#include "simulator.h"
#include "controller.h"
#include "ped_pomdp.h"
#include "custom_particle_belief.h"

using namespace despot;
static DSPOMDP* ped_pomdp_model;
static ACT_TYPE action = (ACT_TYPE)(-1);
static OBS_TYPE obs =(OBS_TYPE)(-1);

DSPOMDP* DrivingController::InitializeModel(option::Option* options) {
	DSPOMDP* model = new PedPomdp();
	static_cast<PedPomdp*>(model)->world_model=&Simulator::worldModel;

	ped_pomdp_model= model;

	return model;
}

void DrivingController::CreateNNPriors(DSPOMDP* model) {
	if (Globals::config.use_multi_thread_) {
		SolverPrior::nn_priors.resize(Globals::config.NUM_THREADS);
	} else
		SolverPrior::nn_priors.resize(1);

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		SolverPrior::nn_priors[i] =
				static_cast<PedPomdp*>(model)->CreateSolverPrior(
						driving_simulator_, "NEURAL", false);
	}
	prior_ = SolverPrior::nn_priors[0];
	logi << "Created solver prior " << typeid(*prior_).name() << endl;
}

World* DrivingController::InitializeWorld(std::string& world_type, DSPOMDP* model, option::Option* options){
	//Create a custom world as defined and implemented by the user
	driving_simulator_=new Simulator(static_cast<DSPOMDP*>(model),
			Globals::config.root_seed/*random seed*/);

	if (Globals::config.useGPU)
		model->InitGPUModel();

	//Establish connection with external system
	driving_simulator_->Connect();
	//Initialize the state of the external system
	driving_simulator_->Initialize();

	CreateNNPriors(model);

	return driving_simulator_;
}

void DrivingController::InitializeDefaultParameters() {
	Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
	Globals::config.num_scenarios=1000;
	Globals::config.discount=0.95;
	Globals::config.sim_len=200;
	Globals::config.pruning_constant= 0.001;

	Globals::config.max_policy_sim_len=90;
	//Globals::config.search_depth = 30;

	Globals::config.GPUid=1;//default GPU
	Globals::config.useGPU=true	;
	Globals::config.use_multi_thread_=true;
	
	Globals::config.NUM_THREADS=5;

	Globals::config.exploration_mode=UCT;
	Globals::config.exploration_constant=0.3;
	Globals::config.exploration_constant_o = 1.0;

	Globals::config.experiment_mode = true;

	Globals::config.silence=false;
	Obs_type=OBS_INT_ARRAY;
	DESPOT::num_Obs_element_in_GPU=1+ModelParams::N_PED_IN*2+3;

	logging::level(0);
}

std::string DrivingController::ChooseSolver(){
	return "DESPOT";
}

bool DrivingController::RunStep(Solver* solver, World* world, Logger* logger) {

	cout << "=========== Customized RunStep ============" << endl;

	logger->CheckTargetTime();

	double step_start_t = get_time_second();

	double start_t = get_time_second();
	driving_simulator_->UpdateWorld();
	double end_t = get_time_second();
	double world_update_time = (end_t - start_t);
	logi << "[RunStep] Time spent in UpdateWorld(): " << world_update_time << endl;

	start_t = get_time_second();
	solver->BeliefUpdate(action, obs);

	//assert(nn_prior->history().Size());

	const State* cur_state=world->GetCurrentState();

	assert(cur_state);

	if(DESPOT::Debug_mode){
		/*cout << "Current simulator state before belief update: \n";
		static_cast<PedPomdp*>(ped_pomdp_model)->PrintWorldState(static_cast<const PomdpStateWorld&>(*cur_state));*/
	}

	State* search_state =static_cast<const PedPomdp*>(ped_pomdp_model)->CopyForSearch(cur_state);//create a new state for search

	static_cast<PedPomdpBelief*>(solver->belief())->DeepUpdate(
			SolverPrior::nn_priors[0]->history_states(),
			SolverPrior::nn_priors[0]->history_states_for_search(),
			cur_state,
			search_state, action);

	for(int i=0; i<SolverPrior::nn_priors.size();i++){
		SolverPrior::nn_priors[i]->Add(action, cur_state);
		SolverPrior::nn_priors[i]->Add_in_search(-1, search_state);
	}

	end_t = get_time_second();
	double update_time = (end_t - start_t);
	logi << "[RunStep] Time spent in Update(): " << update_time << endl;
	start_t = get_time_second();

	// Sample new particles using belieftracker
	static_cast<PedPomdpBelief*>(solver->belief())->ResampleParticles(static_cast<const PedPomdp*>(ped_pomdp_model));

	action = solver->Search().action;
	end_t = get_time_second();
	double search_time = (end_t - start_t);
	logi << "[RunStep] Time spent in " << typeid(*solver).name()
			<< "::Search(): " << search_time << endl;

	cout << "act= " << action << endl;

	start_t = get_time_second();
	bool terminal = world->ExecuteAction(action, obs);
	end_t = get_time_second();
	double execute_time = (end_t - start_t);
	logi << "[RunStep] Time spent in ExecuteAction(): " << execute_time << endl;

	return logger->SummarizeStep(step_++, round_, terminal, action, obs,
			step_start_t);
}


void DrivingController::PlanningLoop(Solver*& solver, World* world, Logger* logger) {
	bool terminal;
	for (int i = 0; i < Globals::config.sim_len; i++) {
		terminal = RunStep(solver, world, logger);
		if (terminal)
			break;
	}

	if(Globals::config.experiment_mode && !terminal){
		cout << "- final_state:\n";
		static_cast<PedPomdp*>(ped_pomdp_model)->PrintWorldState(
				static_cast<PomdpStateWorld&>(*driving_simulator_->GetCurrentState()), cout);
	}
}

int main(int argc, char* argv[]) {

  //bool result = DrivingController().RunPlanning(argc, argv);

	//debugging

  bool result = DrivingController().RunEvaluation(argc, argv);

  exit(result);
}
