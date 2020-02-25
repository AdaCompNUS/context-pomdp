
#define ONEOVERSQRT2PI 1.0 / sqrt(2.0 * M_PI)

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "context_pomdp.h"
#include "default_prior.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>
#include <cmath>

bool do_print = false;
State* debug_state = NULL;

DefaultPrior::DefaultPrior(const DSPOMDP* model,
		WorldModel& world) :
		SolverPrior(model), world_model(world) {

	prior_id_ = 0;
	logv << "DEBUG: Initializing DefaultPrior" << endl;
	action_probs_.resize(model->NumActions());
}

void DefaultPrior::DebugHistory(string msg){
	for (int t = 0; t < as_history_in_search_.Size(); t++){
		auto state = as_history_in_search_.state(t);
		Debug_state(state,  msg + "_t_"+ std::to_string(t), model_);
	}
}


bool Compare_states(PomdpState* state1, PomdpState* state2){
	if ((state1->car.pos.x != state2->car.pos.x) ||
			(state1->car.pos.y != state2->car.pos.y) ||
			(state1->car.vel != state2->car.vel) ||
			(state1->car.heading_dir != state2->car.heading_dir)){
		cerr << "!!!!!!! car diff !!!!!!" << endl;
		return true;
	}

	if (state1->num != state2->num){
		cerr << "!!!!!!! ped num diff !!!!!!" << endl;
		return true;
	}

	bool diff= false;
	for (int i =0 ; i< state1->num; i++){
		diff = diff || (state1->agents[i].pos.x != state2->agents[i].pos.x);
		diff = diff || (state1->agents[i].pos.y != state2->agents[i].pos.y);
		diff = diff || (state1->agents[i].intention != state2->agents[i].intention);

		if (diff){
			cerr << "!!!!!!! ped " << i << " diff !!!!!!" << endl;
			return true;
		}
	}

	return false;
}

void Debug_state(State* state, string msg, const DSPOMDP* model){
	if (state == debug_state){
		bool mode = DESPOT::Debug_mode;
		DESPOT::Debug_mode = false;

		PomdpState* hist_state = static_cast<PomdpState*>(state);
		static_cast<const ContextPomdp*>(model)->PrintState(*hist_state);

		DESPOT::Debug_mode = mode;

		cerr << "=================== " << msg << " breakpoint ==================="<< endl;
		ERR("");
	}
}

void Record_debug_state(State* state){
	debug_state = state;
}

void DefaultPrior::RecordCurHistory(){
	as_history_in_search_recorded.Truncate(0);
	for (int i = 0 ;i<as_history_in_search_.Size(); i++){
		as_history_in_search_recorded.Add(as_history_in_search_.Action(i), as_history_in_search_.state(i));
	}
}

void DefaultPrior::CompareHistoryWithRecorded(){

	if (as_history_in_search_.Size() != as_history_in_search_recorded.Size()){
		cerr << "ERROR: history length changed after search!!!" << endl;
		ERR("");
	}
	for (int i = 0 ;i<as_history_in_search_recorded.Size(); i++){
		PomdpState* recorded_hist_state =  static_cast<PomdpState*>(as_history_in_search_recorded.state(i));
		PomdpState* hist_state =  static_cast<PomdpState*>(as_history_in_search_recorded.state(i));

		bool different = Compare_states(recorded_hist_state, hist_state);

		if( different){
			cerr << "ERROR: history "<< i << " changed after search!!!" << endl;
			static_cast<const ContextPomdp*>(model_)->PrintState(*recorded_hist_state, "Recorded hist state");
			static_cast<const ContextPomdp*>(model_)->PrintState(*hist_state, "Hist state");

			ERR("");
		}
	}
}

std::vector<ACT_TYPE> DefaultPrior::ComputeLegalActions(const State* state, const DSPOMDP* model){
  const PomdpState* pomdp_state = static_cast<const PomdpState*>(state);
  const ContextPomdp* pomdp_model = static_cast<const ContextPomdp*>(model_);

  ACT_TYPE act_start, act_end;

  double steer_to_path = pomdp_model->world_model.GetSteerToPath(pomdp_state->car);

  act_start = pomdp_model->GetActionID(pomdp_model->GetSteerIDfromSteering(steer_to_path), 0);

  act_end = act_start + 2 * ModelParams::NUM_ACC + 1;

  std::vector<ACT_TYPE> legal_actions;
  for (ACT_TYPE action = act_start; action < act_end; action++) {
    legal_actions.push_back(action);
  }

  return legal_actions;
}
