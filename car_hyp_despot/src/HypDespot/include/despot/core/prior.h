#ifndef PRIOR_H
#define PRIOR_H
#include <despot/planner.h>

using namespace std;

using namespace despot;
/* =============================================================================
 * SolverPrior class
 * =============================================================================*/

class SolverPrior {
protected:
	const DSPOMDP* model_;
	ActionStateHistory as_history_;
	VariableActionStateHistory as_history_in_search_;
	std::vector<double> action_probs_;

	int prior_id_;

public:
	SolverPrior(const DSPOMDP* model):model_(model){searched_action = -1; default_action = -1;}
	virtual ~SolverPrior(){;}

	inline virtual int SmartCount(ACT_TYPE action) const {
		return 10;
	}

	inline virtual double SmartValue(ACT_TYPE action) const {
		return 1;
	}

	inline virtual const ActionStateHistory& history() const {
		return as_history_;
	}

	inline virtual VariableActionStateHistory& history_in_search() {
		return as_history_in_search_;
	}

	inline virtual void history_in_search(VariableActionStateHistory h) {
		as_history_in_search_ = h;
	}

	inline virtual void history(ActionStateHistory h) {
		as_history_ = h;
	}

	inline const std::vector<const State*>& history_states() {
		return as_history_.states();
	}

	inline std::vector<State*>& history_states_for_search() {
		return as_history_in_search_.states();
	}

	inline virtual void Add(ACT_TYPE action, const State* state) {
		as_history_.Add(action, state);
	}
	inline virtual void Add_in_search(ACT_TYPE action, State* state) {
		as_history_in_search_.Add(action, state);
	}

	inline virtual void PopLast(bool insearch) {
		(insearch)? as_history_in_search_.RemoveLast(): as_history_.RemoveLast();
	}

	inline virtual void PopAll(bool insearch) {
		(insearch)? as_history_in_search_.Truncate(0): as_history_.Truncate(0);
	}

	inline void Truncate(int d, bool insearch) {
		(insearch)? as_history_in_search_.Truncate(d): as_history_.Truncate(d);
	}

	inline size_t Size(bool insearch) const {
		size_t s = (insearch)? as_history_in_search_.Size(): as_history_.Size();
		return s;
	}

	const std::vector<double>& action_probs() const;

public:

	virtual std::vector<ACT_TYPE> ComputeLegalActions(const State* state, const DSPOMDP* model) = 0;

	virtual void DebugHistory(string msg) = 0;

	virtual void record_cur_history()=0;

	virtual void compare_history_with_recorded()=0;

public:
	static std::vector<SolverPrior*> nn_priors;
	static std::string history_mode;
	static double prior_discount_optact;
	static bool prior_force_steer;
	static bool prior_force_acc;

public:

	void prior_id(int id){
		prior_id_ = id;
	}

	int prior_id(){
		return prior_id_;
	}

public:
	ACT_TYPE searched_action;
	ACT_TYPE default_action;

};

void Debug_state(State* state, std::string msg, const DSPOMDP* model);
void Record_debug_state(State* state);

#endif
