#ifndef PRIOR_H
#define PRIOR_H
#include <despot/planner.h>

//#ifndef __CUDACC__
#include <torch/script.h> // One-stop header.
//#include <ATen/ATen.h>
#include <torch/torch.h>
#include "torch/csrc/jit/ivalue.h"

//#endif

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


//	virtual const std::vector<double>& ComputePreference() = 0;
//
//	virtual double ComputeValue()=0;

	const std::vector<double>& action_probs() const;


public:
//#ifndef __CUDACC__

	// lets drive
	virtual void Process_history(despot::VNode*, int) = 0;
	virtual std::vector<torch::Tensor> Process_history_input(despot::VNode* cur_node) = 0;
	virtual std::vector<torch::Tensor> Process_nodes_input(const std::vector<despot::VNode*>& vnodes,
			const std::vector<State*>& vnode_states) = 0;
//	virtual torch::Tensor Combine_images(const at::Tensor& node_image, const at::Tensor& hist_images) = 0;
	virtual void Compute(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode) =0;
	virtual void ComputePreference(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode) =0;

	virtual void ComputeValue(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode) =0;

	virtual void Record_hist_len() = 0;

	virtual void print_prior_actions(ACT_TYPE) = 0;

	virtual void Clear_hist_timestamps() = 0;

	virtual void DebugHistory(string msg)=0;

//#endif


public:
	static std::vector<SolverPrior*> nn_priors;
	static std::string history_mode;

public:
    static std::chrono::time_point<std::chrono::system_clock> init_time_;

	static double get_timestamp();
	static void record_init_time();

	virtual void record_cur_history()=0;
	virtual void compare_history_with_recorded()=0;

	void prior_id(int id){
		prior_id_ = id;
	}

	int prior_id(){
		return prior_id_;
	}

public:
	ACT_TYPE searched_action;
	ACT_TYPE default_action;

	virtual void root_car_pos(double x, double y) = 0;

	virtual at::Tensor Process_state_to_map_tensor(const State* s) = 0;
	virtual at::Tensor Process_state_to_car_tensor(const State* s) = 0;

	virtual at::Tensor last_car_tensor() = 0;
	virtual void add_car_tensor(at::Tensor) = 0;

	virtual at::Tensor last_map_tensor() = 0;
	virtual void add_map_tensor(at::Tensor) = 0;

	virtual void Add_tensor_hist(const State* s) = 0;
	virtual void Trunc_tensor_hist(int size) = 0;

	virtual int Tensor_hist_size() = 0;
};

void Debug_state(State* state, std::string msg, const DSPOMDP* model);
void Record_debug_state(State* state);

#endif
