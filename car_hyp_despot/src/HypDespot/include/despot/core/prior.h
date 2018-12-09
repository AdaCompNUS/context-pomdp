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

public:
	SolverPrior(const DSPOMDP* model):model_(model){;}
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
#ifndef __CUDACC__

	// lets drive
	virtual void Process_history(int) = 0;
	virtual std::vector<torch::Tensor> Process_history_input() = 0;
	virtual std::vector<torch::Tensor> Process_nodes_input(const std::vector<State*>& vnode_states) = 0;
//	virtual torch::Tensor Combine_images(const at::Tensor& node_image, const at::Tensor& hist_images) = 0;
	virtual void Compute(vector<torch::Tensor>& images, map<OBS_TYPE, despot::VNode*>& vnode)=0;
#endif


public:
	static std::vector<SolverPrior*> nn_priors;
};



#endif
