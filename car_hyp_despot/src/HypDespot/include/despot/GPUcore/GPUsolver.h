#ifndef GPUSOLVER_H
#define GPUSOLVER_H

#include <despot/GPUcore/GPUglobals.h>
#include <despot/GPUcore/GPUhistory.h>

#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class Dvc_DSPOMDP;
class Dvc_Belief;
struct Dvc_ValuedAction;

/* =============================================================================
 * SearchStatistics class
 * =============================================================================*/

struct Dvc_SearchStatistics {
	double initial_lb, initial_ub, final_lb, final_ub;
	double time_search;
	double time_path;
	double time_backup;
	double time_node_expansion;
	int num_policy_nodes;
	int num_tree_nodes;
	int num_expanded_nodes;
	int num_tree_particles;
	int num_particles_before_search;
	int num_particles_after_search;
	int num_trials;
	int longest_trial_length;

	/*HOST Dvc_SearchStatistics();

	friend HOST std::ostream& operator<<(std::ostream& os, const Dvc_SearchStatistics& statitics);*/
};

/* =============================================================================
 * GPUSolver class
 * =============================================================================*/

class GPUSolver {
protected:
	const Dvc_DSPOMDP* model_;
	//Dvc_Belief* belief_;
	//Dvc_History history_;

public:
	/*HOST GPUSolver(const Dvc_DSPOMDP* model, Dvc_Belief* belief);
	virtual HOST ~GPUSolver();

	*
	 * Find the optimal action for current belief, and optionally return the
	 * found value for the action. Return the value Globals::NEG_INFTY if the
	 * value is not to be used.

	//virtual HOST Dvc_ValuedAction Search() = 0;

	*
	 * Update current belief, history, and any other internal states that is
	 * needed for Search() to function correctly.

	//virtual HOST void Update(int action, OBS_TYPE obs);

	*
	 * Set initial belief for planning. Make sure internal states associated with
	 * initial belief are reset. In particular, history need to be cleaned, and
	 * allocated memory from previous searches need to be cleaned if not.

	virtual HOST void belief(Dvc_Belief* b);*/
	Dvc_Belief* belief();
};

} // namespace despot

#endif
