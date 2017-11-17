#include <despot/GPUcore/GPUsolver.h>
#include <despot/util/logging.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUbelief.h>

using namespace std;

namespace despot {

/* =============================================================================
 * Dvc_SearchStatistics class
 * =============================================================================*/
/*

Dvc_SearchStatistics::Dvc_SearchStatistics() :
	initial_lb(Dvc_Globals::NEG_INFTY),
	initial_ub(Dvc_Globals::POS_INFTY),
	final_lb(Dvc_Globals::NEG_INFTY),
	final_ub(Dvc_Globals::POS_INFTY),
	time_search(0),
	time_path(0),
	time_backup(0),
	time_node_expansion(0),
	num_policy_nodes(0),
	num_tree_nodes(0),
	num_expanded_nodes(0),
	num_tree_particles(0),
	num_particles_before_search(0),
	num_particles_after_search(0),
	num_trials(0),
	longest_trial_length(0) {
}

ostream& operator<<(ostream& os, const Dvc_SearchStatistics& statistics) {
	os << "Initial bounds: (" << statistics.initial_lb << ", "
		<< statistics.initial_ub << ")" << endl;
	os << "Final bounds: (" << statistics.final_lb << ", "
		<< statistics.final_ub << ")" << endl;
	os << "Time (CPU s): path / expansion / backup / total = "
		<< statistics.time_path << " / " << statistics.time_node_expansion
		<< " / " << statistics.time_backup << " / " << statistics.time_search
		<< endl;
	os << "Trials: no. / max length = " << statistics.num_trials << " / "
		<< statistics.longest_trial_length << endl;
	os << "# nodes: expanded / total / policy = "
		<< statistics.num_expanded_nodes << " / " << statistics.num_tree_nodes
		<< " / " << statistics.num_policy_nodes << endl;
	os << "# particles: initial / final / tree = "
		<< statistics.num_particles_before_search << " / "
		<< statistics.num_particles_after_search << " / "
		<< statistics.num_tree_particles; // << endl;
	return os;
}

 =============================================================================
 * GPUSolver class
 * =============================================================================

GPUSolver::GPUSolver(const Dvc_DSPOMDP* model, Dvc_Belief* belief) :
	model_(model),
	belief_(belief),
	history_(Dvc_History()) {
}

GPUSolver::~GPUSolver() {
}

void GPUSolver::Update(int action, OBS_TYPE obs) {
	double start = get_time_second();

	belief_->Update(action, obs);
	history_.Add(action, obs);

	logi << "[GPUSolver::Update] Updated belief, history and root with action "
		<< action << ", observation " << obs
		<< " in " << (get_time_second() - start) << "s" << endl;
}

void GPUSolver::belief(Dvc_Belief* b) {
	belief_ = b;
	history_.Truncate(0);
}

Dvc_Belief* GPUSolver::belief() {
	return belief_;
}
*/

} // namespace despot
