#ifndef POLICY_GRAPH_H
#define POLICY_GRAPH_H

#include <despot/random_streams.h>
#include <despot/core/lower_bound.h>
#include <despot/util/random.h>
#include <despot/core/history.h>

#include <string.h>
#include <queue>
#include <vector>
#include <stdlib.h>
#include <despot/core/globals.h>
#include <despot/core/pomdp.h>

namespace despot {

/* =============================================================================
 * PolicyGraph class
 * =============================================================================*/
#define FIX_SCENARIO 0

extern bool Load_Graph;


class PolicyGraph: public ScenarioLowerBound {
protected:

	mutable int initial_depth_;
	ParticleLowerBound* particle_lower_bound_;


public:
	mutable int graph_size_;
	mutable int num_edges_per_node_;
	std::vector<int> action_nodes_;
	mutable std::map<OBS_TYPE, std::vector<int> > obs_edges_;

	mutable int current_node_;
	mutable int entry_node_;

public:
	PolicyGraph(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound,
		Belief* belief = NULL);
	virtual ~PolicyGraph();

	void Reset();

	ParticleLowerBound* particle_lower_bound() const;

	virtual void ConstructGraph(int size, int branch)=0;
	void ClearGraph();

	void SetEntry(int node)
	{entry_node_=node;}

	virtual ValuedAction Value(const std::vector<State*>& particles, RandomStreams& streams,
		History& history) const;

	virtual void ExportGraph(std::ostream& fout);
	virtual void ImportGraph(std::ifstream& fin, int size, int branch);
};

/* =============================================================================
 * RandomPolicyGraph class
 * =============================================================================*/

class RandomPolicyGraph: public PolicyGraph {
private:
	std::vector<double> action_probs_;

public:
	RandomPolicyGraph(const DSPOMDP* model, ParticleLowerBound* ParticleLowerBound,
		Belief* belief = NULL);
	virtual ~RandomPolicyGraph();
	virtual void ConstructGraph(int size, int branch);

};

}// namespace despot

#endif //PPOLICY_GRAPH_H
