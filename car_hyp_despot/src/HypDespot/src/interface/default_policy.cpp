#include <despot/interface/default_policy.h>
#include <despot/interface/pomdp.h>
#include <unistd.h>
#include <despot/GPUcore/thread_globals.h>

using namespace std;

namespace despot {

/* =============================================================================
 * DefaultPolicy class
 * =============================================================================*/

DefaultPolicy::DefaultPolicy(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound) :
	ScenarioLowerBound(model),
	particle_lower_bound_(particle_lower_bound) {
	assert(particle_lower_bound_ != NULL);
}

DefaultPolicy::~DefaultPolicy() {
}

ValuedAction DefaultPolicy::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	vector<State*> copy;
	for (int i = 0; i < particles.size(); i++)
		copy.push_back(model_->Copy(particles[i]));

	initial_depth_ = history.Size();
	ValuedAction va = RecursiveValue(copy, streams, history);

	for (int i = 0; i < copy.size(); i++)
		model_->Free(copy[i]);

	return va;
}

ValuedAction DefaultPolicy::RecursiveValue(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	if (streams.Exhausted()
		|| (history.Size() - initial_depth_
			>= Globals::config.max_policy_sim_len)) {

		auto blb_value = particle_lower_bound_->Value(particles);

		if(false && initial_depth_ == 0 && particles[0]->scenario_id == 0)
			cout << "rollout blb value: " << blb_value.value << endl;

		return blb_value;
	} else {
		ACT_TYPE action = Action(particles, streams, history);
    
    // if (history.Size() == initial_depth_) 
      // cout << "roll out default action at initial depth " << action << endl;

		double value = 0;

		map<OBS_TYPE, vector<State*> > partitions;
		OBS_TYPE obs;
		double reward;
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
			bool terminal = model_->Step(*particle,
				streams.Entry(particle->scenario_id), action, reward, obs);

			if(false && initial_depth_ == 0 && particle->scenario_id == 0){
				cout << "rollout state with reward " << reward << endl;
				// model_->Reward(*particle, action);
				model_->PrintState(*particle);
			}

			value += reward * particle->weight;

			if (!terminal) {
				partitions[obs].push_back(particle);
			}
		}

	    if(DoPrintCPU) printf("action, ave_reward= %d %f\n",action,value);


		for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			history.Add(action, obs);
			streams.Advance();
			ValuedAction va = RecursiveValue(it->second, streams, history);
			value += Globals::Discount() * va.value;
			streams.Back();
			history.RemoveLast();
		}

		return ValuedAction(action, value);
	}
}

void DefaultPolicy::Reset() {
}

ParticleLowerBound* DefaultPolicy::particle_lower_bound() const {
	return particle_lower_bound_;
}

} // namespace despot
