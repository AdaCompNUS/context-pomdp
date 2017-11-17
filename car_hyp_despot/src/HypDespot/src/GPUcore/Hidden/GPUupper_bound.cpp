#include <despot/GPUcore/GPUupper_bound.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUmdp.h>

using namespace std;

namespace despot {

/* =============================================================================
 * ScenarioUpperBound
 * =============================================================================*/

/*Dvc_ScenarioUpperBound::Dvc_ScenarioUpperBound() {
}

Dvc_ScenarioUpperBound::~Dvc_ScenarioUpperBound() {
}

void Dvc_ScenarioUpperBound::Init(const Dvc_RandomStreams& streams) {
}

 =============================================================================
 * ParticleUpperBound
 * =============================================================================

Dvc_ParticleUpperBound::Dvc_ParticleUpperBound() {
}

Dvc_ParticleUpperBound::~Dvc_ParticleUpperBound() {
}

double Dvc_ParticleUpperBound::Value(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	double value = 0;
	for (int i = 0; i < particles.size(); i++) {
		Dvc_State* particle = particles[i];
		value += particle->weight * Value(*particle);
	}
	return value;
}

 =============================================================================
 * TrivialParticleUpperBound
 * =============================================================================

Dvc_TrivialParticleUpperBound::Dvc_TrivialParticleUpperBound(const Dvc_DSPOMDP* model) :
	model_(model) {
}

Dvc_TrivialParticleUpperBound::~Dvc_TrivialParticleUpperBound() {
}

double Dvc_TrivialParticleUpperBound::Value(const Dvc_State& state) const {
	return model_->GetMaxReward() / (1 - Dvc_Globals::Dvc_Discount());
}

double Dvc_TrivialParticleUpperBound::Value(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	return Dvc_State::Weight(particles) * model_->GetMaxReward() / (1 - Dvc_Globals::Dvc_Discount());
}

 =============================================================================
 * LookaheadUpperBound
 * =============================================================================

Dvc_LookaheadUpperBound::Dvc_LookaheadUpperBound(const Dvc_DSPOMDP* model,
	const Dvc_StateIndexer& indexer, Dvc_ParticleUpperBound* bound) :
	model_(model),
	indexer_(indexer),
	particle_upper_bound_(bound) {
}

void Dvc_LookaheadUpperBound::Init(const Dvc_RandomStreams& streams) {
	int num_states = indexer_.NumStates();
	int length = streams.Length();
	int num_particles = streams.NumStreams();

	SetSize(bounds_, num_particles, length + 1, num_states);

	clock_t start = clock();
	for (int p = 0; p < num_particles; p++) {
		if (p % 10 == 0)
			cerr << p << " scenarios done! ["
				<< (double(clock() - start) / CLOCKS_PER_SEC) << "s]" << endl;
		for (int t = length; t >= 0; t--) {
			if (t == length) { // base case
				for (int s = 0; s < num_states; s++) {
					bounds_[p][t][s] = particle_upper_bound_->Value(*indexer_.GetState(s));
				}
			} else { // lookahead
				for (int s = 0; s < num_states; s++) {
					double best = Dvc_Globals::NEG_INFTY;

					for (int a = 0; a < model_->NumActions(); a++) {
						double reward = 0;
						Dvc_State* copy = model_->Copy(indexer_.GetState(s));
						bool terminal = model_->Step(*copy, streams.Entry(p, t),
							a, reward);
						model_->Free(copy);
						reward += (!terminal) * Dvc_Globals::Dvc_Discount()
							* bounds_[p][t + 1][indexer_.GetIndex(copy)];

						if (reward > best)
							best = reward;
					}

					bounds_[p][t][s] = best;
				}
			}
		}
	}
}

double Dvc_LookaheadUpperBound::Value(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	double bound = 0;
	for (int i = 0; i < particles.size(); i++) {
		Dvc_State* particle = particles[i];
		bound +=
			particle->weight
				* bounds_[particle->scenario_id][streams.position()][indexer_.GetIndex(
					particle)];
	}
	return bound;
}


 =============================================================================
 * BeliefUpperBound
 * =============================================================================

Dvc_BeliefUpperBound::Dvc_BeliefUpperBound() {
}

Dvc_BeliefUpperBound::~Dvc_BeliefUpperBound() {
}

Dvc_TrivialBeliefUpperBound::Dvc_TrivialBeliefUpperBound(const Dvc_DSPOMDP* model) :
	model_(model) {
}

double Dvc_TrivialBeliefUpperBound::Value(const Dvc_Belief* belief) const {
	return model_->GetMaxReward() / (1 - Dvc_Globals::Dvc_Discount());
}

 =============================================================================
 * MDPUpperBound
 * =============================================================================

Dvc_MDPUpperBound::Dvc_MDPUpperBound(const Dvc_MDP* model,
	const Dvc_StateIndexer& indexer) :
	model_(model),
	indexer_(indexer) {
	const_cast<Dvc_MDP*>(model_)->ComputeOptimalPolicyUsingVI();
	policy_ = model_->policy();
}

double Dvc_MDPUpperBound::Value(const Dvc_State& state) const {
	return policy_[indexer_.GetIndex(&state)].value;
}

double Dvc_MDPUpperBound::Value(const Dvc_Belief* belief) const {
	const vector<Dvc_State*>& particles =
		static_cast<const Dvc_ParticleBelief*>(belief)->particles();

	double value = 0;
	for (int i = 0; i < particles.size(); i++) {
		Dvc_State* particle = particles[i];
		value += particle->weight * policy_[indexer_.GetIndex(particle)].value;
	}
	return value;
}*/

} // namespace despot
