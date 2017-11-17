#include <despot/GPUcore/GPUpomdp.h>
//#include <despot/GPUcore/GPUpolicy.h>
//#include <despot/GPUcore/GPUlower_bound.h>
//#include <despot/GPUcore/GPUupper_bound.h>
//#include <despot/GPUutil/GPUrandom.h>
//#include <despot/solver/pomcp.h>

using namespace std;

namespace despot {
DEVICE Dvc_State* (*DvcModelGet_)(Dvc_State* , int )=NULL;
Dvc_State** Dvc_stepped_particles_all_a=NULL;

/* =============================================================================
 * Dvc_State class
 * =============================================================================*/

/*ostream& operator<<(ostream& os, const Dvc_State& state) {
	os << "(state_id = " << state.state_id << ", weight = " << state.weight
		<< ", text = " << (&state)->text() << ")";
	return os;
}*/

DEVICE Dvc_State::Dvc_State() :
	state_id(-1) {
}

DEVICE Dvc_State::Dvc_State(int _state_id, double _weight) :
	state_id(_state_id),
	weight(_weight) {
}

DEVICE Dvc_State::~Dvc_State() {
}

/*DEVICE string Dvc_State::text() const {
	return "AbstractState";
}*/

DEVICE double Dvc_State::Weight(int scenarioID, Dvc_State * particles) {
	double weight = 0;
	for (int i = 0; i < /*numParticle*/1; i++)
	{
		//int scenarioID=particleIDs[i];
		weight += DvcModelGet_(particles,scenarioID)->weight;
	}
	return weight;
}

/*HOST void Dvc_State::assign(State* host_state)
{
	HANDLE_ERROR(cudaMemcpy(this, host_state, sizeof(Dvc_State),cudaMemcpyHostToDevice));
}*/

/* =============================================================================
 * Dvc_StateIndexer class
 * =============================================================================*/
/*
Dvc_StateIndexer::~Dvc_StateIndexer() {
}
*/

/* =============================================================================
 * Dvc_StatePolicy class
 * =============================================================================*/
/*Dvc_StatePolicy::~Dvc_StatePolicy() {
}*/

/* =============================================================================
 * MMAPinferencer class
 * =============================================================================*/
/*
Dvc_MMAPInferencer::~Dvc_MMAPInferencer() {
}
*/

/* =============================================================================
 * Dvc_DSPOMDP class
 * =============================================================================*/

DEVICE Dvc_DSPOMDP::Dvc_DSPOMDP() {
}

DEVICE Dvc_DSPOMDP::~Dvc_DSPOMDP() {
}

HOST void Dvc_DSPOMDP::assign(const DSPOMDP* hst_model)
{//no content need to be passed from host to device

}

/*DEVICE bool Dvc_DSPOMDP::Step(Dvc_State& state, int action, double& reward,
	OBS_TYPE& obs) const {
	return Step(state, Dvc_Random::RANDOM.NextDouble(), action, reward, obs);
}*/

DEVICE bool Dvc_DSPOMDP::Step(Dvc_State& state, double random_num, int action,
	double& reward) const {
	OBS_TYPE obs;
	return Step(state, random_num, action, reward, obs);
}


/*HOST Dvc_ParticleUpperBound* Dvc_DSPOMDP::CreateParticleUpperBound(string name) const {
	if (name == "TRIVIAL" || name == "DEFAULT") {
		return new Dvc_TrivialParticleUpperBound(this);
	} else {
		//cerr << "Unsupported particle upper bound: " << name << endl;
		exit(1);
	}
}*/

/*HOST Dvc_ScenarioUpperBound* Dvc_DSPOMDP::CreateScenarioUpperBound(string name,
	string particle_bound_name) const {
	if (name == "TRIVIAL" || name == "DEFAULT") {
		return new Dvc_TrivialParticleUpperBound(this);
	} else {
		cerr << "Unsupported scenario upper bound: " << name << endl;
		exit(1);
		return NULL;
	}
}*/

/*HOST Dvc_ParticleLowerBound* Dvc_DSPOMDP::CreateParticleLowerBound(string name) const {
	if (name == "TRIVIAL" || name == "DEFAULT") {
		return new Dvc_TrivialParticleLowerBound(this);
	} else {
		//cerr << "Unsupported particle lower bound: " << name << endl;
		exit(1);
		return NULL;
	}
}*/

/*HOST Dvc_ScenarioLowerBound* Dvc_DSPOMDP::CreateScenarioLowerBound(string name, string
	particle_bound_name) const {
	if (name == "TRIVIAL" || name == "DEFAULT") {
		return new Dvc_TrivialParticleLowerBound(this);
	} else if (name == "RANDOM") {
		return new Dvc_RandomPolicy(this, CreateParticleLowerBound(particle_bound_name));
	} else {
		cerr << "Unsupported lower bound algorithm: " << name << endl;
		exit(1);
		return NULL;
	}
}*/


/*Dvc_POMCPPrior* Dvc_DSPOMDP::CreatePOMCPPrior(string name) const {
	if (name == "UNIFORM" || name == "DEFAULT") {
		return new UniformPOMCPPrior(this);
	} else {
		cerr << "Unsupported POMCP prior: " << name << endl;
		exit(1);
		return NULL;
	}
}*/

DEVICE Dvc_State** Dvc_DSPOMDP::Copy(const Dvc_State**& particles, int numParticles) const {
	Dvc_State** copy;
	copy=(Dvc_State**)malloc(numParticles*sizeof(Dvc_State*));
	for (int i = 0; i < numParticles; i++)
	{
		copy[i]=Copy(particles[i]);
	}
	return copy;
}

/* =============================================================================
 * Dvc_BeliefMDP classs
 * =============================================================================*/
/*

Dvc_BeliefMDP::Dvc_BeliefMDP() {
}

Dvc_BeliefMDP::~Dvc_BeliefMDP() {
}

Dvc_BeliefLowerBound* Dvc_BeliefMDP::CreateBeliefLowerBound(string name) const {
	if (name == "TRIVIAL" || name == "DEFAULT") {
		return new Dvc_TrivialBeliefLowerBound(this);
	} else {
		cerr << "Unsupported belief lower bound: " << name << endl;
		exit(1);
		return NULL;
	}
}

Dvc_BeliefUpperBound* Dvc_BeliefMDP::CreateBeliefUpperBound(string name) const {
	if (name == "TRIVIAL" || name == "DEFAULT") {
		return new Dvc_TrivialBeliefUpperBound(this);
	} else {
		cerr << "Unsupported belief upper bound: " << name << endl;
		exit(1);
		return NULL;
	}
}
*/

} // namespace despot
