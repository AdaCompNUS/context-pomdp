#include <despot/GPUcore/GPUlower_bound.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUnode.h>
//#include <despot/solver/pomcp.h>

using namespace std;

namespace despot {
DEVICE Dvc_ValuedAction (*DvcModelGetMinRewardAction_)()=NULL;
DEVICE Dvc_ValuedAction (*DvcParticleLowerBound_Value_) (int, Dvc_State *)=NULL;
/* =============================================================================
 * Dvc_ValuedAction class
 * =============================================================================*/

DEVICE Dvc_ValuedAction::Dvc_ValuedAction()
{
	action=-1;
	value=0;
}

DEVICE Dvc_ValuedAction::Dvc_ValuedAction(int _action, float _value)
{
	action=_action;
	value=_value ;
}

/*ostream& operator<<(ostream& os, const Dvc_ValuedAction& va) {
	os << "(" << va.action << ", " << va.value << ")";
	return os;
}*/

/* =============================================================================
 * ScenarioLowerBound class
 * =============================================================================*/

/*DEVICE Dvc_ScenarioLowerBound::Dvc_ScenarioLowerBound(const Dvc_DSPOMDP* model, Dvc_Belief* belief) :
	GPUSolver(model, belief) {
}*/

DEVICE void Dvc_ScenarioLowerBound::Init(const Dvc_RandomStreams& streams) {
}

DEVICE void Dvc_ScenarioLowerBound::Reset() {
}

/*DEVICE Dvc_ValuedAction Dvc_ScenarioLowerBound::Search() {
	Dvc_RandomStreams streams(Dvc_Globals::config.num_scenarios,
		Dvc_Globals::config.search_depth);
	std::vector<Dvc_State*> particles = belief_->Sample(Dvc_Globals::config.num_scenarios);

	Dvc_ValuedAction va = Value(particles, streams, history_);

	for (int i = 0; i < particles.size(); i++)
		model_->Free(particles[i]);

	return va;
}*/

DEVICE void Dvc_ScenarioLowerBound::Learn(Dvc_VNode* tree) {
}

/* =============================================================================
 * POMCPScenarioLowerBound class
 * =============================================================================*/

/*
Dvc_POMCPScenarioLowerBound::Dvc_POMCPScenarioLowerBound(const Dvc_DSPOMDP* model,
	Dvc_POMCPPrior* prior,
	Dvc_Belief* belief) :
	Dvc_ScenarioLowerBound(model, belief),
	prior_(prior) {
	explore_constant_ = model_->GetMaxReward()
		- model_->GetMinRewardAction().value;
}
*/

/*
Dvc_ValuedAction Dvc_POMCPScenarioLowerBound::Value(const std::vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) {
	prior_->history(history);
	Dvc_VNode* root = POMCP::CreateVNode(0, particles[0], prior_, model_);
	// Note that particles are assumed to be of equal weight
	for (int i = 0; i < particles.size(); i++) {
		Dvc_State* particle = particles[i];
		Dvc_State* copy = model_->Copy(particle);
		POMCP::Simulate(copy, streams, root, model_, prior_);
		model_->Free(copy);
	}

	Dvc_ValuedAction va = POMCP::OptimalAction(root);
	va.value *= Dvc_State::Weight(particles);
	delete root;
	return va;
}
*/

/* =============================================================================
 * ParticleLowerBound class
 * =============================================================================*/

/*DEVICE Dvc_ParticleLowerBound::Dvc_ParticleLowerBound(const Dvc_DSPOMDP* model, Dvc_Belief* belief) :
		Dvc_ScenarioLowerBound(model, belief) {
}*/

/*DEVICE Dvc_ValuedAction Dvc_ParticleLowerBound::Value(int scenarioID, Dvc_State * particles,
	Dvc_RandomStreams& streams, Dvc_History& history) {
	return DvcParticleLowerBound_Value_(scenarioID,particles);
}*/

/* =============================================================================
 * TrivialParticleLowerBound class
 * =============================================================================*/

/*DEVICE Dvc_TrivialParticleLowerBound::Dvc_TrivialParticleLowerBound(const Dvc_DSPOMDP* model) :
		Dvc_ParticleLowerBound(model) {
}*/

DEVICE Dvc_ValuedAction Dvc_TrivialParticleLowerBound::Value(
		int scenarioID, Dvc_State * particles) {
	Dvc_ValuedAction va = DvcModelGetMinRewardAction_();
	va.value *= 1.0 / (1 - Dvc_Globals::Dvc_Discount(Dvc_config));
	return va;
}

/* =============================================================================
 * BeliefLowerBound class
 * =============================================================================*/

DEVICE Dvc_BeliefLowerBound::Dvc_BeliefLowerBound(const Dvc_DSPOMDP* model, Dvc_Belief* belief)/* :*/
	/*GPUSolver(model, belief)*/ {
}

/*DEVICE Dvc_ValuedAction Dvc_BeliefLowerBound::Search() {
	return Value(belief_);
}*/

DEVICE void Dvc_BeliefLowerBound::Learn(Dvc_VNode* tree) {
}

/* =============================================================================
 * TrivialBeliefLowerBound class
 * =============================================================================*/

DEVICE Dvc_TrivialBeliefLowerBound::Dvc_TrivialBeliefLowerBound(const Dvc_DSPOMDP* model,
	Dvc_Belief* belief) :
	Dvc_BeliefLowerBound(model, belief) {
}

/*DEVICE Dvc_ValuedAction Dvc_TrivialBeliefLowerBound::Value(const Dvc_Belief* belief) {
	Dvc_ValuedAction va = model_->GetMinRewardAction();
	va.value *= 1.0 / (1 - Dvc_Globals::Dvc_Discount());
	return va;
}*/

} // namespace despot
