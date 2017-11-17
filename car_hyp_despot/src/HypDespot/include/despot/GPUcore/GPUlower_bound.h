#ifndef GPULOWER_BOUND_H
#define GPULOWER_BOUND_H

#include <vector>
#include <despot/GPUrandom_streams.h>
#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUcore/GPUsolver.h>

#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class Dvc_State;
class Dvc_DSPOMDP;
class Dvc_VNode;

/* =============================================================================
 * Dvc_ScenarioLowerBound class
 * =============================================================================*/

/**
 * Interface for an algorithm computing a lower bound for the maximum total
 * discounted reward over obtainable by a policy on a set of weighted scenarios.
 * The horizon is infinite. The first action that need to be followed to obtain
 * the bound is also returned.
 */
class Dvc_ScenarioLowerBound: public GPUSolver {
public:
	//DEVICE Dvc_ScenarioLowerBound(const Dvc_DSPOMDP* model=NULL, Dvc_Belief* belief = NULL);

	DEVICE virtual void Init(const Dvc_RandomStreams& streams);

	//DEVICE virtual Dvc_ValuedAction Search();
	DEVICE virtual void Learn(Dvc_VNode* tree);
	DEVICE virtual void Reset();

	/**
	 * Returns a lower bound for the maximum total discounted reward obtainable
	 * by a policy on a set of weighted scenarios. The horizon is infinite. The
	 * first action that need to be followed to obtain the bound is also
	 * returned.
	 *
	 * @param particles Particles in the scenarios.
	 * @param streams Random numbers attached to the scenarios.
	 * @param history Current action-observation history.
	 * @return (a, v), where v is the lower bound and a is the first action needed
	 * to obtain the lower bound.
	 */
	//DEVICE virtual Dvc_ValuedAction Value(int scenarioID, Dvc_State * particles,
	//	Dvc_RandomStreams& streams, Dvc_History& history) = 0;
};

/* =============================================================================
 * Dvc_POMCPScenarioLowerBound class
 * =============================================================================*/

//class Dvc_POMCPPrior;
/*
class Dvc_POMCPScenarioLowerBound: public Dvc_ScenarioLowerBound {
private:
	double explore_constant_;
	Dvc_POMCPPrior* prior_;

protected:
	double Simulate(Dvc_State* particle, Dvc_RandomStreams& streams, Dvc_VNode* vnode,
		Dvc_History& history) const;
	double Rollout(Dvc_State* particle, Dvc_RandomStreams& streams, int depth,
		Dvc_History& history) const;
	Dvc_VNode* CreateVNode(const Dvc_History& history, int depth) const;

public:
	Dvc_POMCPScenarioLowerBound(const Dvc_DSPOMDP* model, Dvc_POMCPPrior* prior,
		Dvc_Belief* belief = NULL);

	Dvc_ValuedAction Value(const std::vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const;
};
*/

/* =============================================================================
 * Dvc_ParticleLowerBound class
 * =============================================================================*/

/**
 * Interface for an algorithm computing a lower bound for maximum total
 * discounted reward obtainable by a policy on a set of weighted scenarios with
 * only the particles given. The horizon is inifnite. The first action that need
 * to be followed to obtain the bound is also returned.
 */
class Dvc_ParticleLowerBound : public Dvc_ScenarioLowerBound {
public:
	//DEVICE Dvc_ParticleLowerBound(const Dvc_DSPOMDP* model=NULL, Dvc_Belief* belief = NULL);

	/**
	 * Returns a lower bound for the maximum total discounted reward obtainable
	 * by a policy on a set of particles. The horizon is infinite. The horizon is
	 * inifnite. The first action that need to be followed to obtain the bound is
	 * also returned.
	 */
	//DEVICE Dvc_ValuedAction Value(int scenarioID, Dvc_State * particles);

	//DEVICE Dvc_ValuedAction Value(int scenarioID, Dvc_State * particles,
	//		Dvc_RandomStreams& streams, Dvc_History& history);
};

/* =============================================================================
 * Dvc_TrivialParticleLowerBound class
 * =============================================================================*/

class Dvc_TrivialParticleLowerBound: public Dvc_ParticleLowerBound {
public:
	//DEVICE Dvc_TrivialParticleLowerBound(const Dvc_DSPOMDP* model=NULL);

public:
	DEVICE static Dvc_ValuedAction Value(int scenarioID, Dvc_State * particles);
};

/* =============================================================================
 * Dvc_BeliefLowerBound class
 * =============================================================================*/

/**
 * Interface for an algorithm used to compute a lower bound for the infinite
 * horizon reward that can be obtained by the optimal policy on a belief.
 */
class Dvc_BeliefLowerBound: public GPUSolver {
public:
	DEVICE Dvc_BeliefLowerBound(const Dvc_DSPOMDP* model=NULL, Dvc_Belief* belief = NULL);

	//DEVICE virtual Dvc_ValuedAction Search();
	DEVICE virtual void Learn(Dvc_VNode* tree);

	DEVICE virtual Dvc_ValuedAction Value(const Dvc_Belief* belief) const = 0;
};

/* =============================================================================
 * Dvc_TrivialBeliefLowerBound class
 * =============================================================================*/

class Dvc_TrivialBeliefLowerBound: public Dvc_BeliefLowerBound {
public:
	DEVICE Dvc_TrivialBeliefLowerBound(const Dvc_DSPOMDP* model= NULL, Dvc_Belief* belief = NULL);

	//DEVICE virtual Dvc_ValuedAction Value(const Dvc_Belief* belief) const;
};

extern DEVICE Dvc_ValuedAction (*DvcParticleLowerBound_Value_) (int, Dvc_State *);
extern DEVICE Dvc_ValuedAction (*DvcModelGetMinRewardAction_)();
} // namespace despot

#endif
