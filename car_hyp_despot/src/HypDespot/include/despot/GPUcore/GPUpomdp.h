#ifndef GPUPOMDP_H
#define GPUPOMDP_H

#include <despot/GPUcore/GPUglobals.h>
#include <despot/GPUcore/GPUbelief.h>
//#include <despot/GPUrandom_streams.h>
#include <despot/GPUcore/GPUhistory.h>
//#include <despot/GPUcore/GPUlower_bound.h>
#include <despot/GPUcore/GPUpolicy.h>
//#include <despot/GPUcore/GPUupper_bound.h>
#include <despot/GPUutil/GPUmemorypool.h>
//#include <despot/GPUutil/GPUseeds.h>
//#include <despot/util/util.h>
#include <despot/GPUcore/CudaInclude.h>

#include <despot/core/pomdp.h>

namespace despot {

/* =============================================================================
 * Dvc_State class
 * =============================================================================*/
/**
 * Base state class.
 */

class Dvc_State/*: public Dvc_MemoryObject */ {
public:
	bool allocated_;
public:
	int state_id;
	int scenario_id;
	float weight;

	DEVICE Dvc_State();
	DEVICE Dvc_State(int _state_id, double weight);
	DEVICE virtual ~Dvc_State();

	//friend DEVICE std::ostream& operator<<(std::ostream& os, const Dvc_State& state);

	DEVICE std::string text() const;

	DEVICE static double Weight(int scenarioID, Dvc_State * particles);

	DEVICE Dvc_State* operator()(int state_id, double weight) {
		this->state_id = state_id;
		this->weight = weight;
		return this;
	}
	//HOST virtual void assign(State* host_state);
};

/* =============================================================================
 * Dvc_StateIndexer class
 * =============================================================================*/
/**
 * Interface for a mapping between states and indices.
 */
class Dvc_StateIndexer {
public:
	/*virtual DEVICE ~Dvc_StateIndexer();

	virtual DEVICE int NumStates() const = 0;
	virtual DEVICE int GetIndex(const Dvc_State* state) const = 0;
	virtual DEVICE const Dvc_State* GetState(int index) const = 0;*/
};

/* =============================================================================
 * Dvc_StatePolicy class
 * =============================================================================*/
/**
 * Interface for a mapping from states to actions.
 */
/*class Dvc_StatePolicy {
public:
	virtual DEVICE ~Dvc_StatePolicy();
	virtual DEVICE int GetAction(const Dvc_State& state) const = 0;
};*/

/* =============================================================================
 * MMAPinferencer class
 * =============================================================================*/
/**
 * Interface for computing marginal MAP state from a set of particles.
 */
/*class Dvc_MMAPInferencer {
public:
	virtual DEVICE ~Dvc_MMAPInferencer();

	virtual DEVICE const Dvc_State* GetMMAP(const std::vector<Dvc_State*>& particles) const = 0;
};*/

//class Dvc_POMCPPrior;

/* =============================================================================
 * Dvc_DSPOMDP class
 * =============================================================================*/
/**
 * Interface for a deterministic simulative model for POMDP.
 */
class Dvc_DSPOMDP {
public:
	DEVICE Dvc_DSPOMDP();

	DEVICE virtual ~Dvc_DSPOMDP();

	HOST void assign(const DSPOMDP* hst_model);
	/* ========================================================================
	 * Deterministic simulative model and related functions
	 * ========================================================================*/
	/**
	 * Determistic simulative model for POMDP.
	 */
	virtual DEVICE bool Step(Dvc_State& state, double random_num, int action,
		double& reward, OBS_TYPE& obs) const = 0;

	/**
	 * Override this to get speedup for LookaheadUpperBound.
	 */
	virtual DEVICE bool Step(Dvc_State& state, double random_num, int action,
		double& reward) const;

	/**
	 * Simulative model for POMDP.
	 */
	//virtual DEVICE bool Step(Dvc_State& state, int action, double& reward,
	//	OBS_TYPE& obs) const;

	/* ========================================================================
	 * Action
	 * ========================================================================*/
	/**
	 * Returns number of actions.
	 */
	virtual DEVICE int NumActions() const = 0;

	/* ========================================================================
	 * Functions related to beliefs and starting states.
	 * ========================================================================*/
	/**
	 * Returns the observation probability.
	 */
	virtual DEVICE double ObsProb(OBS_TYPE obs, const Dvc_State& state,
		int action) const = 0;

	/**
	 * Returns a starting state.
	 */
	//virtual DEVICE Dvc_State* CreateStartState(std::string type = "DEFAULT") const = 0;

	/**
	 * Returns the initial belief.
	 */
	//virtual DEVICE Dvc_Belief* InitialBelief(const Dvc_State* start,
	//	std::string type = "DEFAULT") const = 0;

	/* ========================================================================
	 * Bound-related functions.
	 * ========================================================================*/
	/**
	 * Returns the maximum reward.
	 */
	//virtual DEVICE double GetMaxReward() const = 0;
	//virtual HOST Dvc_ParticleUpperBound* CreateParticleUpperBound(std::string name = "DEFAULT") const;
	//virtual HOST Dvc_ScenarioUpperBound* CreateScenarioUpperBound(std::string name = "DEFAULT",
	//	std::string particle_bound_name = "DEFAULT") const;

	/**
	 * Returns (a, v), where a is an action with largest minimum reward when it is
	 * executed, and v is its minimum reward, that is, a = \max_{a'} \min_{s}
	 * R(a', s), and v = \min_{s} R(a, s).
	 */
	//virtual DEVICE Dvc_ValuedAction GetMinRewardAction() const = 0;
	//virtual HOST Dvc_ParticleLowerBound* CreateParticleLowerBound(std::string name = "DEFAULT") const;
	//virtual HOST Dvc_ScenarioLowerBound* CreateScenarioLowerBound(std::string bound_name = "DEFAULT",
	///	std::string particle_bound_name = "DEFAULT") const;

	//virtual Dvc_POMCPPrior* CreatePOMCPPrior(std::string name = "DEFAULT") const;

	/* ========================================================================
	 * Display
	 * ========================================================================*/
	/**
	 * Prints a state.
	 */
	//virtual DEVICE void PrintState(const Dvc_State& state, std::ostream& out = std::cout) const = 0;

	/**
	 * Prints an observation.
	 */
	//virtual DEVICE void PrintObs(const Dvc_State& state, OBS_TYPE obs,
	//	std::ostream& out = std::cout) const = 0;

	/**
	 * Prints an action.
	 */
	//virtual DEVICE void PrintAction(int action, std::ostream& out = std::cout) const = 0;

	/**
	 * Prints a belief.
	 */
	//virtual DEVICE void PrintBelief(const Dvc_Belief& belief,
	//	std::ostream& out = std::cout) const = 0;

	/* ========================================================================
	 * Memory management.
	 * ========================================================================*/
	/**
	 * Allocate a state.
	 */
	virtual DEVICE Dvc_State* Allocate(int state_id = -1, double weight = 0) const = 0;

	/**
	 * Returns a copy of the state.
	 */
	virtual DEVICE Dvc_State* Copy(const Dvc_State* state) const = 0;

	/**
	 * Returns a copy of the particle.
	 */
	virtual DEVICE void Free(Dvc_State* state) const = 0;

	/**
	 * Returns a copy of the particles.
	 */
	DEVICE Dvc_State** Copy(const Dvc_State**& particles, int numParticles) const;

	/**
	 * Returns number of allocated particles.
	 */
	//virtual DEVICE int NumActiveParticles() const = 0;

	/**
	 * Returns a copy of this model.
	 */
	//inline virtual DEVICE Dvc_DSPOMDP* MakeCopy() const {
	//	return NULL;
	//}
};

/* =============================================================================
 * Dvc_BeliefMDP class
 * =============================================================================*/
/**
 * The Dvc_BeliefMDP class provides an interface for the belief MDP, which is
 * commonly used in belief tree search algorithms.
 *
 * @see AEMS
 */

/*
class Dvc_BeliefMDP: public Dvc_DSPOMDP {
public:
	DEVICE Dvc_BeliefMDP();
	virtual DEVICE ~Dvc_BeliefMDP();

	virtual DEVICE Dvc_BeliefLowerBound* CreateBeliefLowerBound(std::string name) const;
	virtual DEVICE Dvc_BeliefUpperBound* CreateBeliefUpperBound(std::string name) const;

  *
   * Transition function for the belief MDP.

	virtual DEVICE Dvc_Belief* Tau(const Dvc_Belief* belief, int action,
		OBS_TYPE obs) const = 0;

  *
   * Observation function for the belief MDP.

	virtual DEVICE void Observe(const Dvc_Belief* belief, int action,
		std::map<OBS_TYPE, double>& obss) const = 0;

  *
   * Reward function for the belief MDP.

	virtual DEVICE double StepReward(const Dvc_Belief* belief, int action) const = 0;
};
*/
extern DEVICE Dvc_State* (*DvcModelGet_)(Dvc_State* , int );
extern Dvc_State** Dvc_stepped_particles_all_a;

} // namespace despot

#endif
