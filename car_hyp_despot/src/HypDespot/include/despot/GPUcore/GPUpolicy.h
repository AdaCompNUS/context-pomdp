#ifndef GPUPOLICY_H
#define GPUPOLICY_H

#include <vector>

#include <despot/GPUrandom_streams.h>
#include <despot/GPUcore/GPUlower_bound.h>
#include <despot/GPUutil/GPUrandom.h>
#include <despot/GPUcore/GPUhistory.h>

#include <string.h>
#include <queue>
#include <vector>
#include <stdlib.h>
#include <despot/GPUcore/GPUglobals.h>
#include <despot/GPUcore/GPUpomdp.h>

#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class Dvc_State;
class Dvc_StateIndexer;
class Dvc_StatePolicy;
class Dvc_DSPOMDP;
class Dvc_MMAPInferencer;

/* =============================================================================
 * Dvc_ValuedAction struct
 * =============================================================================*/

struct Dvc_ValuedAction {
	int action;
	float value;

	DEVICE Dvc_ValuedAction();
	DEVICE Dvc_ValuedAction(int _action, float _value);

	DEVICE Dvc_ValuedAction& operator=(const Dvc_ValuedAction& other) // copy assignment
	{
	    if (this != &other) { // self-assignment check expected
	    	action=other.action;
	    	value=other.value;
	    }
	    return *this;
	}
	//friend HOST std::ostream& operator<<(std::ostream& os, const Dvc_ValuedAction& va);
};

/* =============================================================================
 * Dvc_Policy class
 * =============================================================================*/

class Dvc_Policy/*: public Dvc_ScenarioLowerBound */{
private:
	//static /*mutable*/ int initial_depth_;
	//Dvc_ParticleLowerBound* particle_lower_bound_;

	HOST void InitLocalHistory(int NumParticles/*,Dvc_History* history*/);
	HOST void ClearLocalHistory(int NumParticles);
	/*DEVICE Dvc_ValuedAction RecursiveValue(int NumParticles, const Dvc_State*& particles,
		const int*& particleIDs,Dvc_RandomStreams& streams, Dvc_History& history, Dvc_Config* config);*/

public:

	/*DEVICE Dvc_Policy(const Dvc_DSPOMDP* model, Dvc_ParticleLowerBound* particle_lower_bound,
		Dvc_Belief* belief = NULL);
	virtual DEVICE ~Dvc_Policy();*/

	//DEVICE void Reset();
	/*DEVICE virtual int Action(const std::vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const = 0;*/

	//DEVICE Dvc_ParticleLowerBound* particle_lower_bound() const;

	DEVICE static Dvc_ValuedAction Value(
		Dvc_State* particles,
		Dvc_RandomStreams& streams,
		Dvc_History& history);

	//virtual DEVICE Dvc_ValuedAction Search();
};
/* =============================================================================
 * Dvc_BlindPolicy class
 * =============================================================================*/

class Dvc_BlindPolicy: public Dvc_Policy {
private:
	int action_;

public:
	/*DEVICE Dvc_BlindPolicy(const Dvc_DSPOMDP* model, int action, Dvc_ParticleLowerBound*
		particle_lower_bound, Dvc_Belief* belief = NULL);

	DEVICE int Action(const std::vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const;

	DEVICE Dvc_ValuedAction Search();
	DEVICE void Update(int action, OBS_TYPE obs);*/
};

/* =============================================================================
 * Dvc_RandomPolicy class
 * =============================================================================*/

class Dvc_RandomPolicy: public Dvc_Policy {
private:
	//static double* action_probs_;
	//static int num_actions_;

public:

	//HOST void PassActionFunc();
	DEVICE Dvc_RandomPolicy();
	DEVICE static void Init(int num_actions);
	DEVICE static int Action(int scenarioID, const Dvc_State* particles,
			Dvc_RandomStreams& streams,
			Dvc_History& history);
	/*
	DEVICE Dvc_RandomPolicy(const Dvc_DSPOMDP* model, Dvc_ParticleLowerBound* Dvc_ParticleLowerBound,
		Dvc_Belief* belief = NULL);
	DEVICE Dvc_RandomPolicy(const Dvc_DSPOMDP* model, const std::vector<double>& action_probs,
		Dvc_ParticleLowerBound* Dvc_ParticleLowerBound,
		Dvc_Belief* belief = NULL);



	DEVICE Dvc_ValuedAction Search();
	DEVICE void Update(int action, OBS_TYPE obs);*/
};

/* =============================================================================
 * Dvc_ModeStatePolicy class
 * =============================================================================*/

class Dvc_ModeStatePolicy: public Dvc_Policy {
private:
	/*const Dvc_StateIndexer& indexer_;
	const Dvc_StatePolicy& policy_;
	mutable std::vector<double> state_probs_;

public:
	HOST Dvc_ModeStatePolicy(const Dvc_DSPOMDP* model, const Dvc_StateIndexer& indexer,
		const Dvc_StatePolicy& policy, Dvc_ParticleLowerBound* particle_lower_bound,
		Dvc_Belief* belief = NULL);

	HOST int Action(const std::vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const;*/
};

/* =============================================================================
 * Dvc_MMAPStatePolicy class
 * =============================================================================*/

class Dvc_MMAPStatePolicy: public Dvc_Policy { // Marginal MAP state policy
private:
	/*const Dvc_MMAPInferencer& inferencer_;
	const Dvc_StatePolicy& policy_;

public:
	HOST Dvc_MMAPStatePolicy(const Dvc_DSPOMDP* model, const Dvc_MMAPInferencer& inferencer,
		const Dvc_StatePolicy& policy, Dvc_ParticleLowerBound* particle_lower_bound,
		Dvc_Belief* belief = NULL);

	HOST int Action(const std::vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const;*/
};

/* =============================================================================
 * Dvc_MajorityActionPolicy class
 * =============================================================================*/

class Dvc_MajorityActionPolicy: public Dvc_Policy {
private:
	/*const Dvc_StatePolicy& policy_;

public:
	HOST Dvc_MajorityActionPolicy(const Dvc_DSPOMDP* model, const Dvc_StatePolicy& policy,
		Dvc_ParticleLowerBound* particle_lower_bound, Dvc_Belief* belief = NULL);

	HOST int Action(const std::vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const;*/
};
/*Dvc_particles_copy is used for rollouts*/
//extern DEVICE Dvc_State* Dvc_particles_copy;
extern OBS_TYPE** Dvc_observations_long;
extern OBS_TYPE** Dvc_obs_all_a_and_p;
extern OBS_TYPE** Hst_obs_all_a_and_p;
extern int** Dvc_obs_int_all_a_and_p;
extern int** Hst_obs_int_all_a_and_p;
extern bool** Dvc_term_all_a_and_p;
extern bool** Hst_term_all_a_and_p;
//extern DEVICE Dvc_History* Dvc_ParticleHistory;
extern DEVICE Dvc_Random* Dvc_random;
extern DEVICE int* Dvc_initial_depth;

//extern DEVICE Dvc_ValuedAction (*ParticleLowerBound_Value_) (int, const Dvc_State *&, const int*&);
//extern DEVICE int (*DvcPolicyValue_)(int,const Dvc_State* ,const int*, Dvc_RandomStreams&, Dvc_History&);
extern DEVICE Dvc_State* (*DvcModelAlloc_)( int num);
extern DEVICE void (*DvcModelCopyNoAlloc_)(Dvc_State*,const Dvc_State*, int pos, bool offset_des);
extern DEVICE void (*DvcModelCopyToShared_)(Dvc_State*,const Dvc_State*, int pos, bool offset_des);
extern DEVICE bool (*DvcModelStep_)(Dvc_State&, float, int, float&, OBS_TYPE&);
//for more complex observations
extern DEVICE bool (*DvcModelStepIntObs_)(Dvc_State&, float, int, float&, int*);
extern DEVICE int (*DvcPolicyAction_)(int,const Dvc_State* ,Dvc_RandomStreams&, Dvc_History&);

} // namespace despot

#endif
