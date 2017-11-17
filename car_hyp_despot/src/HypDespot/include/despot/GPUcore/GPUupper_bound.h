#ifndef GPUUPPER_BOUND_H
#define GPUUPPER_BOUND_H

#include <vector>
#include <cassert>

#include <despot/GPUrandom_streams.h>
#include <despot/GPUcore/GPUhistory.h>

#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class Dvc_State;
class Dvc_StateIndexer;
class Dvc_DSPOMDP;
class Dvc_Belief;
class Dvc_MDP;
struct Dvc_ValuedAction;

/* =============================================================================
 * Dvc_ScenarioUpperBound class
 * =============================================================================*/

class Dvc_ScenarioUpperBound {
public:
	/*HOST Dvc_ScenarioUpperBound();
	HOST virtual ~Dvc_ScenarioUpperBound();

	HOST virtual void Init(const Dvc_RandomStreams& streams);

	HOST virtual double Value(const std::vector<Dvc_State*>& particles,
		Dvc_RandomStreams& streams, Dvc_History& history) const = 0;*/
};

/* =============================================================================
 * Dvc_ParticleUpperBound class
 * =============================================================================*/

class Dvc_ParticleUpperBound : public Dvc_ScenarioUpperBound {
public:
	/*HOST Dvc_ParticleUpperBound();
	HOST virtual ~Dvc_ParticleUpperBound();

	*
	 * Returns an upper bound to the maximum total discounted reward over an
	 * infinite horizon for the (unweighted) particle.

	HOST virtual double Value(const Dvc_State& state) const = 0;

	HOST virtual double Value(const std::vector<Dvc_State*>& particles,
		Dvc_RandomStreams& streams, Dvc_History& history) const;*/
};

/* =============================================================================
 * Dvc_TrivialParticleUpperBound class
 * =============================================================================*/

class Dvc_TrivialParticleUpperBound: public Dvc_ParticleUpperBound {
protected:
	const Dvc_DSPOMDP* model_;
public:
	/*HOST Dvc_TrivialParticleUpperBound(const Dvc_DSPOMDP* model);
	HOST virtual ~Dvc_TrivialParticleUpperBound();

	HOST double Value(const Dvc_State& state) const;

	HOST virtual double Value(const std::vector<Dvc_State*>& particles,
		Dvc_RandomStreams& streams, Dvc_History& history) const;*/
};

/* =============================================================================
 * Dvc_LookaheadUpperBound class
 * =============================================================================*/

class Dvc_LookaheadUpperBound: public Dvc_ScenarioUpperBound {
protected:
	const Dvc_DSPOMDP* model_;
	const Dvc_StateIndexer& indexer_;
	std::vector<std::vector<std::vector<double> > > bounds_;
	Dvc_ParticleUpperBound* particle_upper_bound_;

public:
	/*HOST Dvc_LookaheadUpperBound(const Dvc_DSPOMDP* model, const Dvc_StateIndexer& indexer,
		Dvc_ParticleUpperBound* bound);

	HOST virtual void Init(const Dvc_RandomStreams& streams);

	HOST double Value(const std::vector<Dvc_State*>& particles,
		Dvc_RandomStreams& streams, Dvc_History& history) const;*/
};

/* =============================================================================
 * Dvc_BeliefUpperBound class
 * =============================================================================*/

class Dvc_BeliefUpperBound {
public:
	/*HOST Dvc_BeliefUpperBound();
	HOST virtual ~Dvc_BeliefUpperBound();

	HOST virtual double Value(const Dvc_Belief* belief) const = 0;*/
};

/* =============================================================================
 * Dvc_TrivialBeliefUpperBound class
 * =============================================================================*/

class Dvc_TrivialBeliefUpperBound: public Dvc_BeliefUpperBound {
protected:
	const Dvc_DSPOMDP* model_;
public:
	/*HOST Dvc_TrivialBeliefUpperBound(const Dvc_DSPOMDP* model);

	HOST double Value(const Dvc_Belief* belief) const;*/
};

/* =============================================================================
 * Dvc_MDPUpperBound class
 * =============================================================================*/

class Dvc_MDPUpperBound: public Dvc_ParticleUpperBound, public Dvc_BeliefUpperBound {
protected:
	const Dvc_MDP* model_;
	const Dvc_StateIndexer& indexer_;
	std::vector<Dvc_ValuedAction> policy_;

public:
	/*HOST Dvc_MDPUpperBound(const Dvc_MDP* model, const Dvc_StateIndexer& indexer);

  // shut off "hides overloaded virtual function" warning
  using Dvc_ParticleUpperBound::Value;
    HOST double Value(const Dvc_State& state) const;

    HOST double Value(const Dvc_Belief* belief) const;*/
};

} // namespace despot

#endif
