#ifndef CUSTOM_PARTICLE_BELIEF_H
#define CUSTOM_PARTICLE_BELIEF_H

#include "state.h"
#include <despot/interface/belief.h>
#include <despot/core/particle_belief.h>


class MaxLikelihoodScenario: public ParticleBelief{
public:
	MaxLikelihoodScenario(vector<State*> particles, const DSPOMDP* model,
		Belief* prior = NULL, bool split = true);

	vector<State*> SampleCustomScenarios(int num, vector<State*> particles,
	const DSPOMDP* model) const;
};

#endif
