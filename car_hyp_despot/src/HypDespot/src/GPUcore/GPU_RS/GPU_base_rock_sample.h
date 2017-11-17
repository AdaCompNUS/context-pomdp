#ifndef GPU_BASEROCKSAMPLE_H
#define GPU_BASEROCKSAMPLE_H

#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUmdp.h>
#include <despot/GPUutil/GPUcoord.h>
#include <despot/GPUcore/CudaInclude.h>

namespace despot {

/* =============================================================================
 * RockSampleState class
 * =============================================================================*/
class RockSampleState;

class Dvc_RockSampleState: public Dvc_State {
public:
	DEVICE Dvc_RockSampleState();

	DEVICE void SetAllocated()
	{
		allocated_=true;
	}

	HOST static void CopyToGPU(Dvc_RockSampleState* Dvc, int scenarioID, const RockSampleState*, bool copy_cells=true);

};

/* =============================================================================
 * Dvc_RockSampleApproxParticleUpperBound class
 * =============================================================================*/
class Dvc_RockSampleApproxParticleUpperBound/*: public ParticleUpperBound*/ {
protected:
public:

	DEVICE static float Value( const Dvc_State* state, int scenarioID, Dvc_History& history) ;
};


class Dvc_RockSampleMDPParticleUpperBound/*: public ParticleUpperBound*/ {
public:


	DEVICE static float Value(const Dvc_State* particles, int scenarioID, Dvc_History& history);
};

class Dvc_RockSampleEastScenarioLowerBound/* : public ScenarioLowerBound */{

public:
	DEVICE static Dvc_ValuedAction Value(
			Dvc_State* particles,
			Dvc_RandomStreams& streams,
			Dvc_History& history);

};

extern DEVICE int Dvc_policy_size_;
extern DEVICE Dvc_ValuedAction* Dvc_policy_;
} // namespace despot

#endif
