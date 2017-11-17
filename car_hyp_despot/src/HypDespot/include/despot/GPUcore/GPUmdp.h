#ifndef GPUMDP_H
#define GPUMDP_H

#include <despot/GPUcore/GPUpomdp.h>

#include <despot/GPUcore/CudaInclude.h>

namespace despot {

/**
 * Interface for a discrete Dvc_MDP. This class implements the following functions:
 * <ol>
 * <li> value iteration,
 * <li> computation of alpha vectors and POMDP value for fixed-action policies.
 * </ol>
 */
class Dvc_MDP {
protected:
	std::vector<Dvc_ValuedAction> policy_;

	std::vector<std::vector<double> > blind_alpha_; // For blind policy

public:
	virtual HOST ~Dvc_MDP();

	virtual HOST int NumStates() const = 0;
	virtual HOST int NumActions() const = 0;
	virtual HOST const std::vector<Dvc_State>& TransitionProbability(int s, int a) const = 0;
	virtual HOST double Reward(int s, int a) const = 0;

	virtual HOST void ComputeOptimalPolicyUsingVI();
	HOST const std::vector<Dvc_ValuedAction>& policy() const;

	virtual HOST void ComputeBlindAlpha();
	HOST double ComputeActionValue(const Dvc_ParticleBelief* belief,
		const Dvc_StateIndexer& indexer, int action) const;
	HOST const std::vector<std::vector<double> >& blind_alpha() const;
};

} // namespace despot

#endif
