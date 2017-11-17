#ifndef GPUBELIEF_H
#define GPUBELIEF_H

#include <vector>

#include <despot/util/random.h>
#include <despot/util/logging.h>
#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class Dvc_State;
class Dvc_StateIndexer;
class Dvc_DSPOMDP;

/* =============================================================================
 * Dvc_Belief class
 * =============================================================================*/

class Dvc_Belief {
public:
	const Dvc_DSPOMDP* model_;
	Dvc_History history_;

public:
	/*HOST Dvc_Belief(const Dvc_DSPOMDP* model);
	virtual HOST ~Dvc_Belief();

	virtual HOST std::vector<Dvc_State*> Sample(int num) const = 0;
	virtual HOST void Update(int action, OBS_TYPE obs) = 0;

	virtual HOST std::string text() const;
	friend HOST std::ostream& operator<<(std::ostream& os, const Dvc_Belief& belief);
	virtual HOST Dvc_Belief* MakeCopy() const = 0;

	static HOST std::vector<Dvc_State*> Sample(int num, std::vector<Dvc_State*> belief,
		const Dvc_DSPOMDP* model);
	static HOST std::vector<Dvc_State*> Resample(int num, const std::vector<Dvc_State*>& belief,
		const Dvc_DSPOMDP* model, Dvc_History history, int hstart = 0);
	static HOST std::vector<Dvc_State*> Resample(int num, const Dvc_Belief& belief,
		Dvc_History history, int hstart = 0);
	static HOST std::vector<Dvc_State*> Resample(int num, const Dvc_DSPOMDP* model,
		const Dvc_StateIndexer* indexer, int action, OBS_TYPE obs);*/
};

/* =============================================================================
 * Dvc_ParticleBelief class
 * =============================================================================*/

class Dvc_ParticleBelief: public Dvc_Belief {
protected:
	std::vector<Dvc_State*> particles_;
	int num_particles_;
	Dvc_Belief* prior_;
	bool split_;
	std::vector<Dvc_State*> initial_particles_;
	const Dvc_StateIndexer* state_indexer_;

public:
	/*HOST Dvc_ParticleBelief(std::vector<Dvc_State*> particles, const Dvc_DSPOMDP* model,
		Dvc_Belief* prior = NULL, bool split = true);

	virtual HOST ~Dvc_ParticleBelief();
	HOST void state_indexer(const Dvc_StateIndexer* indexer);

	virtual HOST const std::vector<Dvc_State*>& particles() const;
	virtual HOST std::vector<Dvc_State*> Sample(int num) const;

	virtual HOST void Update(int action, OBS_TYPE obs);

	virtual HOST Dvc_Belief* MakeCopy() const;

	virtual HOST std::string text() const;*/
};

} // namespace despot

#endif
