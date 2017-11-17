#ifndef GPUDESPOT_H
#define GPUDESPOT_H


#include <despot/solver/despot.h>
#include <despot/GPUcore/GPUpomdp.h>
//#include <despot/GPUcore/GPUsolver.h>
#include <despot/GPUutil/GPUmap.h>
#include <despot/GPUutil/GPUvector.h>

namespace despot {

class GPUDESPOT: public DESPOT {
//friend class VNode;

protected:
	Dvc_DSPOMDP* Dvc_model;
	/*VNode* root_;
	SearchStatistics statistics_;


	Dvc_ScenarioLowerBound* lower_bound_;
	DVc_ScenarioUpperBound* upper_bound_;*/

public:
	void InitGPUModel(const DSPOMDP* model);
	GPUDESPOT(const DSPOMDP* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief = NULL);
	virtual ~GPUDESPOT();

protected:

public:


};



extern DEVICE Dvc_State* (*DvcModelCopy_)(const Dvc_State*, int pos);

extern DEVICE void (*DvcModelFree_)(Dvc_State*);

/*extern DEVICE Dvc_ValuedAction (*DvcLowerBoundValue_)(const Dvc_State*& particles,
		Dvc_RandomStreams& streams, Dvc_History& history);
extern DEVICE Dvc_ValuedAction (*DvcUpperBoundValue_)(const Dvc_State*& particles,
		Dvc_RandomStreams& streams, Dvc_History& history);*/



extern DEVICE Dvc_ValuedAction (*DvcLowerBoundValue_)(Dvc_State *, Dvc_RandomStreams&, Dvc_History&);
extern DEVICE float (*DvcUpperBoundValue_)(const Dvc_State*, int, Dvc_History&);


} // namespace despot

#endif
