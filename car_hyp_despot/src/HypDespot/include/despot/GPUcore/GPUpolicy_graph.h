#ifndef GPUPOLICYGRAPH_H
#define GPUPOLICYGRAPH_H

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
#include <despot/GPUcore/GPUpolicy.h>

#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class Dvc_State;
class Dvc_StateIndexer;
class Dvc_StatePolicy;
class Dvc_DSPOMDP;
class Dvc_MMAPInferencer;



/* =============================================================================
 * Dvc_PolicyGraph class
 * =============================================================================*/

class Dvc_PolicyGraph/*: public Dvc_ScenarioLowerBound */{
public:

	DEVICE static Dvc_ValuedAction Value(
		Dvc_State* particles,
		Dvc_RandomStreams& streams,
		Dvc_History& history);


};

/*These values need to be passed from the CPU side*/
DEVICE extern int graph_size_;
DEVICE extern int num_edges_per_node_;
DEVICE extern int* action_nodes_;
DEVICE extern int* obs_edges_;/*A flattened pointer*/
} // namespace despot

#endif//GPUPOLICYGRAPH_H
