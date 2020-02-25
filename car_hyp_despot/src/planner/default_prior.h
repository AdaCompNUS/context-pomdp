/*
 * neural_prior.h
 *
 *  Created on: Dec 11, 2018
 *      Author: panpan
 */

#ifndef DEFAULT_PRIOR_H_
#define DEFAULT_PRIOR_H_

#include "despot/interface/pomdp.h"
#include "despot/core/globals.h"
#include "despot/util/coord.h"
#include <despot/core/prior.h>
#include "world_model.h"

class DefaultPrior: public SolverPrior {
public:
	DefaultPrior(const DSPOMDP* model, WorldModel& world);

	std::vector<ACT_TYPE> ComputeLegalActions(const State* state, const DSPOMDP* model);
	void DebugHistory(string msg);
	void RecordCurHistory();
	void CompareHistoryWithRecorded();

public:
	WorldModel& world_model;
	VariableActionStateHistory as_history_in_search_recorded;
};


#endif /* DEFAULT_PRIOR_H_ */
