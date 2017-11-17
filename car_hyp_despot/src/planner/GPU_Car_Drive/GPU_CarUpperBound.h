/*
 * DvcPedPomdpSmartScenarioLowerBoundPolicy.h
 *
 *  Created on: 14 Sep, 2017
 *      Author: panpan
 */

#ifndef DVCPEDPOMDPUPPERBOUND_H_
#define DVCPEDPOMDPUPPERBOUND_H_
#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUupper_bound.h>

using despot::Dvc_History;
using despot::Dvc_State;


class Dvc_PedPomdpParticleUpperBound1/*: public Dvc_ParticleUpperBound */{
protected:
	//const PedPomdp* rs_model_;
public:
	/*PedPomdpParticleUpperBound1(const PedPomdp* model) :
		rs_model_(model) {
	}*/

	DEVICE static float Value(const Dvc_State* particles, int scenarioID, Dvc_History& history);
};
#endif /* DVCPEDPOMDPSMARTSCENARIOLOWERBOUNDPOLICY_H_ */

