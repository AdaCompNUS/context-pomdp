/*
 * DvcPedPomdpSmartScenarioLowerBoundPolicy.h
 *
 *  Created on: 14 Sep, 2017
 *      Author: panpan
 */

#ifndef DVCPEDPOMDPUPPERBOUND_H_
#define DVCPEDPOMDPUPPERBOUND_H_
#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUinterface/GPUpomdp.h>
#include <despot/GPUinterface/GPUupper_bound.h>

using despot::Dvc_History;
using despot::Dvc_State;
using despot::Dvc_ScenarioUpperBound;
using namespace despot;

class Dvc_PedPomdpParticleUpperBound1: public Dvc_ScenarioUpperBound{
public:

	DEVICE static float Value(const Dvc_State* particles, int scenarioID, Dvc_History& history);

};
#endif /* DVCPEDPOMDPSMARTSCENARIOLOWERBOUNDPOLICY_H_ */

