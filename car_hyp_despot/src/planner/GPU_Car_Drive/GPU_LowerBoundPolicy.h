/*
 * DvcPedPomdpSmartScenarioLowerBoundPolicy.h
 *
 *  Created on: 14 Sep, 2017
 *      Author: panpan
 */

#ifndef DVCPEDPOMDPSMARTSCENARIOLOWERBOUNDPOLICY_H_
#define DVCPEDPOMDPSMARTSCENARIOLOWERBOUNDPOLICY_H_
#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUcore/GPUpolicy.h>
#include <despot/GPUcore/GPUpolicy_graph.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUrandom_streams.h>
#include <despot/GPUutil/GPUlimits.h>

#include "GPU_Car_Drive.h"
using namespace despot;
using archaeopteryx::util::numeric_limits;

class Dvc_PedPomdpSmartPolicy: public Dvc_Policy {
public:
	DEVICE static int Action(int scenarioID, const Dvc_State* particles,
		Dvc_RandomStreams& streams,
		Dvc_History& history);
};

class Dvc_PedPomdpSmartPolicyGraph: public Dvc_PolicyGraph {
public:

	DEVICE static Dvc_ValuedAction Value(
		Dvc_State* particles,
		Dvc_RandomStreams& streams,
		Dvc_History& history);
};

class Dvc_PedPomdpParticleLowerBound: public Dvc_ParticleLowerBound {
private:
	//DEVICE Dvc_TrivialParticleLowerBound(const Dvc_DSPOMDP* model=NULL);
	enum {
			ACT_CUR,
			ACT_ACC,
			ACT_DEC
		};
public:
	DEVICE static Dvc_ValuedAction Value(int scenarioID, Dvc_State * particles)
	{
		Dvc_PomdpState* state = static_cast<Dvc_PomdpState*>(particles);
		int min_step = archaeopteryx::util::numeric_limits<int>::max();
		auto& carpos = path->way_points_[state->car.pos];
		float carvel = state->car.vel;

		// Find mininum num of steps for car-pedestrian collision
		for (int i=0; i<state->num; i++) {
			auto& p = state->peds[i];
			// 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
			int step = (p.vel + carvel<=1e-5)?min_step: int(ceil(Dvc_ModelParams::control_freq
						* max(Dvc_COORD::EuclideanDistance(carpos, p.pos) - 1.0, 0.0)
						/ ((p.vel + carvel))));
			if(step<0) printf("Wrong step, vel_sum=%d %f!!!\n"
								,step, (p.vel + carvel));
			min_step = min(step, min_step);
		}

		//double move_penalty = ped_pomdp_->MovementPenalty(*state);
		float value = Dvc_ModelParams::REWARD_FACTOR_VEL *
							(state->car.vel - Dvc_ModelParams::VEL_MAX) / Dvc_ModelParams::VEL_MAX;

		// Case 1, no pedestrian: Constant car speed
		value = value / (1 - Dvc_Globals::Dvc_Discount(Dvc_config));
		// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
		if (min_step != archaeopteryx::util::numeric_limits<int>::max()) {
			//double crash_penalty = ped_pomdp_->CrashPenalty(*state);
			value = (value) * (1 - Dvc_Globals::Dvc_Discount(Dvc_config,min_step))
					/*/ (1 - Dvc_Globals::Dvc_Discount(Dvc_config))*/;
			value= value + Dvc_ModelParams::CRASH_PENALTY *
					(state->car.vel * state->car.vel + Dvc_ModelParams::REWARD_BASE_CRASH_VEL)
							* Dvc_Globals::Dvc_Discount(Dvc_config,min_step);
		}

		//printf("here\n");
		/*if(state->scenario_id==418 &&blockIdx.x==0)
			printf("min_step,move_penalty, value=%d %f %f\n"
					,min_step,Dvc_ModelParams::REWARD_FACTOR_VEL *
					(state->car.vel - Dvc_ModelParams::VEL_MAX) / Dvc_ModelParams::VEL_MAX,value);*/
		return Dvc_ValuedAction(ACT_CUR,  value);
	}
};

#endif /* DVCPEDPOMDPSMARTSCENARIOLOWERBOUNDPOLICY_H_ */

