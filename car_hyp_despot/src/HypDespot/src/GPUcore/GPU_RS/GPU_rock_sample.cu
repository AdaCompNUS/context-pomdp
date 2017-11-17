#include "GPU_rock_sample.h"

using namespace std;

namespace despot {

/* =============================================================================
 * RockSample class
 * =============================================================================*/

DEVICE bool Dvc_RockSample::Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
			OBS_TYPE& obs)
{
	Dvc_RockSampleState& rockstate = static_cast<Dvc_RockSampleState&>(state);

	int obs_i=threadIdx.x;

	if(obs_i==0)
	{
		reward = 0;
		obs = E_NONE;

		if (action < E_SAMPLE) { // Move
			switch (action) {
			case Dvc_Compass::EAST:
				if (GetX(&rockstate) + 1 < map_size_) {
					IncX(&rockstate);
					break;
				} else {
					reward = +10;
					return true;
				}

			case Dvc_Compass::NORTH:
				if (GetY(&rockstate) + 1 < map_size_)
					IncY(&rockstate);
				else
					reward = -100;
				break;

			case Dvc_Compass::SOUTH:
				if (GetY(&rockstate) - 1 >= 0)
					DecY(&rockstate);
				else
					reward = -100;
				break;

			case Dvc_Compass::WEST:
				if (GetX(&rockstate) - 1 >= 0)
					DecX(&rockstate);
				else
					reward = -100;
				break;
			}
		}

		if (action == E_SAMPLE) { // Sample
			int rock = grid_[GetRobPosIndex(&rockstate)];
			if (rock >= 0) {
				if (GetRock(&rockstate, rock))
					reward = +10;
				else
					reward = -10;
				SampleRock(&rockstate, rock);
			} else {
				reward = -100;
			}
		}

		if (action > E_SAMPLE) { // Sense
			int rock = action - E_SAMPLE - 1;
			assert(rock < num_rocks_);
			float distance = DvcCoord::EuclideanDistance(GetRobPos(&rockstate),
				rock_pos_[rock]);
			float efficiency = (1 + powf(2, -distance / half_efficiency_distance_))
				* 0.5;

			if (rand_num < efficiency)
				obs= GetRock(&rockstate, rock) & E_GOOD;
			else
				obs= !(GetRock(&rockstate, rock) & E_GOOD);
		}
	}
	// assert(reward != -100);
	return false;
}

DEVICE int Dvc_RockSample::NumActions()
{
	return num_rocks_ + 5;
}


DEVICE int Dvc_RockSample::Dvc_NumObservations()
{
	return 3;
}
DEVICE Dvc_State* Dvc_RockSample::Dvc_Get(Dvc_State* particles, int pos)
{
	Dvc_RockSampleState* particle_i= static_cast<Dvc_RockSampleState*>(particles)+pos;

	return particle_i;
}
DEVICE void Dvc_RockSample::Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des)
{
	/*Pass member values, assign member pointers to existing state pointer*/
	const Dvc_RockSampleState* src_i= static_cast<const Dvc_RockSampleState*>(src)+pos;
	if(!offset_des) pos=0;
	Dvc_RockSampleState* des_i= static_cast<const Dvc_RockSampleState*>(des)+pos;

	des_i->weight = src_i->weight;
	des_i->scenario_id = src_i->scenario_id;
	des_i->state_id = src_i->state_id;
	des_i->allocated_=true;
}

} // namespace despot
