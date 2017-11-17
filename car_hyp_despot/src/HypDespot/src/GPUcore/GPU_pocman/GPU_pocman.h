#ifndef GPU_POCMAN_H
#define GPU_POCMAN_H


#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUmdp.h>
#include <despot/GPUutil/GPUcoord.h>
//#include <despot/util/grid.h>
#include <despot/GPUcore/CudaInclude.h>

namespace despot {

class PocmanState;
/* =============================================================================
 * poman class
 * =============================================================================*/
//These values need to be passed from the CPU side
DEVICE extern int* maze_;//A flattened pointer of a 2D maz
DEVICE extern int maze_size_x_,maze_size_y_;
DEVICE extern int num_ghosts_;//num_food_;
DEVICE extern int passage_y_;
DEVICE extern int ghost_range_;
DEVICE extern int smell_range_;
DEVICE extern int hear_range_;
DEVICE extern DvcCoord* pocman_home_, * ghost_home_;
DEVICE extern float food_prob_, chase_prob_, defensive_slip_;
DEVICE extern float reward_clear_level_, reward_default_, reward_die_;
DEVICE extern float reward_eat_food_, reward_eat_ghost_, reward_hit_wall_;
DEVICE extern int power_num_steps_;


class Dvc_PocmanState: public Dvc_State {
public:
	DvcCoord pocman_pos;
	DvcCoord* ghost_pos;
	int* ghost_dir;
	int* food; // bit vector
	int num_food;
	int power_steps;
	bool b_Extern_pointers;

	DEVICE Dvc_PocmanState& operator=(const Dvc_PocmanState& src) // copy assignment
	{
	    if (this != &src) { // self-assignment check expected
                         // storage can be reused
	    	pocman_pos.x=src.pocman_pos.x; pocman_pos.y=src.pocman_pos.y;
	    	num_food=src.num_food;
	    	power_steps=src.power_steps;

	    	ghost_pos=src.ghost_pos;
	    	ghost_dir=src.ghost_dir;
	    	food=src.food;

			state_id=src.state_id;
			scenario_id=src.scenario_id;
			weight=src.weight;

			b_Extern_pointers=true;
	    }
	    return *this;
	}

	HOST static void CopyToGPU(Dvc_PocmanState* Dvc, int scenarioID, const PocmanState*, int maze_size_x, int maze_size_y, bool copy_ptr_contents=true);

};

class Dvc_Pocman {
public:

	enum {
		E_PASSABLE, E_SEED, E_POWER
	};
	DEVICE static bool Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
			OBS_TYPE& obs);
	DEVICE static int NumActions();
	DEVICE static int Dvc_NumObservations();
	DEVICE static Dvc_ValuedAction Dvc_GetMinRewardAction() {
		return Dvc_ValuedAction(0, reward_hit_wall_);
	}
	DEVICE static Dvc_State* Dvc_Get(Dvc_State* particles, int pos);
	DEVICE static void Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des=true);
	DEVICE static void Dvc_Copy_ToShared(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des);
};

/* ==============================================================================
 * Dvc_PocmanLegalParticleLowerBound class
 * ==============================================================================*/

class Dvc_PocmanLegalParticleLowerBound/*: public ParticleLowerBound */{

public:
	DEVICE static Dvc_ValuedAction Value(int Sid,
				Dvc_State* particles,
				Dvc_RandomStreams& streams,
				Dvc_History& history);

};
/* ==============================================================================
 * Dvc_PocmanApproxScenarioUpperBound class
 * ==============================================================================*/
class Dvc_PocmanApproxScenarioUpperBound/*: public ScenarioUpperBound*/ {

public:
	DEVICE static float Value( const Dvc_State* state, int scenarioID, Dvc_History& history);

};
DEVICE extern Dvc_Pocman* poc_model_;
} // namespace despot



#endif
