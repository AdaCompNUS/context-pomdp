#ifndef GPU_ROCKSAMPLE_H
#define GPU_ROCKSAMPLE_H


#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUmdp.h>
#include "GPU_base_rock_sample.h"
#include <despot/GPUutil/GPUcoord.h>
//#include <despot/util/grid.h>
#include <despot/GPUcore/CudaInclude.h>

namespace despot {

/* =============================================================================
 * RockSample class
 * =============================================================================*/

class Dvc_RockSample {
public:
	enum { // FRAGILE: Don't change!
			E_BAD = 0,
			E_GOOD = 1,
			E_NONE = 2
		};
	enum { //Actions of the robot. FRAGILE: Don't change!
			E_SAMPLE = 4,
			E_SOUTH = 2,
			E_EAST = 1,
			E_WEST = 3,
			E_NORTH = 0,
			//"NE", "SE", "SW", "NW"
			/*E_SOUTH_EAST = 5,
			E_SOUTH_WEST = 6,
			E_NORTH_EAST = 4,
			E_NORTH_WEST = 7,*/
	};

	DEVICE static bool Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
			OBS_TYPE& obs);
	DEVICE static int NumActions();
	DEVICE static int Dvc_NumObservations();
	DEVICE static Dvc_ValuedAction Dvc_GetMinRewardAction() {
		return Dvc_ValuedAction(E_SAMPLE+1, 0);
	}
	DEVICE static Dvc_State* Dvc_Get(Dvc_State* particles, int pos);
	DEVICE static void Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des=true);

	DEVICE static DvcCoord GetCoord(int index);
	DEVICE static DvcCoord GetRobPos(const Dvc_State* state);
	DEVICE static bool GetRock(const Dvc_State* state, int rock);
	DEVICE static int GetX(const Dvc_State* state);
	DEVICE static int GetY(const Dvc_State* state);

	DEVICE static int GetRobPosIndex(const Dvc_State* state);
	DEVICE static void SampleRock(Dvc_State* state, int rock);
	DEVICE static void IncX(Dvc_State* state);
	DEVICE static void DecX(Dvc_State* state);
	DEVICE static void IncY(Dvc_State* state);
	DEVICE static void DecY(Dvc_State* state);
};
/*These values need to be passed from the CPU side*/
DEVICE extern int map_size_;
DEVICE extern int num_rocks_;
DEVICE extern double half_efficiency_distance_;
DEVICE extern int* grid_;/*A flattened pointer of a 2D map*/
DEVICE extern DvcCoord* rock_pos_;
DEVICE extern Dvc_RockSample* rs_model_;
} // namespace despot



#endif
