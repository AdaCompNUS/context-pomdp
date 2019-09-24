#ifndef GPUPedPomdp_H
#define GPUPedPomdp_H

#include <despot/GPUinterface/GPUpomdp.h>
#include <despot/GPUinterface/GPUlower_bound.h>

#include <despot/GPUutil/GPUcoord.h>
#include <despot/GPUcore/CudaInclude.h>
#include "GPU_param.h"
#include "GPU_Path.h"
#include "../WorldModel.h"

using namespace despot;

/* =============================================================================
 * Dvc_PomdpState class
 * =============================================================================*/
struct Dvc_PedStruct {
	DEVICE Dvc_PedStruct() {
		vel = Dvc_ModelParams::PED_SPEED;
		mode = AGENT_DIS;
	}
	DEVICE Dvc_PedStruct(Dvc_COORD a, int b, int c) {
		pos = a;
		goal = b;
		id = c;
		vel = Dvc_ModelParams::PED_SPEED;
		mode = AGENT_DIS;
	}
	Dvc_COORD pos; //pos
	int goal;  //goal
	int id;   //id
	double vel;
	int mode;
};
struct Dvc_CarStruct {
	Dvc_COORD pos;
	float vel;
	float heading_dir;/*[0, 2*PI) heading direction with respect to the world X axis */
};
class PomdpState;
class Dvc_PomdpState: public Dvc_State {
public:
	Dvc_CarStruct car;
	int num;
	Dvc_PedStruct* peds/*[Dvc_ModelParams::N_PED_IN]*/;

	DEVICE Dvc_PomdpState();

	DEVICE Dvc_PomdpState(const Dvc_PomdpState& src);

	DEVICE void Init_peds()
	{
		if (peds == NULL)
		{
			peds = new Dvc_PedStruct[Dvc_ModelParams::N_PED_IN];
			memset((void*)peds, 0, Dvc_ModelParams::N_PED_IN * sizeof(Dvc_PedStruct));
		}
	}

	DEVICE Dvc_PomdpState& operator=(const Dvc_PomdpState& other) // copy assignment
	{
		if (this != &other) { // self-assignment check expected
			// storage can be reused
			num = other.num;
			car.heading_dir = other.car.heading_dir;
			car.pos.x = other.car.pos.x;
			car.pos.y = other.car.pos.y;
			car.vel = other.car.vel;

			for (int i = 0; i < num; i++)
			{
				peds[i].goal = other.peds[i].goal;
				peds[i].id = other.peds[i].id;
				peds[i].pos.x = other.peds[i].pos.x;
				peds[i].pos.y = other.peds[i].pos.y;
				peds[i].vel = other.peds[i].vel;
			}
		}
		return *this;
	}


	DEVICE ~Dvc_PomdpState()
	{
	}

	HOST static void CopyMainStateToGPU(Dvc_PomdpState* Dvc, int scenarioID, const PomdpState*);
	HOST static void CopyPedsToGPU(Dvc_PomdpState* Dvc, int NumParticles, bool deep_copy = true);
	HOST static void ReadMainStateBackToCPU(const Dvc_PomdpState*, PomdpState*);
	HOST static void ReadPedsBackToCPU(const Dvc_PomdpState* Dvc, std::vector<State*>, bool deep_copy = true);

};
class Dvc_PedPomdp: public Dvc_DSPOMDP {

public:
	DEVICE Dvc_PedPomdp(/*int size, int obstacles*/);
	DEVICE ~Dvc_PedPomdp();
	DEVICE static bool Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
	                            int* obs);
	DEVICE static int NumActions();
	DEVICE static float ObsProb(OBS_TYPE obs, const Dvc_State& state, int action);
	DEVICE static int Dvc_NumObservations();

	DEVICE Dvc_State* Allocate(int state_id, double weight) const;
	DEVICE static Dvc_State* Dvc_Get(Dvc_State* particles, int pos);
	DEVICE static Dvc_State* Dvc_Alloc( int num);
	DEVICE static Dvc_State* Dvc_Copy(const Dvc_State* particle, int pos);
	DEVICE static void Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des = true);
	DEVICE static void Dvc_Copy_ToShared(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des = true);
	DEVICE static void Dvc_Free(Dvc_State* particle);

	DEVICE static Dvc_ValuedAction Dvc_GetBestAction();

	DEVICE static float Dvc_GetMaxReward() {return Dvc_ModelParams::GOAL_REWARD;}

	enum {
		ACT_CUR,
		ACT_ACC,
		ACT_DEC
	};
};


//World model parameters from CPU
DEVICE extern Dvc_COORD* car_goal;
DEVICE extern Dvc_COORD* goals;
DEVICE extern double freq;
DEVICE extern double in_front_angle_cos;


#endif
