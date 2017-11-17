#ifndef GPUPedPomdp_H
#define GPUPedPomdp_H

#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUmdp.h>
#include <despot/GPUutil/GPUcoord.h>
#include <despot/GPUcore/CudaInclude.h>
#include "GPU_param.h"
#include "GPU_Path.h"
using namespace despot;

/* =============================================================================
 * Dvc_PomdpState class
 * =============================================================================*/
struct Dvc_PedStruct {
	DEVICE Dvc_PedStruct(){
        vel = Dvc_ModelParams::PED_SPEED;
    }
	DEVICE Dvc_PedStruct(Dvc_COORD a, int b, int c) {
		pos = a;
		goal = b;
		id = c;
        vel = Dvc_ModelParams::PED_SPEED;
	}
	Dvc_COORD pos; //pos
	int goal;  //goal
	int id;   //id
    double vel;
};
struct Dvc_CarStruct {
	int pos;
	float vel;
	float dist_travelled;
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
		if(peds==NULL)
		{
			peds=new Dvc_PedStruct[Dvc_ModelParams::N_PED_IN];
			memset((void*)peds,0, Dvc_ModelParams::N_PED_IN*sizeof(Dvc_PedStruct));
		}
	}

	DEVICE Dvc_PomdpState& operator=(const Dvc_PomdpState& other) // copy assignment
	{
	    if (this != &other) { // self-assignment check expected
                         // storage can be reused
	    	num=other.num;
	    	car.dist_travelled=other.car.dist_travelled;
	    	car.pos=other.car.pos;
	    	car.vel=other.car.vel;

	    	for(int i=0;i</*Dvc_ModelParams::N_PED_IN*/num;i++)
	    	{
				peds[i].goal=other.peds[i].goal;
				peds[i].id=other.peds[i].id;
				peds[i].pos.x=other.peds[i].pos.x;
				peds[i].pos.y=other.peds[i].pos.y;
				peds[i].vel=other.peds[i].vel;
	    	}
	    }
	    return *this;
	}


	DEVICE ~Dvc_PomdpState()
	{
	}

	HOST static void CopyToGPU(Dvc_PomdpState* Dvc, int scenarioID, const PomdpState*, bool copy_cells=true);
	HOST static void CopyToGPU2(Dvc_PomdpState* Dvc, int NumParticles, bool copy_cells=true);
	HOST static void ReadBackToCPU(const Dvc_PomdpState*,PomdpState*, bool copy_cells=true);
	HOST static void ReadBackToCPU2(const Dvc_PomdpState* Dvc,std::vector<State*>, bool copy_cells=true);

};
class Dvc_PedPomdp {

public:
	DEVICE Dvc_PedPomdp(/*int size, int obstacles*/);
	DEVICE ~Dvc_PedPomdp();
	DEVICE static bool Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
		int* obs);
	DEVICE int NumActions() const;
	DEVICE static float ObsProb(OBS_TYPE obs, const Dvc_State& state, int action);
	DEVICE static int Dvc_NumObservations();

	DEVICE Dvc_State* Allocate(int state_id, double weight) const;
	DEVICE static Dvc_State* Dvc_Get(Dvc_State* particles, int pos);
	DEVICE static Dvc_State* Dvc_Alloc( int num);
	DEVICE static Dvc_State* Dvc_Copy(const Dvc_State* particle, int pos);
	DEVICE static void Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des=true);
	DEVICE static void Dvc_Copy_ToShared(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des=true);
	DEVICE static void Dvc_Free(Dvc_State* particle);

	DEVICE static Dvc_ValuedAction Dvc_GetMinRewardAction();

	enum {
		ACT_CUR,
		ACT_ACC,
		ACT_DEC
	};
};


//World model parameters from CPU
DEVICE extern Dvc_Path* path;
DEVICE extern Dvc_COORD* goals;
DEVICE extern double freq;
DEVICE extern double in_front_angle_cos;


#endif
