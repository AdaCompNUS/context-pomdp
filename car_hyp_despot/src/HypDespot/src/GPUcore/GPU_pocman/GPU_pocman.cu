#include "GPU_pocman.h"
#include <pocman.h>
#include <despot/GPUutil/Dvc_memorypool.h>
#include <despot/GPUutil/GPUutil.h>
#include <despot/GPUcore/thread_globals.h>
#include <despot/solver/GPUdespot.h>
using namespace std;

namespace despot {
#define THREADDIM 128
/* =============================================================================
 * pocman class
 * =============================================================================*/

static DvcCoord* tmp_ghost_pos=NULL;
static int* tmp_ghost_dir=NULL;
static int* tmp_food=NULL; // bit vector

DEVICE int* maze_=NULL;//A flattened pointer of a 2D maz
DEVICE int maze_size_x_, maze_size_y_;
DEVICE int num_ghosts_;//,num_food_;
DEVICE int passage_y_;
DEVICE int ghost_range_;
DEVICE int smell_range_;
DEVICE int hear_range_;
DEVICE DvcCoord* pocman_home_=NULL, * ghost_home_=NULL;
DEVICE float food_prob_, chase_prob_, defensive_slip_;
DEVICE float reward_clear_level_, reward_default_, reward_die_;
DEVICE float reward_eat_food_, reward_eat_ghost_, reward_hit_wall_;
DEVICE int power_num_steps_;
DEVICE Dvc_Pocman* poc_model_=NULL;

//Dvc_State* Dvc_stepped_particles_all_a=NULL;

static GPU_MemoryPool<Dvc_PocmanState>* gpu_memory_pool_=NULL;
DEVICE static Dvc_MemoryPool<DvcCoord>* dvc_pos_pool_=NULL;
DEVICE static Dvc_MemoryPool<int>* dvc_dir_pool=NULL;
DEVICE static Dvc_MemoryPool<int>* dvc_food_pool=NULL;
static Dvc_PocmanState* Dvc_particles=NULL;
//static int* Dvc_particleIDs=NULL;
static Dvc_PocmanState* tmp=NULL;
static float* tmp_result=NULL;
static DvcCoord* tmp_pos=NULL;
static int* tmp_dir=NULL;

__global__ void CopyArrays(Dvc_PocmanState* Dvc, DvcCoord* pos,int* dir,  int* food)
{
	int x=threadIdx.x;
	/*if(x==0 && y==0)
	{
		Dvc->cells=(bool*)malloc(Dvc->sizeX_*Dvc->sizeY_*sizeof(bool));
	}
	__syncthreads();*/

	if(x<maze_size_y_)
		Dvc->food[x]=food[x];
	bool error=false;
	if(x<num_ghosts_)
	{
		Dvc->ghost_pos[x]=pos[x];
		Dvc->ghost_dir[x]=dir[x];

		if(Dvc->ghost_pos[x].x>=maze_size_x_)
			error=true;
		if(Dvc->ghost_dir[x]>=4)
			error=true;
		//dir[x]=error;
	}
}
HOST void Dvc_PocmanState::CopyToGPU(Dvc_PocmanState* Dvc, int scenarioID, const PocmanState* Hst, int maze_size_x, int maze_size_y, bool copy_ptr_contents)
{
	//HANDLE_ERROR(cudaMemcpy((void*)Dvc, (const void*)&(Hst->allocated_), sizeof(Dvc_UncNavigationState), cudaMemcpyHostToDevice));
	Dvc[scenarioID].pocman_pos.x=Hst->pocman_pos.x;
	Dvc[scenarioID].pocman_pos.y=Hst->pocman_pos.y;

	Dvc[scenarioID].weight=Hst->weight;

	if(copy_ptr_contents)
	{
		Dvc[scenarioID].state_id=Hst->state_id;
		Dvc[scenarioID].scenario_id=Hst->scenario_id;
		Dvc[scenarioID].num_food=Hst->num_food;
		Dvc[scenarioID].power_steps=Hst->power_steps;
	}

	Dvc[scenarioID].b_Extern_pointers=false;

	if(Dvc!=NULL)
	{

		if(copy_ptr_contents)
		{
			int num_ghost=Hst->ghost_pos.size();
			HANDLE_ERROR(cudaMemcpy((void*)tmp_ghost_pos, (const void*)Hst->ghost_pos.data(),
					num_ghost*sizeof(DvcCoord), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)tmp_ghost_dir, (const void*)Hst->ghost_dir.data(),
					num_ghost*sizeof(int), cudaMemcpyHostToDevice));
			//cout<<"("<<tmp_ghost_pos[0].x<<","<< tmp_ghost_dir[0]<<") ";

			//HANDLE_ERROR(cudaMemcpy((void*)tmp_food, (const void*)Hst->food.data(),
			//		maze_size_x*maze_size_y*sizeof(int), cudaMemcpyHostToDevice));//CUDA bool cannot be copied like this

			for(int i=0;i<maze_size_x;i++)
				for(int j=0;j<maze_size_y;j++)
				{
					if(Hst->food[j*maze_size_x+i])
						SetFlag(tmp_food[j],i);
					else
						UnsetFlag(tmp_food[j],i);
				}
			//HANDLE_ERROR(cudaMemcpy((void*)tmp_food, (const void*)Hst->food.data(),
			//		Hst->num_food*sizeof(bool), cudaMemcpyHostToDevice));
			dim3 grid1(1,1);dim3 threads1(max(4, maze_size_y));
			CopyArrays<<<grid1, threads1>>>(Dvc+scenarioID,tmp_ghost_pos,tmp_ghost_dir,tmp_food);
			cudaDeviceSynchronize();
		}
	}

}


DEVICE bool Dvc_Pocman::Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
			OBS_TYPE& obs)
{
	Dvc_PocmanState& pocstate = static_cast<Dvc_PocmanState&>(state);

	int obs_i=threadIdx.y;
	Dvc_Random random;
	__shared__ int hitGhost[MC_DIM];
	__shared__ bool ghost_heard[MC_DIM];
	__shared__ unsigned int particle_obs[MC_DIM];
	bool error=false;
	DvcCoord newpos;
	if(obs_i==0)
	{
		//random[threadIdx.x].Seed(rand_num);//An equivalent line was in CPUcode.
		hitGhost[threadIdx.x]= -1;
		ghost_heard[threadIdx.x]=false;
		particle_obs[threadIdx.x]=0;
	}
	__syncthreads();
	reward = reward_default_;
	int observation = 0;

	//Move Pocman
	if(obs_i==0)
	{
		if (pocstate.pocman_pos.x == 0 && pocstate.pocman_pos.y == passage_y_ && action == Dvc_Compass::WEST)
			newpos = DvcCoord(maze_size_x_ - 1, pocstate.pocman_pos.y);
		else if (pocstate.pocman_pos.x == maze_size_x_ - 1 && pocstate.pocman_pos.y == passage_y_
			&& action == Dvc_Compass::EAST)
			newpos = DvcCoord(0, pocstate.pocman_pos.y);
		else
			newpos = pocstate.pocman_pos + Dvc_Compass::GetDirections(action);

		if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
				&& newpos.y < maze_size_y_
				&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
			;
		else
			{newpos.x=-1;newpos.y=-1;}

		if (newpos.x >= 0 && newpos.y >= 0)
			pocstate.pocman_pos = newpos;
		else
			reward += reward_hit_wall_;

		if (pocstate.power_steps > 0)
			pocstate.power_steps--;
	}

	__syncthreads();
	//Move ghost

	//for (int g = 0; g < num_ghosts_; g++) {
	if(obs_i<num_ghosts_){

		if(pocstate.ghost_pos[obs_i].x>=maze_size_x_)
			error=true;
		if(pocstate.ghost_dir[obs_i]>=4)
		    error=true;

		if (pocstate.ghost_pos[obs_i] == pocstate.pocman_pos)
			hitGhost[threadIdx.x] = obs_i;
		//MoveGhost(pocstate, g, random);
		if (DvcCoord::ManhattanDistance(pocstate.pocman_pos, pocstate.ghost_pos[obs_i])
				< ghost_range_) {
			if (pocstate.power_steps > 0)
			{
				//MoveGhostDefensive(pocstate, g, random);
				if (random.NextDouble(state.scenario_id) < defensive_slip_ && pocstate.ghost_dir[obs_i] >= 0) {
					pocstate.ghost_dir[obs_i] = -1;
					//return;
				}
				else
				{
					int bestDist = 0;
					DvcCoord bestPos = pocstate.ghost_pos[obs_i];
					int bestDir = -1;
					for (int dir = 0; dir < 4; dir++) {
						int dist = DvcCoord::DirectionalDistance(pocstate.pocman_pos,
							pocstate.ghost_pos[obs_i], dir);
						//DvcCoord newpos = NextPos(pocstate.ghost_pos[obs_i], dir);
						if (pocstate.ghost_pos[obs_i].x == 0 && pocstate.ghost_pos[obs_i].y == passage_y_ && dir == Dvc_Compass::WEST)
							newpos = DvcCoord(maze_size_x_ - 1, pocstate.ghost_pos[obs_i].y);
						else if (pocstate.ghost_pos[obs_i].x == maze_size_x_ - 1 && pocstate.ghost_pos[obs_i].y == passage_y_
							&& dir == Dvc_Compass::EAST)
							newpos = DvcCoord(0, pocstate.ghost_pos[obs_i].y);
						else
							newpos = pocstate.ghost_pos[obs_i] + Dvc_Compass::GetDirections(dir);

						if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
								&& newpos.y < maze_size_y_
								&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
							;
						else
							{newpos.x=-1;newpos.y=-1;}

						if (dist >= bestDist && newpos.x >= 0 && newpos.y >= 0
							&& Dvc_Compass::Opposite(dir) != pocstate.ghost_dir[obs_i]) {
							bestDist = dist;
							bestPos = newpos;
						}
					}

					pocstate.ghost_pos[obs_i] = bestPos;
					pocstate.ghost_dir[obs_i] = bestDir;
				}

			}
			else
			{//MoveGhostAggressive(pocstate, g, random);
				if (random.NextDouble(state.scenario_id) > chase_prob_) {
					//MoveGhostRandom(pocstate, g, random);
					//DvcCoord newpos;
					int dir;
					do {
						dir = random.NextInt(4,state.scenario_id);
						//newpos = NextPos(pocstate.ghost_pos[obs_i], dir);
						if (pocstate.ghost_pos[obs_i].x == 0 && pocstate.ghost_pos[obs_i].y == passage_y_ && dir == Dvc_Compass::WEST)
							newpos = DvcCoord(maze_size_x_ - 1, pocstate.ghost_pos[obs_i].y);
						else if (pocstate.ghost_pos[obs_i].x == maze_size_x_ - 1 && pocstate.ghost_pos[obs_i].y == passage_y_
							&& dir == Dvc_Compass::EAST)
							newpos = DvcCoord(0, pocstate.ghost_pos[obs_i].y);
						else
							newpos = pocstate.ghost_pos[obs_i] + Dvc_Compass::GetDirections(dir);

						if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
								&& newpos.y < maze_size_y_
								&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
							;
						else
							{newpos.x=-1;newpos.y=-1;}
					} while (Dvc_Compass::Opposite(dir) == pocstate.ghost_dir[obs_i]
						|| !(newpos.x >= 0 && newpos.y >= 0));
					pocstate.ghost_pos[obs_i] = newpos;
					pocstate.ghost_dir[obs_i] = dir;
					//return;
				}
				else
				{
					int bestDist = maze_size_x_+maze_size_y_;
					DvcCoord bestPos = pocstate.ghost_pos[obs_i];
					int bestDir = -1;
					for (int dir = 0; dir < 4; dir++) {
						int dist = DvcCoord::DirectionalDistance(pocstate.pocman_pos,
							pocstate.ghost_pos[obs_i], dir);
						//DvcCoord newpos = NextPos(pocstate.ghost_pos[obs_i], dir);
						if (pocstate.ghost_pos[obs_i].x == 0 && pocstate.ghost_pos[obs_i].y == passage_y_ && dir == Dvc_Compass::WEST)
							newpos = DvcCoord(maze_size_x_ - 1, pocstate.ghost_pos[obs_i].y);
						else if (pocstate.ghost_pos[obs_i].x == maze_size_x_ - 1 && pocstate.ghost_pos[obs_i].y == passage_y_
							&& dir == Dvc_Compass::EAST)
							newpos = DvcCoord(0, pocstate.ghost_pos[obs_i].y);
						else
							newpos = pocstate.ghost_pos[obs_i] + Dvc_Compass::GetDirections(dir);

						if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
								&& newpos.y < maze_size_y_
								&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
							;
						else
							{newpos.x=-1;newpos.y=-1;}
						if (dist <= bestDist && newpos.x >= 0 && newpos.y >= 0
							&& Dvc_Compass::Opposite(dir) != pocstate.ghost_dir[obs_i]) {
							bestDist = dist;
							bestPos = newpos;
						}
					}

					pocstate.ghost_pos[obs_i] = bestPos;
					pocstate.ghost_dir[obs_i] = bestDir;
				}
			}
		} else {
			//MoveGhostRandom(pocstate, g, random);
			//DvcCoord newpos;
			int dir;
			do {
				dir = random.NextInt(4,state.scenario_id);
				//newpos = NextPos(pocstate.ghost_pos[obs_i], dir);
				if (pocstate.ghost_pos[obs_i].x == 0 && pocstate.ghost_pos[obs_i].y == passage_y_ && dir == Dvc_Compass::WEST)
					newpos = DvcCoord(maze_size_x_ - 1, pocstate.ghost_pos[obs_i].y);
				else if (pocstate.ghost_pos[obs_i].x == maze_size_x_ - 1 && pocstate.ghost_pos[obs_i].y == passage_y_
					&& dir == Dvc_Compass::EAST)
					newpos = DvcCoord(0, pocstate.ghost_pos[obs_i].y);
				else
					newpos = pocstate.ghost_pos[obs_i] + Dvc_Compass::GetDirections(dir);

				if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
						&& newpos.y < maze_size_y_
						&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
					;
				else
					{newpos.x=-1;newpos.y=-1;}
			} while (Dvc_Compass::Opposite(dir) == pocstate.ghost_dir[obs_i]
				|| !(newpos.x >= 0 && newpos.y >= 0));
			pocstate.ghost_pos[obs_i] = newpos;
			pocstate.ghost_dir[obs_i] = dir;
		}
		if (pocstate.ghost_pos[obs_i] == pocstate.pocman_pos)
			hitGhost[threadIdx.x] = obs_i;
	}
	__syncthreads();
	if(obs_i==0)
	{
		if (hitGhost[threadIdx.x] >= 0) {
			if (pocstate.power_steps > 0) {
				reward += reward_eat_ghost_;
				pocstate.ghost_pos[hitGhost[threadIdx.x]] = *ghost_home_;
				pocstate.ghost_dir[hitGhost[threadIdx.x]] = -1;
			} else {
				reward += reward_die_;
				return true;
			}
		}
	}
	__syncthreads();
    //observation = MakeObservations(pocstate);
	//int observation = 0;
	//See ghosts
	if(obs_i < 4) {
		bool ghost_seen=false;
		newpos = pocstate.pocman_pos + Dvc_Compass::GetDirections(obs_i);
		//while (maze_.Inside(newpos) && Passable(newpos))
		while (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
				&& newpos.y < maze_size_y_
				&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
		{
			for (int g = 0; g < num_ghosts_; g++)
				if (pocstate.ghost_pos[g] == newpos)
					ghost_seen=true;
			if(ghost_seen) break;
			newpos += Dvc_Compass::GetDirections(obs_i);
		}

		if (ghost_seen)
			Dvc_SetFlag(observation, obs_i);
		//Observe walls
		//newpos = NextPos(pocstate.pocman_pos, obs_i);
		if (pocstate.pocman_pos.x == 0 && pocstate.pocman_pos.y == passage_y_ && obs_i == Dvc_Compass::WEST)
			newpos = DvcCoord(maze_size_x_ - 1, pocstate.pocman_pos.y);
		else if (pocstate.pocman_pos.x == maze_size_x_ - 1 && pocstate.pocman_pos.y == passage_y_
			&& obs_i == Dvc_Compass::EAST)
			newpos = DvcCoord(0, pocstate.pocman_pos.y);
		else
			newpos = pocstate.pocman_pos + Dvc_Compass::GetDirections(obs_i);

		if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
				&& newpos.y < maze_size_y_
				&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
			;
		else
			{newpos.x=-1;newpos.y=-1;}
		if (newpos.x >= 0 && newpos.y >= 0 && Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x], E_PASSABLE))
			Dvc_SetFlag(observation, obs_i + 4);
	}
	//Smell food
	if(obs_i<(smell_range_*2+1)*(smell_range_*2+1))
	{
		bool food_smelled=false;

		DvcCoord dir(obs_i/(smell_range_*2+1)-smell_range_,
				obs_i%(smell_range_*2+1)-smell_range_);

		newpos=pocstate.pocman_pos + dir;
		if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_ && newpos.y < maze_size_y_
			&& Dvc_CheckFlag(pocstate.food[newpos.y],newpos.x))
			{food_smelled= true;}
		if (food_smelled)
			Dvc_SetFlag(observation, 8);
	}
	//Hear ghosts
	if (obs_i< num_ghosts_)
	{
		if (DvcCoord::ManhattanDistance(pocstate.ghost_pos[obs_i], pocstate.pocman_pos)
			<= hear_range_)
			ghost_heard[threadIdx.x]=true;
		if (ghost_heard[threadIdx.x])
			Dvc_SetFlag(observation, 9);
	}
	//Eat food
	if(obs_i==0)
	{
		//int pocIndex = pocstate.pocman_pos.y * maze_size_x_ + pocstate.pocman_pos.x;
		if (Dvc_CheckFlag(pocstate.food[pocstate.pocman_pos.y],pocstate.pocman_pos.x)) {
			Dvc_UnsetFlag(pocstate.food[pocstate.pocman_pos.y],pocstate.pocman_pos.x);
			//pocstate.food[pocIndex] = false;
			pocstate.num_food--;
			if (pocstate.num_food == 0) {
				reward += reward_clear_level_;
				return true;
			}
			//Eat power pills
			if (Dvc_CheckFlag(maze_[maze_size_x_ * pocstate.pocman_pos.y + pocstate.pocman_pos.x],E_POWER))
				pocstate.power_steps = power_num_steps_;
			reward += reward_eat_food_;
		}
	}/**/

	if(obs_i<max(max(num_ghosts_,(smell_range_*2+1)*(smell_range_*2+1)),4))
		atomicOr(particle_obs+threadIdx.x, observation);
	syncthreads();

	obs=particle_obs[threadIdx.x];

	error=false;
	// assert(reward != -100);
	return error;
}

DEVICE int Dvc_Pocman::NumActions()
{
	return 4;
}


DEVICE int Dvc_Pocman::Dvc_NumObservations()
{
	return 1024;
}
DEVICE Dvc_State* Dvc_Pocman::Dvc_Get(Dvc_State* particles, int pos)
{
	Dvc_PocmanState* particle_i= static_cast<Dvc_PocmanState*>(particles)+pos;

	return particle_i;
}
DEVICE void Dvc_Pocman::Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des)
{
	//Pass member values, assign member pointers to existing state pointer
	const Dvc_PocmanState* src_i= static_cast<const Dvc_PocmanState*>(src)+pos;
	if(!offset_des) pos=0;
	Dvc_PocmanState* des_i= static_cast<const Dvc_PocmanState*>(des)+pos;

	des_i->pocman_pos.x=src_i->pocman_pos.x;des_i->pocman_pos.y=src_i->pocman_pos.y;
    des_i->num_food = src_i->num_food;
    des_i->power_steps = src_i->power_steps;

   /* if(des_i->ghost_dir==NULL)
	{
		des_i->ghost_dir=(int*)(des_i+1);//just after the Dvc_PocmanState
		des_i->ghost_pos=(DvcCoord*)(des_i->ghost_dir+num_ghosts_);
		des_i->food=(int*)(des_i->ghost_pos+num_ghosts_);
		des_i->b_Extern_pointers=false;
	}*/

	if(des_i->b_Extern_pointers==false)
	{

		bool error=false;
		memcpy(des_i->ghost_pos, src_i->ghost_pos, num_ghosts_*sizeof(DvcCoord));
		memcpy(des_i->ghost_dir, src_i->ghost_dir, num_ghosts_*sizeof(int));
		memcpy(des_i->food, src_i->food, maze_size_y_*sizeof(int));

		if(des_i->ghost_pos[0].x>=maze_size_x_)
			error=true;
		if(des_i->ghost_dir[0]>=4)
			error=true;
		pos=error;
	}

	des_i->weight = src_i->weight;
	des_i->scenario_id = src_i->scenario_id;
	des_i->state_id = src_i->state_id;
	des_i->allocated_=true;
}
DEVICE void Dvc_Pocman::Dvc_Copy_ToShared(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des)
{
	//Pass member values, assign member pointers to existing state pointer
	const Dvc_PocmanState* src_i= static_cast<const Dvc_PocmanState*>(src)+pos;
	if(!offset_des) pos=0;
	Dvc_PocmanState* des_i= static_cast<const Dvc_PocmanState*>(des)+pos;

	des_i->pocman_pos.x=src_i->pocman_pos.x;des_i->pocman_pos.y=src_i->pocman_pos.y;
    des_i->num_food = src_i->num_food;
    des_i->power_steps = src_i->power_steps;

	des_i->ghost_dir=(int*)(des_i+1);//just after the Dvc_PocmanState
	des_i->ghost_pos=(DvcCoord*)(des_i->ghost_dir+num_ghosts_);
	des_i->food=(int*)(des_i->ghost_pos+num_ghosts_);
	des_i->b_Extern_pointers=false;

	if(des_i->b_Extern_pointers==false)
	{

		bool error=false;
		memcpy(des_i->ghost_pos, src_i->ghost_pos, num_ghosts_*sizeof(DvcCoord));
		memcpy(des_i->ghost_dir, src_i->ghost_dir, num_ghosts_*sizeof(int));
		memcpy(des_i->food, src_i->food, maze_size_y_*sizeof(int));

		if(des_i->ghost_pos[0].x>=maze_size_x_)
			error=true;
		if(des_i->ghost_dir[0]>=4)
			error=true;
		pos=error;
	}

	des_i->weight = src_i->weight;
	des_i->scenario_id = src_i->scenario_id;
	des_i->state_id = src_i->state_id;
	des_i->allocated_=true;
}

__global__ void AllocContents(Dvc_PocmanState* state, int numParticles)
{
	//int i=threadIdx.x+blockIdx.x*blockDim.x;
	for(int i=0;i<numParticles;i++)
	{
		state[i].ghost_dir=dvc_dir_pool->Allocate(num_ghosts_);
		state[i].ghost_pos=dvc_pos_pool_->Allocate(num_ghosts_);
		state[i].food=dvc_food_pool->Allocate(maze_size_y_);
		state[i].b_Extern_pointers=false;
		//state->num_food=maze_size_x_*maze_size_y_;
		//state->
	}
}
__global__ void InitContents(Dvc_PocmanState* state, int numParticles)
{
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;
	int i=y*gridDim.x*blockDim.x+x;
	if(i<numParticles)
	{
		for(int j=0;j<num_ghosts_;j++)
		{
			state[i].ghost_dir[j]=100;
			state[i].ghost_pos[j].x=100;state[i].ghost_pos[j].y=100;
		}
	}
}

Dvc_State* Pocman::AllocGPUParticles(int numParticles, int alloc_mode) const
{//numParticles==num_Scenarios
	clock_t start=clock();
	//dim3 grid((numParticles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);
	dim3 grid((numParticles+THREADDIM-1)/THREADDIM,NumActions()); dim3 threads(THREADDIM,1);
	int num_threads=1;

	if(use_multi_thread_)
	{
		num_threads=NUM_THREADS;
	}

	switch(alloc_mode)//At launch of the program
	{
	case 0:
		CreateMemoryPool(0);

		HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles, numParticles*sizeof(Dvc_PocmanState)));
		AllocContents<<<1, 1>>>(Dvc_particles, numParticles);
		HANDLE_ERROR(cudaDeviceSynchronize());

		//HANDLE_ERROR(cudaMallocManaged((void**)&tmp,numParticles*sizeof(Dvc_PocmanState)));
		HANDLE_ERROR(cudaMallocManaged((void**)&tmp_pos,numParticles*sizeof(DvcCoord)));
		HANDLE_ERROR(cudaMallocManaged((void**)&tmp_dir,numParticles*sizeof(int)));

		//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particleIDs,numParticles*sizeof(int) ));
		HANDLE_ERROR(cudaMalloc(&tmp_result, sizeof(float)));

		Dvc_stepped_particles_all_a=new Dvc_State*[num_threads];
		for(int i=0;i<num_threads;i++)
		{
			HANDLE_ERROR(cudaMalloc((void**)&Dvc_stepped_particles_all_a[i], NumActions()*numParticles*sizeof(Dvc_PocmanState)));
			AllocContents<<<1, 1>>>((Dvc_PocmanState*)Dvc_stepped_particles_all_a[i], NumActions()*numParticles);
			HANDLE_ERROR(cudaDeviceSynchronize());
			InitContents<<<grid,threads>>>((Dvc_PocmanState*)Dvc_stepped_particles_all_a[i], NumActions()*numParticles);
			HANDLE_ERROR(cudaDeviceSynchronize());
		}

		HANDLE_ERROR(cudaMallocManaged((void**)&tmp_ghost_pos, num_ghosts_*sizeof(DvcCoord)));
		HANDLE_ERROR(cudaMallocManaged((void**)&tmp_ghost_dir, num_ghosts_*sizeof(int)));
		HANDLE_ERROR(cudaMallocManaged((void**)&tmp_food, maze_.ysize()*sizeof(int)));

		return Dvc_particles;
	case 1://At beginning of each search
		CreateMemoryPool(1);
		AllocContents<<<1, 1>>>(Dvc_particles, numParticles);
		HANDLE_ERROR(cudaDeviceSynchronize());

		for(int i=0;i<num_threads;i++)
		{
			AllocContents<<<1, 1>>>((Dvc_PocmanState*)Dvc_stepped_particles_all_a[i], NumActions()*numParticles);
			HANDLE_ERROR(cudaDeviceSynchronize());
			InitContents<<<grid,threads>>>((Dvc_PocmanState*)Dvc_stepped_particles_all_a[i], NumActions()*numParticles);
			HANDLE_ERROR(cudaDeviceSynchronize());
		}

		return Dvc_particles;
	case 2://during the despot search
		//cout<<__FUNCTION__ <<endl;
		//cout<<"numParticles"<<numParticles<<endl;
		//cout<<"gpu_memory_pool_"<<gpu_memory_pool_<<endl;

		Dvc_PocmanState* tmp=gpu_memory_pool_->Allocate(numParticles);

		AllocContents<<<1, 1>>>(tmp, numParticles);
		HANDLE_ERROR(cudaDeviceSynchronize());
		//Dvc_PocmanState* tmp;
		//HANDLE_ERROR(cudaMalloc((void**)&tmp, numParticles*sizeof(Dvc_PocmanState)));

		return tmp;
	}
	/*Dvc_particles_copy is an extern variable declared in GPUpolicy.h*/
	//HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles_copy, numParticles*sizeof(Dvc_PocmanState)));
	/*AllocParticleCopy<<<grid, threads>>>(size_, size_,numParticles);
	HANDLE_ERROR(cudaDeviceSynchronize());*/

	cout<<"GPU particles alloc time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl;
}
__global__ void CopyParticles(Dvc_PocmanState* des,Dvc_PocmanState* src,float* weight,
		int* IDs,int num_particles,
		Dvc_RandomStreams* streams, int stream_pos
		)
{
	int pos=blockIdx.x*blockDim.x+threadIdx.x;
	//int y=threadIdx.y;

	if(pos==0)
	{
		weight[0]=0;
		if(streams) streams->position_=stream_pos;
	}
	__syncthreads();
	if(pos < num_particles)
	{
		bool error=false;
		int scenarioID=IDs[pos];
		Dvc_PocmanState* src_i=src+scenarioID;//src is a full length array for all particles
		Dvc_PocmanState* des_i=des+pos;//des is short, only for the new partition

		des_i->pocman_pos.x=src_i->pocman_pos.x;des_i->pocman_pos.y=src_i->pocman_pos.y;
		des_i->num_food = src_i->num_food;
		des_i->power_steps = src_i->power_steps;

		if(des_i->b_Extern_pointers==false)
		{
			memcpy(des_i->ghost_pos, src_i->ghost_pos, num_ghosts_*sizeof(DvcCoord));
			memcpy(des_i->ghost_dir, src_i->ghost_dir, num_ghosts_*sizeof(int));
			memcpy(des_i->food, src_i->food, maze_size_y_*sizeof(int));

			if(des_i->ghost_pos[0].x>=maze_size_x_)
				error=true;
			if(des_i->ghost_dir[0]>=4)
				error=true;
		}

		des_i->weight = src_i->weight;
		des_i->scenario_id = src_i->scenario_id;
		des_i->state_id = src_i->state_id;
		des_i->allocated_=true;

		num_particles=false;
		atomicAdd(weight, des_i->weight);
	}
}

__global__ void PassDeviceValue(Dvc_PocmanState* dvc, int* managed_dir, DvcCoord* managed_pos,int* IDs,int num_particles)
{
	int pos=blockIdx.x*blockDim.x+threadIdx.x;
	//int y=threadIdx.y;
	if(pos < num_particles)
	{
		int scenarioID=IDs[pos];
		Dvc_PocmanState* src_i=dvc+scenarioID;//src is a full length array for all particles
		managed_dir[pos]=src_i->ghost_dir[0];
		managed_pos[pos]=src_i->ghost_pos[0];
	}
	__syncthreads();
}
void Pocman::CopyGPUParticles(Dvc_State* des,Dvc_State* src,int src_offset,int* IDs,
		int num_particles,bool interleave,
		Dvc_RandomStreams* streams, int stream_pos,
		void* CUDAstream, int shift) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);

	/*cout<<"src_offset="<<src_offset<<": ";
	PassDeviceValue<<<grid, threads>>>(static_cast<Dvc_PocmanState*>(src)+src_offset, tmp_dir, tmp_pos, IDs,num_particles);
	HANDLE_ERROR(cudaStreamSynchronize(0));
	for(int i=0;i<num_particles;i++)
	{
		cout<<"("<<tmp_pos[i].x<<","<<tmp_dir[i]<<") ";
	}
	cout<<endl;*/

	if(CUDAstream)
	{
		CopyParticles<<<grid, threads,0, *(cudaStream_t*)CUDAstream>>>(static_cast<Dvc_PocmanState*>(des),
				static_cast<Dvc_PocmanState*>(src)+src_offset,tmp_result,
				IDs,num_particles, streams, stream_pos);
		if(!interleave)
			HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)CUDAstream));
	}
	else
	{
		CopyParticles<<<grid, threads,0, 0>>>(static_cast<Dvc_PocmanState*>(des),
				static_cast<Dvc_PocmanState*>(src)+src_offset,tmp_result,
				IDs,num_particles, streams, stream_pos);
		if(!interleave)
			HANDLE_ERROR(cudaStreamSynchronize(0));
	}

		//HANDLE_ERROR(cudaDeviceSynchronize());
}
void Pocman::CopyGPUWeight1(void* cudaStream, int shift) const
{
	;
}
float Pocman::CopyGPUWeight2(void* cudaStream, int shift) const
{
	float particle_weight=0;
	HANDLE_ERROR(cudaMemcpy(&particle_weight, tmp_result, sizeof(float),cudaMemcpyDeviceToHost));
	return particle_weight;
}
void Pocman::DeleteGPUParticles( int num_particles) const
{
	HANDLE_ERROR(cudaFree(static_cast<Dvc_PocmanState*>(Dvc_particles)));

	//HANDLE_ERROR(cudaFree(Dvc_particleIDs));

	HANDLE_ERROR(cudaFree(tmp_result));
	HANDLE_ERROR(cudaFree(tmp_ghost_dir));
	HANDLE_ERROR(cudaFree(tmp_ghost_pos));
	HANDLE_ERROR(cudaFree(tmp_food));

	int num_threads=1;

	if(use_multi_thread_)
	{
		num_threads=NUM_THREADS;
	}
	for(int i=0;i<num_threads;i++)
	{
		if(Dvc_stepped_particles_all_a[i]!=NULL)
			{HANDLE_ERROR(cudaFree(Dvc_stepped_particles_all_a[i]));Dvc_stepped_particles_all_a[i]=NULL;}
	}
	if(Dvc_stepped_particles_all_a)delete [] Dvc_stepped_particles_all_a;Dvc_stepped_particles_all_a=NULL;

	cudaFree(tmp);cudaFree(tmp_dir);cudaFree(tmp_pos);
}

void Pocman::DeleteGPUParticles(Dvc_State* particles, int num_particles) const
{
	HANDLE_ERROR(cudaFree(static_cast<Dvc_PocmanState*>(particles)));
}

Dvc_State* Pocman::GetGPUParticles() const
{
	return Dvc_particles;
}
Dvc_State* Pocman::CopyToGPU(const std::vector<State*>& particles, bool copy_cells) const
{
	//Dvc_PocmanState* Dvc_particles;
	//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particles, particles.size()*sizeof(Dvc_PocmanState)));
	//dvc_particles should be managed device memory
	clock_t start=clock();

	for (int i=0;i<particles.size();i++)
	{
		const PocmanState* src=static_cast<const PocmanState*>(particles[i]);
		//Dvc_particles[i].Assign(src);
		Dvc_PocmanState::CopyToGPU(Dvc_particles,src->scenario_id,src, maze_.xsize(), maze_.ysize());
		//Dvc_PocmanState::CopyToGPU(NULL,src->scenario_id,src, false);//copy to Dvc_particles_copy, do not copy cells, leave it a NULL pointer
	}
	//cout<<"GPU particles copy time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl;

	return Dvc_particles;
}
void Pocman::CopyToGPU(const std::vector<int>& particleIDs,int* Dvc_ptr, void* CUDAstream) const
{
	if(CUDAstream)
	{
		//HANDLE_ERROR(cudaHostRegister((void*)particleIDs.data(),particleIDs.size()*sizeof(int),cudaHostRegisterPortable));
		HANDLE_ERROR(cudaMemcpyAsync(Dvc_ptr,particleIDs.data(),particleIDs.size()*sizeof(int), cudaMemcpyHostToDevice,*(cudaStream_t*)CUDAstream));
		HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)CUDAstream));
		//HANDLE_ERROR(cudaHostUnregister((void*)particleIDs.data()));
	}
	else
	{
		HANDLE_ERROR(cudaMemcpy(Dvc_ptr,particleIDs.data(),particleIDs.size()*sizeof(int), cudaMemcpyHostToDevice));
	}
	//return Dvc_particleIDs;
}

__global__ void CreateDvcMemoryPools()
{
	if(dvc_pos_pool_==NULL)
		dvc_pos_pool_=new Dvc_MemoryPool<DvcCoord>;
	if(dvc_dir_pool==NULL)
		dvc_dir_pool=new Dvc_MemoryPool<int>;
	if(dvc_food_pool==NULL)
		dvc_food_pool=new Dvc_MemoryPool<int>;
}

void Pocman::CreateMemoryPool(int mode) const
{
	if(mode==0)
	{
		CreateDvcMemoryPools<<<1,1,1>>>();
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	//cout<<__FUNCTION__<<endl;
	if(gpu_memory_pool_==NULL)
	{
		gpu_memory_pool_=new GPU_MemoryPool<Dvc_PocmanState>;
	}

	//gpu_memory_pool_->SetChunkSize(chunk_size);

}

__global__ void DestoryDvcMemoryPools(int mode)
{
	switch (mode)
	{
	case 0:
		if(dvc_pos_pool_){delete dvc_pos_pool_;dvc_pos_pool_=NULL;}
		if(dvc_dir_pool){delete dvc_dir_pool;dvc_dir_pool=NULL;}
		if(dvc_food_pool){delete dvc_food_pool;dvc_food_pool=NULL;}
		break;
	case 1:
		dvc_pos_pool_->DeleteContents();
		dvc_dir_pool->DeleteContents();
		dvc_food_pool->DeleteContents();
		break;
	};
}
void Pocman::DestroyMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;
	if(gpu_memory_pool_){delete gpu_memory_pool_;gpu_memory_pool_=NULL;}
	DestoryDvcMemoryPools<<<1,1,1>>>(mode);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

DEVICE Dvc_ValuedAction Dvc_PocmanLegalParticleLowerBound::Value(int Sid,
				Dvc_State* particles,
				Dvc_RandomStreams& streams,
				Dvc_History& history){
	const Dvc_PocmanState& pocstate =
				*static_cast<const Dvc_PocmanState*>(particles);
	int action=0;

	if(threadIdx.x==0 && threadIdx.y==0)//only for particle 0
	{
		bool legal[4];int num=0;DvcCoord newpos;
		for (int a = 0; a < 4; ++a) {
			legal[a]=false;
			// DvcCoord newpos= pocman_->NextPos(pocstate.pocman_pos, a);
			if (pocstate.pocman_pos.x == 0 && pocstate.pocman_pos.y == passage_y_ && a == Dvc_Compass::WEST)
				newpos = DvcCoord(maze_size_x_ - 1, pocstate.pocman_pos.y);
			else if (pocstate.pocman_pos.x == maze_size_x_ - 1 && pocstate.pocman_pos.y == passage_y_
				&& a == Dvc_Compass::EAST)
				newpos = DvcCoord(0, pocstate.pocman_pos.y);
			else
				newpos = pocstate.pocman_pos + Dvc_Compass::GetDirections(a);

			if (newpos.x >= 0 && newpos.y >= 0 && newpos.x < maze_size_x_
					&& newpos.y < maze_size_y_
					&& Dvc_CheckFlag(maze_[newpos.y * maze_size_x_ + newpos.x],Dvc_Pocman::E_PASSABLE))
				;
			else
				{newpos.x=-1;newpos.y=-1;}
			if (newpos.x >= 0 && newpos.y >= 0)
				{legal[a]=true;num++;}

		}
		int chosen=Dvc_random->NextInt(num, Sid);
		num=-1;
		for (int a = 0; a < 4; ++a)
		{
			if(legal[a])
				num++;
			if(num==chosen)
			{
				action=a;break;
			}
		}
	}
	return Dvc_ValuedAction(action,
			reward_die_
				+ reward_default_ / (1 - Dvc_Globals::Dvc_Discount(Dvc_config)));
}
DEVICE float Dvc_PocmanApproxScenarioUpperBound::Value( const Dvc_State* particles, int scenarioID, Dvc_History& history)
{
	const Dvc_PocmanState& state =
			(static_cast<const Dvc_PocmanState*>(particles))[scenarioID];
	float value = 0;

	int max_dist = 0;

	for (int i = 0; i < maze_size_x_; i++)
		for (int j = 0; j < maze_size_y_; j++)
	{
		//if (state.food[i] != 1)
		if(!Dvc_CheckFlag(state.food[j],i))
			continue;

		DvcCoord food_pos = DvcCoord(i, j);
		int dist = DvcCoord::ManhattanDistance(state.pocman_pos, food_pos);
		value += reward_eat_food_ * Dvc_Globals::Dvc_Discount(Dvc_config,dist);
		max_dist = max(max_dist, dist);
	}

	// Clear level
	value += reward_clear_level_ * pow(Dvc_config->discount, max_dist);

	// Default move-reward
	value += reward_default_ * (Dvc_Globals::Dvc_Discount(Dvc_config) < 1
			? (1 - Dvc_Globals::Dvc_Discount(Dvc_config,max_dist)) / (1 - Dvc_Globals::Dvc_Discount(Dvc_config))
			: max_dist);

	// If pocman is chasing a ghost, encourage it
	if (state.power_steps > 0 && history.Size() &&
			(history.LastObservation() & 15) != 0) {
		int act = history.LastAction();
		int obs = history.LastObservation();
		if (Dvc_CheckFlag(obs, act)) {
			bool seen_ghost = false;
			for (int dist = 1; !seen_ghost; dist++) {
				DvcCoord ghost_pos = state.pocman_pos + Dvc_Compass::GetDirections(act) * dist;
				for (int g = 0; g < num_ghosts_; g++)
					if (state.ghost_pos[g] == ghost_pos) {
						value += reward_eat_ghost_ * Dvc_Globals::Dvc_Discount(Dvc_config,dist);
						seen_ghost = true;
						break;
					}
			}
		}
	}

	// Ghost penalties
	float dist = 0;
	for (int g = 0; g < num_ghosts_; g++)
		dist += DvcCoord::ManhattanDistance(state.pocman_pos, state.ghost_pos[g]);
	value += reward_die_ * __powf(Dvc_Globals::Dvc_Discount(Dvc_config), dist / num_ghosts_);

	// Penalize for doubling back, but not so much as to prefer hitting a wall
	if (history.Size() >= 2 &&
			Dvc_Compass::Opposite(history.Action(history.Size() - 1))
			== history.Action(history.Size() - 2))
		value += reward_hit_wall_ / 2;

	return value;
}
} // namespace despot
