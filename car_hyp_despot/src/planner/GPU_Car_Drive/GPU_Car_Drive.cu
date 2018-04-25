#include "GPU_Car_Drive.h"

#include <ped_pomdp.h>
#include <cuda_runtime_api.h>
#include <despot/util/coord.h>
#include <driver_types.h>
#include <stddef.h>
#include "despot/GPUutil/GPUmemorypool.h"
#include <despot/GPUcore/thread_globals.h>

#include "GPU_CarUpperBound.h"
#define THREADDIM 128
using namespace std;

static Dvc_PomdpState* Dvc_particles=NULL;

//Dvc_State** Dvc_stepped_particles_all_a=NULL;
static GPU_MemoryPool<Dvc_PomdpState>* gpu_memory_pool_=NULL;
static GPU_MemoryPool<Dvc_PedStruct>* gpu_ped_pool_=NULL;

static Dvc_PedStruct **Dvc_tempPeds=NULL;
static Dvc_PedStruct **Hst_tempPeds=NULL;

static float** tmp_result=NULL;
static int** tempHostID=NULL;
static float** temp_weight=NULL;
static Dvc_PomdpState** tmp_state=NULL;


DEVICE Dvc_Path* path=NULL;
DEVICE Dvc_COORD* goals=NULL;
DEVICE double freq=0;
DEVICE double in_front_angle_cos=0;

//const DvcCoord Dvc_NavCompass::DIRECTIONS[] = {  DvcCoord(0, 1), DvcCoord(1, 0),DvcCoord(0, -1),
//	DvcCoord(-1, 0), DvcCoord(1, 1), DvcCoord(1, -1), DvcCoord(-1, -1), DvcCoord(-1, 1) };
//const string Dvc_NavCompass::CompassString[] = { "North", "East","South", "West",
//	"NE", "SE", "SW", "NW" };
/* ==============================================================================
 * Dvc_PomdpState class
 * ==============================================================================*/

DEVICE Dvc_PomdpState::Dvc_PomdpState():num(0), peds(NULL)
{
}

DEVICE Dvc_PomdpState::Dvc_PomdpState(const Dvc_PomdpState& src)
{
	*this=src;
}

__global__ void CopyPeds(Dvc_PomdpState* Dvc,  Dvc_PedStruct* src)
{
	int scenarioID=blockIdx.x;
	int ped_id=threadIdx.x;
	Dvc_PomdpState* Dvc_i=Dvc+scenarioID;
	Dvc_PedStruct* Src_i=src+scenarioID*Dvc_ModelParams::N_PED_IN;
	/*if(x==0 && y==0)
	{
		Dvc->cells=(bool*)malloc(Dvc->sizeX_*Dvc->sizeY_*sizeof(bool));
	}
	__syncthreads();*/

	if(ped_id<Dvc_i->num)
		Dvc_i->peds[ped_id]=Src_i[ped_id];

}

__global__ void CopyPeds_back(const Dvc_PomdpState* Dvc,  Dvc_PedStruct* des)
{
	int scenarioID=blockIdx.x;
	int ped_id=threadIdx.x;
	const Dvc_PomdpState* Dvc_i=Dvc+scenarioID;
	Dvc_PedStruct* Des_i=des+scenarioID*Dvc_ModelParams::N_PED_IN;
	/*if(x==0 && y==0)
	{
		Dvc->cells=(bool*)malloc(Dvc->sizeX_*Dvc->sizeY_*sizeof(bool));
	}
	__syncthreads();*/
	/*if(GPUDoPrint && Dvc_i->scenario_id==GPUPrintPID && threadIdx.x==0){
		int sid= Dvc_i->scenario_id;
		const Dvc_PomdpState* dvc=Dvc_i;
		printf("(GPU) copy peds: scenario=%d", sid);
		printf(" from global memory %d \n",dvc);
		printf("Peds:\n");
		int pos=Dvc_i->car.pos;
		printf("car pox= %d ",pos);
		printf("dist=%f\n",Dvc_i->car.dist_travelled);
		printf("car vel= %f\n",Dvc_i->car.vel);

		for(int i=0;i<Dvc_i->num;i++)
		{
			printf("ped %d pox_x= %f pos_y=%f\n",i,
					Dvc_i->peds[i].pos.x,Dvc_i->peds[i].pos.y);
		}
	}*/

	if(ped_id<Dvc_i->num)
		Des_i[ped_id]=Dvc_i->peds[ped_id];
}

HOST void Dvc_PomdpState::CopyToGPU(Dvc_PomdpState* Dvc, int scenarioID, const PomdpState* Hst, bool copy_cells)
{
	Dvc[scenarioID].car.dist_travelled=Hst->car.dist_travelled;
	Dvc[scenarioID].car.pos=Hst->car.pos;
	Dvc[scenarioID].car.vel=Hst->car.vel;
	Dvc[scenarioID].num=Hst->num;
	Dvc[scenarioID].weight=Hst->weight;
	Dvc[scenarioID].state_id=Hst->state_id;
	Dvc[scenarioID].scenario_id=Hst->scenario_id;


	int Data_block_size=ModelParams::N_PED_IN;
	int hst_size=sizeof(PedStruct);

	if(use_multi_thread_ && cuda_streams)
	{
		memcpy((void*)(Hst_tempPeds[GetCurrentStream()]+Data_block_size*scenarioID),
				(const void*)Hst->peds,
				Data_block_size*sizeof(Dvc_PedStruct));
	}
	else
	{
		//cout<<"scenarioID="<<scenarioID<<endl;
		//cout<<"Hst_tempPeds[0][Data_block_size*scenarioID].goal="<<Hst_tempPeds[0][Data_block_size*scenarioID].goal<<endl;
		//cout<<"Hst->peds[ModelParams::N_PED_IN-1].goal="<<Hst->peds[ModelParams::N_PED_IN-1].goal<<endl;
		memcpy((void*)(Hst_tempPeds[0]+Data_block_size*scenarioID),
				(const void*)Hst->peds,
				Data_block_size*sizeof(Dvc_PedStruct));
	}
}
HOST void Dvc_PomdpState::CopyToGPU2(Dvc_PomdpState* Dvc, int NumParticles, bool copy_cells)
{

	if(copy_cells)
	{
		int Data_size=NumParticles*ModelParams::N_PED_IN;
		dim3 grid1(NumParticles,1);dim3 threads1(ModelParams::N_PED_IN,1);
		if(use_multi_thread_ && cuda_streams)
		{
			HANDLE_ERROR(cudaMemcpyAsync((void*)Dvc_tempPeds[GetCurrentStream()],
					(const void*)Hst_tempPeds[GetCurrentStream()],
					Data_size*sizeof(Dvc_PedStruct),
					cudaMemcpyHostToDevice,((cudaStream_t*)cuda_streams)[GetCurrentStream()]));

			CopyPeds<<<grid1, threads1, 0, ((cudaStream_t*)cuda_streams)[GetCurrentStream()]>>>
					(Dvc,Dvc_tempPeds[GetCurrentStream()]);
			//AdvanceStreamCounter(1);
		}
		else
		{
			HANDLE_ERROR(cudaMemcpy((void*)Dvc_tempPeds[0],
					(const void*)Hst_tempPeds[0],
					Data_size*sizeof(Dvc_PedStruct),
					cudaMemcpyHostToDevice));
			CopyPeds<<<grid1, threads1>>>(Dvc,Dvc_tempPeds[0]);
			//cudaDeviceSynchronize();
		}


	}
}

HOST void Dvc_PomdpState::ReadBackToCPU(const Dvc_PomdpState* Dvc,PomdpState* Hst, bool copy_cells)
{
	int ThreadID=0;
	if(use_multi_thread_)
		ThreadID=MapThread(this_thread::get_id());
	HANDLE_ERROR(cudaMemcpy((void*)tmp_state[ThreadID], (const void*)Dvc, sizeof(Dvc_PomdpState), cudaMemcpyDeviceToHost));
	Hst->car.dist_travelled=tmp_state[ThreadID]->car.dist_travelled;
	Hst->car.pos=tmp_state[ThreadID]->car.pos;
	Hst->car.vel=tmp_state[ThreadID]->car.vel;

	Hst->num=tmp_state[ThreadID]->num;
	Hst->weight=tmp_state[ThreadID]->weight;
	Hst->state_id=tmp_state[ThreadID]->state_id;
	Hst->scenario_id=tmp_state[ThreadID]->scenario_id;
}
HOST void Dvc_PomdpState::ReadBackToCPU2(const Dvc_PomdpState* Dvc,
		std::vector<State*> Hst, bool copy_cells)
{
	if(copy_cells)
	{
		int ThreadID=0;
		if(use_multi_thread_)
			ThreadID=MapThread(this_thread::get_id());
		int NumParticles=Hst.size();
		int Data_size=NumParticles*ModelParams::N_PED_IN;
		dim3 grid1(NumParticles,1);dim3 threads1(ModelParams::N_PED_IN,1);
		if(use_multi_thread_ && cuda_streams)
		{
			CopyPeds_back<<<grid1, threads1, 0, ((cudaStream_t*)cuda_streams)[ThreadID]>>>
					(Dvc,Dvc_tempPeds[ThreadID]);
			//AdvanceStreamCounter(1);
			HANDLE_ERROR(cudaMemcpyAsync((void*)Hst_tempPeds[ThreadID],
					(const void*)Dvc_tempPeds[ThreadID],
					Data_size*sizeof(Dvc_PedStruct),
					cudaMemcpyDeviceToHost,((cudaStream_t*)cuda_streams)[ThreadID]));
			cudaStreamSynchronize(((cudaStream_t*)cuda_streams)[ThreadID]);
		}
		else
		{
			CopyPeds_back<<<grid1, threads1>>>(Dvc,Dvc_tempPeds[0]);
			HANDLE_ERROR(cudaMemcpy((void*)Hst_tempPeds[0],
					(const void*)Dvc_tempPeds[0],
					Data_size*sizeof(Dvc_PedStruct),
					cudaMemcpyDeviceToHost));
			//cudaDeviceSynchronize();
		}


		int Data_block_size=ModelParams::N_PED_IN;
		int hst_size=sizeof(PedStruct);
		for(int i=0;i<NumParticles;i++)
		{
			PomdpState* car_state=static_cast<PomdpState*>(Hst[i]);

			if(use_multi_thread_ && cuda_streams)
			{
				memcpy((void*)car_state->peds,
						(const void*)(Hst_tempPeds[ThreadID]+Data_block_size*i),
						Data_block_size*sizeof(Dvc_PedStruct));
			}
			else
			{
				memcpy((void*)car_state->peds,
						(const void*)(Hst_tempPeds[0]+Data_block_size*i),
						Data_block_size*sizeof(Dvc_PedStruct));
			}
		}
	}
}


__global__ void CopyParticle(Dvc_PomdpState* des,Dvc_PomdpState* src,int num_particles)
{
	//int pos=threadIdx.x;
	int pos=blockIdx.x*blockDim.x+threadIdx.x;

	//int y=threadIdx.y;
	if(pos < num_particles)
	{
		Dvc_PomdpState* src_i=src+pos;
		Dvc_PomdpState* des_i=des+pos;
		des_i->car.dist_travelled=src_i->car.dist_travelled;
		des_i->car.pos=src_i->car.pos;
		des_i->car.vel=src_i->car.vel;
		des_i->num=src_i->num;
		des_i->weight=src_i->weight;
		des_i->state_id=src_i->state_id;
		des_i->scenario_id=src_i->scenario_id;

		for(int i=0;i</*Dvc_ModelParams::N_PED_IN*/src_i->num;i++)
		{
			des_i->peds[i].goal=src_i->peds[i].goal;
			des_i->peds[i].id=src_i->peds[i].id;
			des_i->peds[i].pos.x=src_i->peds[i].pos.x;
			des_i->peds[i].pos.y=src_i->peds[i].pos.y;
			des_i->peds[i].vel=src_i->peds[i].vel;
		}
	}
}

__global__ void CopyParticles(Dvc_PomdpState* des,Dvc_PomdpState* src,
		float* weight,int* IDs,int num_particles,
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
	if(pos < num_particles)
	{
		bool error=false;
		int scenarioID=IDs[pos];
		Dvc_PomdpState* src_i=src+scenarioID;//src is a full length array for all particles
		Dvc_PomdpState* des_i=des+pos;//des is short, only for the new partition

		des_i->car.dist_travelled=src_i->car.dist_travelled;
		des_i->car.pos=src_i->car.pos;
		des_i->car.vel=src_i->car.vel;
		des_i->num=src_i->num;
		des_i->weight=src_i->weight;
		des_i->state_id=src_i->state_id;
		des_i->scenario_id=src_i->scenario_id;

		for(int i=0;i</*Dvc_ModelParams::N_PED_IN*/src_i->num;i++)
		{
			des_i->peds[i].goal=src_i->peds[i].goal;
			des_i->peds[i].id=src_i->peds[i].id;
			des_i->peds[i].pos.x=src_i->peds[i].pos.x;
			des_i->peds[i].pos.y=src_i->peds[i].pos.y;
			des_i->peds[i].vel=src_i->peds[i].vel;
		}

		atomicAdd(weight, des_i->weight);
	}
}

void PedPomdp::CreateMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;

	if(gpu_memory_pool_==NULL)
		gpu_memory_pool_=new GPU_MemoryPool<Dvc_PomdpState>;
	if(gpu_ped_pool_==NULL)
		gpu_ped_pool_=new GPU_MemoryPool<Dvc_PedStruct>;


	/*switch(mode)
	{
	case 0:
		if(cell_memory_pool_==NULL && use_multi_thread_)
			cell_memory_pool_=new Array_MemoryPool<bool>(use_multi_thread_);
		break;
	case 1:
		if(cell_memory_pool_ && use_multi_thread_){ cell_memory_pool_->ResetChuncks();}; break;
	}
*/
	//gpu_memory_pool_->SetChunkSize(chunk_size);
}
void PedPomdp::DestroyMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;
	switch(mode)
	{
		case 0:
			if(gpu_memory_pool_){delete gpu_memory_pool_;gpu_memory_pool_=NULL;}
			if(gpu_ped_pool_){delete gpu_ped_pool_;gpu_ped_pool_=NULL;}
			break;
		case 1:
			if(gpu_memory_pool_ ){ gpu_memory_pool_->ResetChuncks();};
			if(gpu_ped_pool_ ){ gpu_ped_pool_->ResetChuncks();};
			break;
	}

	/*switch(mode)
	{
	case 0:
		if(cell_memory_pool_ && use_multi_thread_){delete cell_memory_pool_;cell_memory_pool_=NULL;}; break;
	case 1:
		if(cell_memory_pool_&& use_multi_thread_){ cell_memory_pool_->ResetChuncks();}; break;
	}*/
}
__global__ void LinkPeds(Dvc_PomdpState* state, Dvc_PedStruct* peds_memory, int numParticles)
{
	//int i=threadIdx.x+blockIdx.x*blockDim.x;
	for(int i=0;i<numParticles;i++)
	{
		state[i].peds=peds_memory+i*Dvc_ModelParams::N_PED_IN;
	}
}
Dvc_State* PedPomdp::AllocGPUParticles(int numParticles, int alloc_mode) const
{//numParticles==num_Scenarios
	clock_t start=clock();
	dim3 grid((numParticles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);
	int num_threads=1;

	if(use_multi_thread_)
	{
		num_threads=NUM_THREADS;
	}
	Dvc_PedStruct* peds_tmp=NULL;
	switch(alloc_mode)
	{
	case 0:
		CreateMemoryPool(0);
		peds_tmp=gpu_ped_pool_->Allocate((1+NumActions()*num_threads)*numParticles*ModelParams::N_PED_IN);

		HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles, numParticles*sizeof(Dvc_PomdpState)));
		LinkPeds<<<dim3(numParticles,1), dim3(ModelParams::N_PED_IN,1)>>>(Dvc_particles, peds_tmp, numParticles);
		HANDLE_ERROR(cudaDeviceSynchronize());

		Dvc_tempPeds=new Dvc_PedStruct*[num_threads];
		Hst_tempPeds=new Dvc_PedStruct*[num_threads];
		for(int i=0;i<num_threads;i++)
		{
			HANDLE_ERROR(cudaMalloc((void**)&Dvc_tempPeds[i],numParticles*ModelParams::N_PED_IN*sizeof(Dvc_PedStruct) ));
			HANDLE_ERROR(cudaHostAlloc((void**)&Hst_tempPeds[i],numParticles*ModelParams::N_PED_IN*sizeof(Dvc_PedStruct),0 ));
		}
		cout<<"numParticles="<<numParticles<<endl;

		Dvc_stepped_particles_all_a=new Dvc_State*[num_threads];
		for(int i=0;i<num_threads;i++)
		{
			HANDLE_ERROR(cudaMalloc((void**)&Dvc_stepped_particles_all_a[i],
					NumActions()*numParticles*sizeof(Dvc_PomdpState)));
			LinkPeds<<<dim3(numParticles,1), dim3(ModelParams::N_PED_IN,1)>>>
					(static_cast<Dvc_PomdpState*>(Dvc_stepped_particles_all_a[i]),
					peds_tmp+(1+NumActions()*i)*numParticles*ModelParams::N_PED_IN,
					NumActions()*numParticles);
		}
		//Record the ped memory used by the fixed lists
		//never reuse these memory for vnode particles
		gpu_ped_pool_->RecordHead();

		tempHostID=new int*[num_threads];
		for(int i=0;i<num_threads;i++)
		{
			cudaHostAlloc(&tempHostID[i],numParticles*sizeof(int),0);
		}

		temp_weight=new float*[num_threads];
		for(int i=0;i<num_threads;i++)
			cudaHostAlloc(&temp_weight[i],1*sizeof(float),0);

		tmp_result=new float*[num_threads];
		for(int i=0;i<num_threads;i++)
			HANDLE_ERROR(cudaMalloc(&tmp_result[i], sizeof(float)));

		tmp_state=new Dvc_PomdpState*[num_threads];

		for(int i=0;i<num_threads;i++)
			HANDLE_ERROR(cudaHostAlloc((void**)&tmp_state[i],1*sizeof(Dvc_PomdpState),0));

		return Dvc_particles;
	case 1:
		CreateMemoryPool(1);
		return Dvc_particles;
	case 2:
		Dvc_PomdpState* states_tmp=gpu_memory_pool_->Allocate(numParticles);
		peds_tmp=gpu_ped_pool_->Allocate(numParticles*ModelParams::N_PED_IN);

		LinkPeds<<<dim3(numParticles,1), dim3(ModelParams::N_PED_IN,1)>>>(states_tmp, peds_tmp, numParticles);
		HANDLE_ERROR(cudaDeviceSynchronize());
		return states_tmp;
	};

	cout<<"GPU particles alloc time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl;
}


void PedPomdp::CopyGPUParticles(Dvc_State* des,Dvc_State* src,int src_offset,
		int* IDs,int num_particles,bool interleave,
		Dvc_RandomStreams* streams, int stream_pos,
		void* CUDAstream, int shift) const
{
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);
	if(num_particles<THREADDIM)
	{
		grid.x=1;grid.y=1;threads.x=num_particles;
	}

	int ThreadID=0;
	if(use_multi_thread_)
		ThreadID=MapThread(this_thread::get_id());
	if(CUDAstream)
	{
		CopyParticles<<<grid, threads,0, *(cudaStream_t*)CUDAstream>>>(static_cast<Dvc_PomdpState*>(des),
				static_cast<Dvc_PomdpState*>(src)+src_offset,tmp_result[(ThreadID+shift)%NUM_THREADS],
				IDs,num_particles, streams,stream_pos);
		if(!interleave)
			;//HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)CUDAstream));
	}
	else
	{
		CopyParticles<<<grid, threads,0, 0>>>(static_cast<Dvc_PomdpState*>(des),
				static_cast<Dvc_PomdpState*>(src)+src_offset,tmp_result[ThreadID],
				IDs,num_particles, streams,stream_pos);
		if(!interleave)
			HANDLE_ERROR(cudaDeviceSynchronize());
	}
}

void PedPomdp::CopyGPUWeight1(void* cudaStream, int shift) const
{
	int ThreadID=0;
	if(use_multi_thread_)
		ThreadID=MapThread(this_thread::get_id());
	if(cudaStream==NULL)
	{
		;
	}
	else
	{
		HANDLE_ERROR(cudaMemcpyAsync(temp_weight[(ThreadID+shift)%NUM_THREADS], tmp_result[(ThreadID+shift)%NUM_THREADS], sizeof(float),cudaMemcpyDeviceToHost, *(cudaStream_t*)cudaStream));
	}
}
float PedPomdp::CopyGPUWeight2(void* cudaStream, int shift) const
{
	int ThreadID=0;
	if(use_multi_thread_)
		ThreadID=MapThread(this_thread::get_id());
	if(cudaStream==NULL)
	{
		float particle_weight=0;
		HANDLE_ERROR(cudaMemcpy(&particle_weight, tmp_result[ThreadID], sizeof(float),cudaMemcpyDeviceToHost));
		return particle_weight;
	}
	else
	{
		HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)cudaStream));
		if(*temp_weight[(ThreadID+shift)%NUM_THREADS]>1)
			Global_print_value(this_thread::get_id(),*temp_weight[(ThreadID+shift)%NUM_THREADS],"Wrong weight from CopyGPUWeight");
		return *temp_weight[(ThreadID+shift)%NUM_THREADS];
	}
}

Dvc_State* PedPomdp::GetGPUParticles() const
{
	return Dvc_particles;
}
Dvc_State* PedPomdp::CopyToGPU(const std::vector<State*>& particles, bool copy_cells) const
{
	//Dvc_PomdpState* Dvc_particles;
	//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particles, particles.size()*sizeof(Dvc_PomdpState)));
	//dvc_particles should be managed device memory
	auto start = Time::now();

	for (int i=0;i<particles.size();i++)
	{
		const PomdpState* src=static_cast<const PomdpState*>(particles[i]);
		//Dvc_particles[i].Assign(src);
		Dvc_PomdpState::CopyToGPU(Dvc_particles,src->scenario_id,src);
		//Dvc_PomdpState::CopyToGPU(NULL,src->scenario_id,src, false);//copy to Dvc_particles_copy, do not copy cells, leave it a NULL pointer
	}
	Dvc_PomdpState::CopyToGPU2(Dvc_particles,particles.size());

	//for (int i=0;i<particles.size();i++)
	//{
		//const Dvc_PomdpState* src=static_cast<const Dvc_PomdpState*>(particles[i]);
		//Dvc_particles[i].Assign(src);
		//Dvc_PomdpState::CopyToGPU2(Dvc_particles,src->scenario_id,src);
		//Dvc_PomdpState::CopyToGPU(NULL,src->scenario_id,src, false);//copy to Dvc_particles_copy, do not copy cells, leave it a NULL pointer
	//}
	//cudaDeviceSynchronize();
	//cout<<"GPU particles copy time:"<<chrono::duration_cast<sec>(Time::now() - start).count()<<endl;

	return Dvc_particles;
}
void PedPomdp::CopyToGPU(const std::vector<int>& particleIDs, int* Dvc_ptr, void *CUDAstream) const
{
	if(CUDAstream)
	{
		int ThreadID=MapThread(this_thread::get_id());
		memcpy(tempHostID[ThreadID],particleIDs.data(),particleIDs.size()*sizeof(int));

		//HANDLE_ERROR(cudaHostRegister((void*)particleIDs.data(),particleIDs.size()*sizeof(int),cudaHostRegisterPortable));
		HANDLE_ERROR(cudaMemcpyAsync(Dvc_ptr,tempHostID[ThreadID],particleIDs.size()*sizeof(int), cudaMemcpyHostToDevice,*(cudaStream_t*)CUDAstream));
		//HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)CUDAstream));
		//HANDLE_ERROR(cudaHostUnregister((void*)particleIDs.data()));
	}
	else
	{
		HANDLE_ERROR(cudaMemcpy(Dvc_ptr,particleIDs.data(),particleIDs.size()*sizeof(int), cudaMemcpyHostToDevice));
	}
	//return Dvc_particleIDs;
}


void PedPomdp::DeleteGPUParticles( int num_particles) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);

	HANDLE_ERROR(cudaFree(static_cast<Dvc_PomdpState*>(Dvc_particles)));

	//HANDLE_ERROR(cudaFree(Dvc_particleIDs));

	//HANDLE_ERROR(cudaFree(tmp_result));
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
	for(int i=0;i<num_threads;i++)
	{
		cudaFreeHost(tempHostID[i]);
	}
	delete [] tempHostID;
	for(int i=0;i<num_threads;i++)
	{
		cudaFreeHost(temp_weight[i]);
	}
	delete [] temp_weight;
	for(int i=0;i<num_threads;i++)
	{
		cudaFree(tmp_result[i]);
	}
	delete [] tmp_result;

	for(int i=0;i<num_threads;i++)
	{
		cudaFree(Dvc_tempPeds[i]);
		cudaFreeHost(Hst_tempPeds[i]);
		cudaFreeHost(tmp_state[i]);
	}
	delete [] Dvc_tempPeds;
	delete [] Hst_tempPeds;
	delete [] tmp_state;

	/*FreeParticleCopy<<<grid, threads>>>(num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());*/
	//HANDLE_ERROR(cudaFree(static_cast<Dvc_PomdpState*>(Dvc_particles_copy)));
}
void PedPomdp::DeleteGPUParticles(Dvc_State* particles, int num_particles) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);

	HANDLE_ERROR(cudaFree(static_cast<Dvc_PomdpState*>(particles)));
}

DEVICE int minStepToGoal(const Dvc_PomdpState& state) {
    double d = Dvc_ModelParams::GOAL_TRAVELLED - state.car.dist_travelled;
    if (d < 0) d = 0;
    return int(ceil(d / (Dvc_ModelParams::VEL_MAX/freq)));
}

DEVICE bool InRectangle(float HNx, float HNy, float HMx, float HMy, float front_margin, float back_margin, float side_margin) {
	float HLx = - HNy, // direction after 90 degree anticlockwise rotation
				 HLy = HNx;

	float HM_HN = HMx * HNx + HMy * HNy, // HM . HN
				 HN_HN = HNx * HNx + HNy * HNy; // HN . HN
	if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
		return false;
	if (HM_HN <= 0 && HM_HN * HM_HN > HN_HN * back_margin * back_margin)
		return false;

	float HM_HL = HMx * HLx + HMy * HLy, // HM . HL
				 HL_HL = HLx * HLx + HLy * HLy; // HL . HL
	return HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
}
DEVICE bool Dvc_inCollision(float Mx, float My, float Hx, float Hy, float Nx, float Ny) {

	float HNx = Nx - Hx, // car direction
				 HNy = Ny - Hy;
	float HMx = Mx - Hx,
				 HMy = My - Hy;

	/*float car_width = 1.2,
				 car_length = 2.2;
	float safe_margin = 0.3,
				 side_margin = car_width / 2.0 + safe_margin,
				 front_margin = safe_margin,
				 back_margin = car_length + safe_margin;
*/

/// for golfcart
/*	double car_width = 0.87,
			car_length = 1.544;

	double safe_margin = 0.92, side_safe_margin = 0.4, back_safe_margin = 0.33,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = car_length/2.0 + safe_margin,
				 back_margin = car_length/2.0 + back_safe_margin;*/

/// for audi r8
	double car_width = 2.0,
				 car_length = 4.4;

				 double safe_margin = 0.8, side_safe_margin = 0.35, back_safe_margin = 0.2,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = 3.6 + safe_margin,
				 back_margin = 0.8 + back_safe_margin;

	return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);
}

DEVICE bool Dvc_inCollision(const Dvc_PomdpState& state) {
    const int car = state.car.pos;
	const Dvc_COORD& car_pos = path->way_points_[car];
	const Dvc_COORD& forward_pos = path->way_points_[path->forward(car, 1.0)];

    for(int i=0; i<state.num; i++) {
        const Dvc_COORD& pedpos = state.peds[i].pos;
        if(Dvc_inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)) {
            return true;
        }
    }
    return false;
}

DEVICE float Dvc_PedPomdpParticleUpperBound1::Value(
		const Dvc_State* particles, int scenarioID, Dvc_History& history) {
	const Dvc_PomdpState* pedpomdp_state =
			static_cast<const Dvc_PomdpState*>(particles) + scenarioID;
	//When using trial upper bound, unblock the line below
	return Dvc_ModelParams::GOAL_REWARD / (1 - Dvc_Globals::Dvc_Discount(Dvc_config));
	//When using smart upper bound, unblock the lines below
	/*__shared__ int iscollision[32];
    iscollision[threadIdx.x]=false;
    __syncthreads();
	//if (Dvc_inCollision(*pedpomdp_state))
	const int car = pedpomdp_state->car.pos;
	const Dvc_COORD& car_pos = path->way_points_[car];
	const Dvc_COORD& forward_pos = path->way_points_[path->forward(car, 1.0)];
	//for(int i=0; i<pedpomdp_state.num; i++)
	if(threadIdx.y<pedpomdp_state->num){
		const Dvc_COORD& pedpos = pedpomdp_state->peds[threadIdx.y].pos;
		//Dvc_inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)
		bool collide_ped=false;
		float HNx = forward_pos.x - car_pos.x, // car direction
					 HNy = forward_pos.y - car_pos.y;
		float HMx = pedpos.x - car_pos.x,
					 HMy = pedpos.y - car_pos.y;

		float side_margin = 1.2 / 2.0 + 0.3,
					 front_margin = 0.3,
					 back_margin = 2.2 + 0.3;

		//return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);
		float HLx = - HNy, // direction after 90 degree anticlockwise rotation
					 HLy = HNx;

		float HM_HN = HMx * HNx + HMy * HNy, // HM . HN
					 HN_HN = HNx * HNx + HNy * HNy; // HN . HN
		if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
			collide_ped = false;
		else if (HM_HN <= 0 && HM_HN * HM_HN > HN_HN * back_margin * back_margin)
			collide_ped = false;
		else
		{
			float HM_HL = HMx * HLx + HMy * HLy, // HM . HL
					 HL_HL = HLx * HLx + HLy * HLy; // HL . HL
			collide_ped= HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
		}
		atomicOr(iscollision+threadIdx.x, collide_ped);
	}

	if(iscollision[threadIdx.x])
		return Dvc_ModelParams::CRASH_PENALTY *
	    		 (pedpomdp_state->car.vel * pedpomdp_state->car.vel
	    		+ Dvc_ModelParams::REWARD_BASE_CRASH_VEL);

	//int min_step = minStepToGoal(*pedpomdp_state);

	int min_step= int(ceil(
			max(Dvc_ModelParams::GOAL_TRAVELLED - pedpomdp_state->car.dist_travelled,0.0f)
			/ (Dvc_ModelParams::VEL_MAX/freq)));
	return Dvc_ModelParams::GOAL_REWARD * Dvc_Globals::Dvc_Discount(Dvc_config,min_step);*/
}

DEVICE Dvc_PedPomdp::Dvc_PedPomdp(/*int size, int obstacles*/)// :
{

}


DEVICE Dvc_PedPomdp::~Dvc_PedPomdp()
{
	//half_efficiency_distance_ = 20;

}


static
__device__ float RandGeneration(unsigned long long int *record, float seed)
{
	//float value between 0 and 1
	//seed have to be within 0 and 1
	record[0]+=seed*ULLONG_MAX/1000.0;
	//unsigned long long int record_l=atomicAdd(record,t);//atomic add returns old value of record

	//record_l+=t;
	float record_f=0;
	/*randMethod(record_l,record_f);*/
	record[0]*=16807;
	record[0]=record[0]%2147483647;
	record_f=((double)record[0])/2147483647;
	return record_f;
}
DEVICE bool isLocalGoal(const Dvc_PomdpState& state)
{
	return state.car.dist_travelled > Dvc_ModelParams::GOAL_TRAVELLED
			|| state.car.pos >= path->size_-1;
}


DEVICE void RobStep(Dvc_CarStruct &car, float random) {
    double dist = car.vel / freq;
    //double dist_l=max(0.0,dist-Dvc_ModelParams::AccSpeed/freq);
    //double dist_r=min(Dvc_ModelParams::VEL_MAX,dist+Dvc_ModelParams::AccSpeed/freq);
    //double sample_dist=random.NextDouble(dist_l,dist_r);
    //int nxt = Path->forward(car.pos, sample_dist);
    int nxt = path->forward(car.pos, dist);
    car.pos = nxt;
    car.dist_travelled += dist;
}

DEVICE void RobVelStep(Dvc_CarStruct &car, double acc, float random) {
    const double N = Dvc_ModelParams::NOISE_ROBVEL;
    if (N>0) {
        double prob = random/*.NextDouble()*/;
        if (prob > N) {
            car.vel += acc / freq;
        }
    } else {
        car.vel += acc / freq;
    }

	car.vel = max(min(car.vel, Dvc_ModelParams::VEL_MAX), 0.0);

	return;
}

DEVICE void PedStep(Dvc_PedStruct &ped, float random, unsigned long long int &Temp) {
    const Dvc_COORD& goal = goals[ped.goal];
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	Dvc_Vector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
    double a = goal_vec.GetAngle();
    double noise = sqrt(-2 * log(random));
    random=RandGeneration(&Temp, random);
    noise *= cos(2 * M_PI * random)* Dvc_ModelParams::NOISE_GOAL_ANGLE;
    a += noise;

	//TODO noisy speed
    Dvc_Vector move(a, ped.vel/freq, 0);
    ped.pos.x += move.dw;
    ped.pos.y += move.dh;
    return;
}


DEVICE bool Dvc_PedPomdp::Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
	int* obs) {

	Dvc_PomdpState& pedpomdp_state = static_cast<Dvc_PomdpState&>(state);//copy contents, link cells to existing ones
	__shared__ int iscollision[32];

	/*if(GPUDoPrint && pedpomdp_state.scenario_id==GPUPrintPID && blockIdx.x==0 && threadIdx.y==0){
		printf("(GPU) Before step: scenario=%d \n", pedpomdp_state.scenario_id);
		printf("Before step:\n");
		int pos=pedpomdp_state.car.pos;
		printf("car pox= %d ",pos);
		printf("dist=%f\n",pedpomdp_state.car.dist_travelled);
		printf("car vel= %f\n",pedpomdp_state.car.vel);

		for(int i=0;i<pedpomdp_state.num;i++)
		{
			printf("ped %d pox_x= %f pos_y=%f\n",i,
					pedpomdp_state.peds[i].pos.x,pedpomdp_state.peds[i].pos.y);
		}
	}*/
	bool terminal=false;
	reward = 0;

	unsigned long long int Temp=123456;
	if(threadIdx.y==0)
	{
		// CHECK: relative weights of each reward component
		// Terminate upon reaching goal
		if (pedpomdp_state.car.dist_travelled > Dvc_ModelParams::GOAL_TRAVELLED-1e-4
				|| pedpomdp_state.car.pos >= path->size_-1) {
			reward = Dvc_ModelParams::GOAL_REWARD;
			/*if(GPUDoPrint && pedpomdp_state.scenario_id==GPUPrintPID && blockIdx.x==0 && threadIdx.y==0){
				printf("Reach goal\n");
			}*/
			terminal= true;
		}
	}

	iscollision[threadIdx.x]=false;
	__syncthreads();

	if(!terminal)
	{
		//Dvc_inCollision(pedpomdp_state)
		const int car = pedpomdp_state.car.pos;
		const Dvc_COORD& car_pos = path->way_points_[car];
		const Dvc_COORD& forward_pos = path->way_points_[path->forward(car, 1.0)];

		//for(int i=0; i<pedpomdp_state.num; i++)
		if(threadIdx.y<pedpomdp_state.num){
			const Dvc_COORD& pedpos = pedpomdp_state.peds[threadIdx.y].pos;
			//Dvc_inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)
			bool collide_ped=false;
			float HNx = forward_pos.x - car_pos.x, // car direction
						 HNy = forward_pos.y - car_pos.y;
			float HMx = pedpos.x - car_pos.x,
						 HMy = pedpos.y - car_pos.y;

			/*float car_width = 0.82,
						 car_length = 0.732;
			float safe_margin = 0.4,
						 side_margin = car_width / 2.0 + safe_margin,
						 front_margin = car_length + safe_margin,
						 back_margin = car_length + safe_margin;*/

/// for golfcart
/*			double car_width = 0.87,
			car_length = 1.544;

			double safe_margin = 0.92, side_safe_margin = 0.4, back_safe_margin = 0.33,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = car_length/2.0 + safe_margin,
				 back_margin = car_length/2.0 + back_safe_margin;
*/


/// for audi r8
	double car_width = 2.0,
				 car_length = 4.4;

				 double safe_margin = 0.8, side_safe_margin = 0.35, back_safe_margin = 0.2,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = 3.6 + safe_margin,
				 back_margin = 0.8 + back_safe_margin;

/*			double safe_margin = 0.95, side_safe_margin = 0.4, back_safe_margin = 0.3,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = car_length/2.0 + safe_margin,
				 back_margin = car_length/2.0 + back_safe_margin;*/

			//return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);
			float HLx = - HNy, // direction after 90 degree anticlockwise rotation
						 HLy = HNx;

			float HM_HN = HMx * HNx + HMy * HNy, // HM . HN
						 HN_HN = HNx * HNx + HNy * HNy; // HN . HN
			if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
				collide_ped = false;
			else if (HM_HN <= 0 && HM_HN * HM_HN > HN_HN * back_margin * back_margin)
				collide_ped = false;
			else
			{
			    float HM_HL = HMx * HLx + HMy * HLy, // HM . HL
						 HL_HL = HLx * HLx + HLy * HLy; // HL . HL
			    collide_ped= HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
			}
			atomicOr(iscollision+threadIdx.x, collide_ped);
		}
	}
	__syncthreads();
	if(threadIdx.y==0 && !terminal)
	{
		if(pedpomdp_state.car.vel > 0.001 && iscollision[threadIdx.x] ) { /// collision occurs only when car is moving
			//reward = CrashPenalty(pedpomdp_state); //, closest_ped, closest_dist);
		    reward= Dvc_ModelParams::CRASH_PENALTY *
		    		(pedpomdp_state.car.vel * pedpomdp_state.car.vel +
		    				Dvc_ModelParams::REWARD_BASE_CRASH_VEL);

		    if(action == ACT_DEC) reward += 0.1;
		    /*if(threadIdx.x+blockIdx.y+blockIdx.x==0)
		    printf("action,reward,iscollision=%d %f %d\n"
		    					,action,reward,1);*/

			/*if(GPUDoPrint && pedpomdp_state.scenario_id==GPUPrintPID && blockIdx.x==0 && threadIdx.y==0){
				printf("Crash\n");
			}*/
			terminal= true;
		}

		if(!terminal)
		{
			// Smoothness control
			//reward += ActionPenalty(action);
			reward += (action == ACT_DEC || action == ACT_ACC) ? -0.1 : 0.0;
			/*if(pedpomdp_state.scenario_id==52 && blockIdx.x+threadIdx.y==0){
				printf("+ActionPenalty (act %d)=%f\n",action, reward);
			}*/
			// Speed control: Encourage higher speed
			//reward += MovementPenalty(pedpomdp_state);
			reward += Dvc_ModelParams::REWARD_FACTOR_VEL *
					(pedpomdp_state.car.vel - Dvc_ModelParams::VEL_MAX) / Dvc_ModelParams::VEL_MAX;
			/*if(reward<-100)printf("action,reward,car_vel=%d %f %d\n"
					    					,action,reward,pedpomdp_state.car.vel);*/
			/*if(pedpomdp_state.scenario_id==52 && blockIdx.x+threadIdx.y==0){
				printf("+MovementPenalty=%f\n",reward);
			}*/
			float acc = (action == ACT_ACC) ? Dvc_ModelParams::AccSpeed :
				((action == ACT_CUR) ?  0 : (-Dvc_ModelParams::AccSpeed));

			//if(action==0 && pedpomdp_state.scenario_id==0) printf("%f\n",rand_num);

			//rand_num=RandGeneration(&Temp, rand_num);
			//RobStep(pedpomdp_state.car, freq);
			float dist = pedpomdp_state.car.vel / freq;
			int nxt = path->forward(pedpomdp_state.car.pos, dist);
			pedpomdp_state.car.pos = nxt;
			pedpomdp_state.car.dist_travelled += dist;

			//if(action==0 && pedpomdp_state.scenario_id==0) printf("1-%f\n",rand_num);

			//RobVelStep(pedpomdp_state.car, acc, rand_num);
			const float N = Dvc_ModelParams::NOISE_ROBVEL;
			if (N>0) {
				/*if(pedpomdp_state.scenario_id==52 && gridDim.x==1)
					printf("start rand=%f\n", rand_num);*/
				if(FIX_SCENARIO!=1)
					rand_num=RandGeneration(&Temp, rand_num);
					/*if(pedpomdp_state.scenario_id==52 && blockIdx.x==0 && threadIdx.y==0)
						printf("rand=%f\n", rand_num);*/
				float prob = rand_num/*.NextDouble()*/;
				if (prob > N) {
					pedpomdp_state.car.vel += acc / freq;
				}
			} else {
				pedpomdp_state.car.vel += acc / freq;
			}
			pedpomdp_state.car.vel = max(min(pedpomdp_state.car.vel, Dvc_ModelParams::VEL_MAX), 0.0);
		}
	}
	__syncthreads();

	//if(threadIdx.y==0&&action==0 && pedpomdp_state.scenario_id==0) printf("2-%f\n",rand_num);

	if(!terminal)
	{
		// State transition
		if(threadIdx.y<pedpomdp_state.num)
		{
			int i=0;
			while(i</*=*/threadIdx.y)
			{
				if(FIX_SCENARIO!=1)
					rand_num=RandGeneration(&Temp, rand_num);
				i++;
			}
			if(threadIdx.y!=0 && FIX_SCENARIO!=1)
				rand_num=RandGeneration(&Temp, rand_num);

			//PedStep(pedpomdp_state.peds[threadIdx.y], rand_num, Temp);
			const Dvc_COORD& goal = goals[pedpomdp_state.peds[threadIdx.y].goal];
			if (abs(goal.x+1)<1e-5 && abs(goal.y+1)<1e-5) {  //stop intention
				;
			}
			else
			{
				Dvc_Vector goal_vec(goal.x - pedpomdp_state.peds[threadIdx.y].pos.x, goal.y - pedpomdp_state.peds[threadIdx.y].pos.y);
				float a = goal_vec.GetAngle();
				float noise = sqrt(-2 * log(rand_num));
				/*if(pedpomdp_state.scenario_id==52 && gridDim.x==1 && threadIdx.y==8)
					printf("ped %d rand=%f\n",threadIdx.y, rand_num);*/
				if(FIX_SCENARIO!=1)
					rand_num=RandGeneration(&Temp, rand_num);
				noise *= cos(2 * M_PI * rand_num)* Dvc_ModelParams::NOISE_GOAL_ANGLE;
				a += noise;

				//TODO noisy speed
				Dvc_Vector move(a, pedpomdp_state.peds[threadIdx.y].vel/freq, 0);
				pedpomdp_state.peds[threadIdx.y].pos.x += move.dw;
				pedpomdp_state.peds[threadIdx.y].pos.y += move.dh;
			}
		}
	}
	__syncthreads();


	// Observation
	//obs = Observe(pedpomdp_state);//original CPU code
	if(threadIdx.y==0 && obs!=NULL)//for each particle in the block
	{
		//if(action==0 && pedpomdp_state.scenario_id==0) printf("3-%f\n",rand_num);

		if(!terminal)
		{
			int i=0;
			obs[i++]=2+2*pedpomdp_state.num;
			//printf("obs[i]=%d\n",obs[i]);
			obs[i++] = int(pedpomdp_state.car.pos);
			obs[i++] = int((pedpomdp_state.car.vel+1e-5) / Dvc_ModelParams::vel_rln);
			for(int j = 0; j < pedpomdp_state.num; j ++) {
				obs[i++] = int(pedpomdp_state.peds[j].pos.x / Dvc_ModelParams::pos_rln);
				obs[i++] = int(pedpomdp_state.peds[j].pos.y / Dvc_ModelParams::pos_rln);
			}
		}
		else
		{
			int i=0;
			obs[i++]=0;
			obs[i++] = 0;
			obs[i++] = 0;
			for(int j = 0; j < pedpomdp_state.num; j ++) {
				obs[i++] = 0;
				obs[i++] = 0;
			}
		}
	}
	/*if(!terminal && GPUDoPrint && pedpomdp_state.scenario_id==GPUPrintPID && blockIdx.x==0 && threadIdx.y==0){
		//printf("address=%#0x \n", &pedpomdp_state);

		printf("rand=%f, action=%d \n", rand_num, action);
		printf("After step:\n");
		printf("Reward=%f\n",reward);
		int pos=pedpomdp_state.car.pos;
		printf("car pox= %d ",pos);
		printf("dist=%f\n",pedpomdp_state.car.dist_travelled);
		printf("car vel= %f\n",pedpomdp_state.car.vel);
		for(int i=0;i<pedpomdp_state.num;i++)
		{
			printf("ped %d pox_x= %f pos_y=%f\n",i,
					pedpomdp_state.peds[i].pos.x,pedpomdp_state.peds[i].pos.y);
		}
	}*/
	return terminal/*temp*/;//Debug,test time
}

DEVICE int Dvc_PedPomdp::NumActions() const {
	return /*5*/3;
}

DEVICE void Dvc_PedPomdp::Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des) {
	/*Pass member values, assign member pointers to existing state pointer*/
	const Dvc_PomdpState* src_i= static_cast<const Dvc_PomdpState*>(src)+pos;
	if(!offset_des) pos=0;
	Dvc_PomdpState* des_i= static_cast<const Dvc_PomdpState*>(des)+pos;
	//*des_i = *src_i;
	des_i->weight=src_i->weight;
	des_i->scenario_id=src_i->scenario_id;
	des_i->num=src_i->num;
	des_i->car.dist_travelled=src_i->car.dist_travelled;
	des_i->car.pos=src_i->car.pos;
	des_i->car.vel=src_i->car.vel;
	//des_i->peds=(Dvc_PedStruct*)(&(des_i->num)+1);
	for(int i=0;i</*Dvc_ModelParams::N_PED_IN*/des_i->num;i++)
	{
		des_i->peds[i].vel=src_i->peds[i].vel;
		des_i->peds[i].pos.x=src_i->peds[i].pos.x;
		des_i->peds[i].pos.y=src_i->peds[i].pos.y;
		des_i->peds[i].goal=src_i->peds[i].goal;
		des_i->peds[i].id=src_i->peds[i].id;
	}

	/*if(GPUDoPrint && des_i->scenario_id==GPUPrintPID && blockIdx.x==0 && threadIdx.y==0){
		Dvc_PomdpState* des=des_i;
		int sid=des_i->scenario_id;
		printf("(GPU) copy to global memory %d",des);
		printf(" for scenario=%d \n", sid);
		int pos=des_i->car.pos;
		printf("car pox= %d ",pos);
		printf("dist=%f\n",des_i->car.dist_travelled);
		printf("car vel= %f\n",des_i->car.vel);

		for(int i=0;i<des_i->num;i++)
		{
			printf("ped %d pox_x= %f pos_y=%f\n",i,
					des_i->peds[i].pos.x,des_i->peds[i].pos.y);
		}
	}*/
}
DEVICE void Dvc_PedPomdp::Dvc_Copy_ToShared(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des) {
	/*Pass member values, assign member pointers to existing state pointer*/
	const Dvc_PomdpState* src_i= static_cast<const Dvc_PomdpState*>(src)+pos;
	if(!offset_des) pos=0;
	Dvc_PomdpState* des_i= static_cast<const Dvc_PomdpState*>(des)+pos;
	//*des_i = *src_i;
	des_i->weight=src_i->weight;
	des_i->scenario_id=src_i->scenario_id;
	des_i->num=src_i->num;
	des_i->car.dist_travelled=src_i->car.dist_travelled;
	des_i->car.pos=src_i->car.pos;
	des_i->car.vel=src_i->car.vel;
	//des_i->peds=(Dvc_PedStruct*)(&(des_i->num)+1);
	des_i->peds=(Dvc_PedStruct*)((void*)(des_i)+3*sizeof(Dvc_PedStruct));
	for(int i=0;i</*Dvc_ModelParams::N_PED_IN*/des_i->num;i++)
	{
		des_i->peds[i].vel=src_i->peds[i].vel;
		des_i->peds[i].pos.x=src_i->peds[i].pos.x;
		des_i->peds[i].pos.y=src_i->peds[i].pos.y;
		des_i->peds[i].goal=src_i->peds[i].goal;
		des_i->peds[i].id=src_i->peds[i].id;
	}
}
DEVICE Dvc_State* Dvc_PedPomdp::Dvc_Get(Dvc_State* particles, int pos) {
	Dvc_PomdpState* particle_i= static_cast<Dvc_PomdpState*>(particles)+pos;

	return particle_i;
}
DEVICE Dvc_ValuedAction Dvc_PedPomdp::Dvc_GetMinRewardAction() {
	return Dvc_ValuedAction(/*E_STAY*/0,
			Dvc_ModelParams::CRASH_PENALTY * (Dvc_ModelParams::VEL_MAX*Dvc_ModelParams::VEL_MAX + Dvc_ModelParams::REWARD_BASE_CRASH_VEL));
}
void PedPomdp::ReadBackToCPU(std::vector<State*>& particles ,const Dvc_State* dvc_particles,
			bool copycells) const
{
	for (int i=0;i<particles.size();i++)
	{
		const Dvc_PomdpState* src=static_cast<const Dvc_PomdpState*>(dvc_particles)+i;
		PomdpState* des=static_cast<PomdpState*>(particles[i]);
		Dvc_PomdpState::ReadBackToCPU(src,des);
	}
	Dvc_PomdpState::ReadBackToCPU2(
			static_cast<const Dvc_PomdpState*>(dvc_particles),
			particles,true);


	/*if(CPUDoPrint && particles[0]->scenario_id==CPUPrintPID){
		for (int i=0;i<particles.size();i++)
		{
			PomdpState* pedpomdp_state=static_cast<PomdpState*>(particles[i]);
			printf("Read back particle %d, scenario %d, list size %d\n,", dvc_particles,particles[i]->scenario_id, particles.size());
			printf("car pox= %d ",pedpomdp_state->car.pos);
			printf("dist=%f\n",pedpomdp_state->car.dist_travelled);
			printf("car vel= %f\n",pedpomdp_state->car.vel);
			for(int i=0;i<pedpomdp_state->num;i++)
			{
				printf("ped %d pox_x= %f pos_y=%f\n",i,
						pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
			}
		}
	}*/

}
