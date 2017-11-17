#include "GPU_base_unc_navigation.h"

#include <base/base_unc_navigation.h>
#include <cuda_runtime_api.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <despot/util/coord.h>
#include <driver_types.h>
#include <stddef.h>
#include "despot/GPUutil/GPUmemorypool.h"
#include <despot/GPUcore/thread_globals.h>

#define THREADDIM 128
using namespace std;

namespace despot {
//static int NumObstacles;
//static bool NewRound=true;
//static DvcCoord* obstacle_pos;
static Dvc_UncNavigationState* Dvc_particles=NULL;
//static int* Dvc_particleIDs=NULL;
static bool **Dvc_tempCells=NULL;
static bool **Hst_tempCells=NULL;

static Dvc_UncNavigationState* tmp=NULL;

static GPU_MemoryPool<Dvc_UncNavigationState>* gpu_memory_pool_=NULL;
//static Array_MemoryPool<bool>* cell_memory_pool_=NULL;


static float** tmp_result=NULL;
static int** tempHostID=NULL;
static float** temp_weight=NULL;

//const DvcCoord Dvc_NavCompass::DIRECTIONS[] = {  DvcCoord(0, 1), DvcCoord(1, 0),DvcCoord(0, -1),
//	DvcCoord(-1, 0), DvcCoord(1, 1), DvcCoord(1, -1), DvcCoord(-1, -1), DvcCoord(-1, 1) };
//const string Dvc_NavCompass::CompassString[] = { "North", "East","South", "West",
//	"NE", "SE", "SW", "NW" };
/* ==============================================================================
 * Dvc_UncNavigationState class
 * ==============================================================================*/

DEVICE Dvc_UncNavigationState::Dvc_UncNavigationState()
{
	sizeX_=0;
	sizeY_=0;
	rob.x=-1;rob.y=-1;
	cells=NULL;
	b_Extern_cells=false;
}
DEVICE Dvc_UncNavigationState::Dvc_UncNavigationState(int _state_id)
{
	sizeX_=0;sizeY_=0;
	rob.x=-1;rob.y=-1;
	goal.x=-1;goal.y=-1;
	cells=NULL;
	state_id=_state_id;
	b_Extern_cells=false;
}
DEVICE Dvc_UncNavigationState::Dvc_UncNavigationState(int sizeX, int sizeY)
{
	cells=NULL;
	InitCells(sizeX,sizeY);
	rob.x=-1;rob.y=-1;
	goal.x=-1;goal.y=-1;
	b_Extern_cells=false;
}

DEVICE Dvc_UncNavigationState::Dvc_UncNavigationState(const Dvc_UncNavigationState& src)
{
	cells=NULL;
	Assign_NoAlloc(src);
	b_Extern_cells=true;
	//rob.x=src.rob.x; rob.y=src.rob.y;
	//goal=src.goal;
	//InitCells(src.sizeX_,src.sizeY_);
	//sizeX_=src.sizeX_;sizeY_=src.sizeY_;
	//cells=new bool[sizeX_*sizeY_];

	//memcpy((void*)cells,(const void*)src.cells, sizeX_*sizeY_*sizeof(bool));
}
HOST void Dvc_UncNavigationState::InitCellsManaged(int sizeX, int sizeY)
{
	sizeX_=sizeX; sizeY_=sizeY;
	if(cells==NULL)
	{
		HANDLE_ERROR(cudaMalloc((void**)&cells, sizeX_*sizeY_*sizeof(bool)));
		//cells=new bool[sizeX_*sizeY_];
		//memset((void*)cells,0, sizeX*sizeY*sizeof(bool));
		b_Extern_cells=false;
	}
}

/*HOST void Dvc_UncNavigationState::Assign(const UncNavigationState* src)
{
	//Assume this is a managed variable

	rob.x=src->rob.x; rob.y=src->rob.y;
	goal.x=src->goal.x;goal.y=src->goal.y;
	state_id=src->state_id;
	scenario_id=src->scenario_id;
	weight=src->weight;

	InitCellsManaged(src->sizeX_,src->sizeY_);

	HANDLE_ERROR(cudaMemcpy((void*)cells, (const void*)src->cells, sizeX_*sizeY_*sizeof(bool), cudaMemcpyHostToDevice));

}*/
/*__global__ void CopyCells(int scenarioID,  bool* src)
{
	int x=threadIdx.x;
	int y=threadIdx.y;
	if(x==0 && y==0)
	{
		Dvc->cells=(bool*)malloc(Dvc->sizeX_*Dvc->sizeY_*sizeof(bool));
	}
	__syncthreads();
	Dvc_UncNavigationState* Dvc=static_cast<Dvc_UncNavigationState*>(Dvc_particles_copy)+scenarioID;

	if(x<Dvc->sizeX_ && y<Dvc->sizeY_)
		Dvc->GridOpen(x,y)=src[y*Dvc->sizeX_+x];

}*/
__global__ void CopyCells(Dvc_UncNavigationState* Dvc,  bool* src)
{
	int scenarioID=blockIdx.x;
	int x=threadIdx.x;
	int y=threadIdx.y;
	Dvc_UncNavigationState* Dvc_i=Dvc+scenarioID;
	bool* Src_i=src+scenarioID*Dvc_i->sizeX_*Dvc_i->sizeY_;
	/*if(x==0 && y==0)
	{
		Dvc->cells=(bool*)malloc(Dvc->sizeX_*Dvc->sizeY_*sizeof(bool));
	}
	__syncthreads();*/

	if(x<Dvc_i->sizeX_ && y<Dvc_i->sizeY_)
		Dvc_i->GridOpen(x,y)=Src_i[y*Dvc_i->sizeX_+x];

}
__global__ void CopyCells_back(const Dvc_UncNavigationState* Dvc,  bool* des)
{
	int scenarioID=blockIdx.x;
	int x=threadIdx.x;
	int y=threadIdx.y;
	const Dvc_UncNavigationState* Dvc_i=Dvc+scenarioID;
	bool* Des_i=des+scenarioID*Dvc_i->sizeX_*Dvc_i->sizeY_;
	/*if(x==0 && y==0)
	{
		Dvc->cells=(bool*)malloc(Dvc->sizeX_*Dvc->sizeY_*sizeof(bool));
	}
	__syncthreads();*/

	if(x<Dvc_i->sizeX_ && y<Dvc_i->sizeY_)
		Des_i[y*Dvc_i->sizeX_+x]=Dvc_i->Grid(x,y);

}
/*__global__ void CopyMembers(int scenarioID, Dvc_UncNavigationState* src)
{
	int x=threadIdx.x;
	int y=threadIdx.y;
	if(x==0 && y==0)Hst->num_food
	{
		Dvc_UncNavigationState* Dvc=static_cast<Dvc_UncNavigationState*>(Dvc_particles_copy)+scenarioID;
		Dvc->rob.x=src->rob.x; Dvc->rob.y=src->rob.y;
		Dvc->goal.x=src->goal.x;Dvc->goal.y=src->goal.y;
		Dvc->allocated_=src->allocated_;
		Dvc->state_id=src->state_id;
		Dvc->scenario_id=src->scenario_id;
		Dvc->weight=src->weight;
		Dvc->sizeX_=src->sizeX_; Dvc->sizeY_=src->sizeY_;
	}

	__syncthreads();
}*/
__global__ void CopyMembers(Dvc_UncNavigationState* Dvc, Dvc_UncNavigationState* src)
{
	int x=threadIdx.x;
	int y=threadIdx.y;
	if(x==0 && y==0)
	{
		Dvc->rob.x=src->rob.x; Dvc->rob.y=src->rob.y;
		Dvc->goal.x=src->goal.x;Dvc->goal.y=src->goal.y;
		Dvc->allocated_=src->allocated_;
		Dvc->state_id=src->state_id;
		Dvc->scenario_id=src->scenario_id;
		Dvc->weight=src->weight;
		Dvc->sizeX_=src->sizeX_; Dvc->sizeY_=src->sizeY_;
	}

	__syncthreads();
}

HOST void Dvc_UncNavigationState::CopyToGPU(Dvc_UncNavigationState* Dvc, int scenarioID, const UncNavigationState* Hst, bool copy_cells)
{
	//HANDLE_ERROR(cudaMemcpy((void*)Dvc, (const void*)&(Hst->allocated_), sizeof(Dvc_UncNavigationState), cudaMemcpyHostToDevice));

	Dvc[scenarioID].rob.x=Hst->rob.x; Dvc[scenarioID].rob.y=Hst->rob.y;
	Dvc[scenarioID].weight=Hst->weight;
	if(copy_cells)
	{
		Dvc[scenarioID].goal.x=Hst->goal.x;Dvc[scenarioID].goal.y=Hst->goal.y;
		Dvc[scenarioID].allocated_=Hst->allocated_;
		Dvc[scenarioID].state_id=Hst->state_id;
		Dvc[scenarioID].scenario_id=Hst->scenario_id;
		Dvc[scenarioID].sizeX_=Hst->sizeX_; Dvc[scenarioID].sizeY_=Hst->sizeY_;
		int Data_block_size=Hst->sizeX_*Hst->sizeY_;

		if(use_multi_thread_ && cuda_streams)
		{
			memcpy((void*)(Hst_tempCells[GetCurrentStream()]+Data_block_size*scenarioID),
					(const void*)Hst->cells,
					Data_block_size*sizeof(bool));
		}
		else
		{
			memcpy((void*)(Hst_tempCells[0]+Data_block_size*scenarioID),
					(const void*)Hst->cells,
					Data_block_size*sizeof(bool));
		}
	}
}
HOST void Dvc_UncNavigationState::CopyToGPU2(Dvc_UncNavigationState* Dvc, int NumParticles, bool copy_cells)
{
	/*tmp->rob.x=Hst->rob.x; tmp->rob.y=Hst->rob.y;
	tmp->goal.x=Hst->goal.x;tmp->goal.y=Hst->goal.y;
	tmp->allocated_=Hst->allocated_;
	tmp->state_id=Hst->state_id;
	tmp->scenario_id=Hst->scenario_id;
	tmp->weight=Hst->weight;
	tmp->sizeX_=Hst->sizeX_; tmp->sizeY_=Hst->sizeY_;*/

	//UncNavigationState tmp;
	//HANDLE_ERROR(cudaMemcpy((void*)&(tmp.allocated_), (const void*)Dvc, sizeof(Dvc_UncNavigationState), cudaMemcpyDeviceToHost));
	//dim3 grid(1,1);dim3 threads(1,1);

	if(Dvc!=NULL)
	{
		//CopyMembers<<<grid, threads>>>(Dvc+scenarioID,tmp);
		//cudaDeviceSynchronize();

		if(copy_cells)
		{
			int Data_size=NumParticles*Dvc->sizeX_*Dvc->sizeY_;
			dim3 grid1(NumParticles,1);dim3 threads1(Dvc->sizeX_,Dvc->sizeY_);
			if(use_multi_thread_ && cuda_streams)
			{
				HANDLE_ERROR(cudaMemcpyAsync((void*)Dvc_tempCells[GetCurrentStream()],
						(const void*)Hst_tempCells[GetCurrentStream()],
						Data_size*sizeof(bool),
						cudaMemcpyHostToDevice,((cudaStream_t*)cuda_streams)[GetCurrentStream()]));

				CopyCells<<<grid1, threads1, 0, ((cudaStream_t*)cuda_streams)[GetCurrentStream()]>>>
						(Dvc,Dvc_tempCells[GetCurrentStream()]);
				//AdvanceStreamCounter(1);
			}
			else
			{
				HANDLE_ERROR(cudaMemcpy((void*)Dvc_tempCells[0],
						(const void*)Hst_tempCells[0],
						Data_size*sizeof(bool),
						cudaMemcpyHostToDevice));
				CopyCells<<<grid1, threads1>>>(Dvc,Dvc_tempCells[0]);
				//cudaDeviceSynchronize();
			}


		}
	}
	/*else
	{
		CopyMembers<<<grid, threads>>>(scenarioID,tmp);
		cudaDeviceSynchronize();

		if(copy_cells)
		{
			HANDLE_ERROR(cudaMemcpy((void*)Dvc_tempCells, (const void*)Hst->cells, Hst->sizeX_*Hst->sizeY_*sizeof(bool), cudaMemcpyHostToDevice));
			dim3 grid1(1,1);dim3 threads1(Hst->sizeX_,Hst->sizeY_);
			CopyCells<<<grid1, threads1>>>(scenarioID,Dvc_tempCells);
			cudaDeviceSynchronize();
		}
	}*/
}

HOST void Dvc_UncNavigationState::ReadBackToCPU(const Dvc_UncNavigationState* Dvc, UncNavigationState* Hst, bool copy_cells)
{
	HANDLE_ERROR(cudaMemcpy((void*)tmp, (const void*)Dvc, sizeof(Dvc_UncNavigationState), cudaMemcpyDeviceToHost));

	Hst->rob.x=tmp->rob.x; Hst->rob.y=tmp->rob.y;
	Hst->weight=tmp->weight;
	if(copy_cells)
	{
		Hst->goal.x=tmp->goal.x;Hst->goal.y=tmp->goal.y;
		Hst->allocated_=tmp->allocated_;
		Hst->state_id=tmp->state_id;
		Hst->scenario_id=tmp->scenario_id;
		Hst->sizeX_=tmp->sizeX_; Hst->sizeY_=tmp->sizeY_;
	}
}
HOST void Dvc_UncNavigationState::ReadBackToCPU2(const Dvc_UncNavigationState* Dvc,std::vector<State*> Hst, bool copy_cells)
{
	int NumParticles=Hst.size();
	if(NumParticles!=0)
	{
		if(copy_cells)
		{
			UncNavigationState* nav_state=static_cast<UncNavigationState*>(Hst[0]);

			int Data_block_size=nav_state->sizeX_*nav_state->sizeY_;
			int Data_size=NumParticles*Data_block_size;
			dim3 grid1(NumParticles,1);dim3 threads1(nav_state->sizeX_,nav_state->sizeY_);
			if(use_multi_thread_ && cuda_streams)
			{

				CopyCells_back<<<grid1, threads1, 0, ((cudaStream_t*)cuda_streams)[GetCurrentStream()]>>>
						(Dvc,Dvc_tempCells[GetCurrentStream()]);

				HANDLE_ERROR(cudaMemcpyAsync((void*)Hst_tempCells[GetCurrentStream()],
						(const void*)Dvc_tempCells[GetCurrentStream()],
						Data_size*sizeof(bool),
						cudaMemcpyDeviceToHost,((cudaStream_t*)cuda_streams)[GetCurrentStream()]));

				//AdvanceStreamCounter(1);
			}
			else
			{

				CopyCells_back<<<grid1, threads1>>>(Dvc,Dvc_tempCells[0]);
				//cudaDeviceSynchronize();
				HANDLE_ERROR(cudaMemcpy((void*)Hst_tempCells[0],
						(const void*)Dvc_tempCells[0],
						Data_size*sizeof(bool),
						cudaMemcpyDeviceToHost));
			}
			for(int i=0;i<NumParticles;i++)
			{
				nav_state=static_cast<UncNavigationState*>(Hst[i]);
				if(use_multi_thread_ && cuda_streams)
				{
					memcpy((void*)nav_state->cells,
							(const void*)(Hst_tempCells[GetCurrentStream()]+Data_block_size*i),
							Data_block_size*sizeof(bool));
				}
				else
				{
					memcpy((void*)nav_state->cells,
							(const void*)(Hst_tempCells[0]+Data_block_size*i),
							Data_block_size*sizeof(bool));
				}
			}
		}
	}
	/*else
	{
		CopyMembers<<<grid, threads>>>(scenarioID,tmp);
		cudaDeviceSynchronize();

		if(copy_cells)
		{
			HANDLE_ERROR(cudaMemcpy((void*)Dvc_tempCells, (const void*)Hst->cells, Hst->sizeX_*Hst->sizeY_*sizeof(bool), cudaMemcpyHostToDevice));
			dim3 grid1(1,1);dim3 threads1(Hst->sizeX_,Hst->sizeY_);
			CopyCells<<<grid1, threads1>>>(scenarioID,Dvc_tempCells);
			cudaDeviceSynchronize();
		}
	}*/
}

__global__ void AllocCells(Dvc_UncNavigationState* Dvc,int Cell_x,int Cell_y, int num_particles)
{
	int pos=blockIdx.x*blockDim.x+threadIdx.x;
	//int y=threadIdx.y;
	if(pos < num_particles)
	{
		Dvc_UncNavigationState* Dvc_i=Dvc+pos;
		Dvc_i->cells=(bool*)malloc(Cell_x*Cell_y*sizeof(bool));
		Dvc_i->b_Extern_cells=false;
	}
}
__global__ void CopyParticle(Dvc_UncNavigationState* des,Dvc_UncNavigationState* src,int num_particles)
{
	//int pos=threadIdx.x;
	int pos=blockIdx.x*blockDim.x+threadIdx.x;

	//int y=threadIdx.y;
	if(pos < num_particles)
	{
		Dvc_UncNavigationState* src_i=src+pos;
		Dvc_UncNavigationState* des_i=des+pos;
		des_i->rob.x=src_i->rob.x; des_i->rob.y=src_i->rob.y;
		des_i->weight=src_i->weight;
		des_i->goal.x=src_i->goal.x;des_i->goal.y=src_i->goal.y;
		des_i->allocated_=src_i->allocated_;
		des_i->state_id=src_i->state_id;
		des_i->scenario_id=src_i->scenario_id;
		des_i->sizeX_=src_i->sizeX_; des_i->sizeY_=src_i->sizeY_;
		des_i->cells=src_i->cells;
		des_i->b_Extern_cells=true;
	}
}

__global__ void CopyParticles(Dvc_UncNavigationState* des,Dvc_UncNavigationState* src,
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
		Dvc_UncNavigationState* src_i=src+scenarioID;//src is a full length array for all particles
		Dvc_UncNavigationState* des_i=des+pos;//des is short, only for the new partition
		if(src_i->rob.x>4 ||src_i->rob.y>4 ||src_i->goal.x>4 ||src_i->goal.y>4 )
			error=true;
		des_i->rob.x=src_i->rob.x; des_i->rob.y=src_i->rob.y;
		des_i->weight=src_i->weight;
		des_i->goal.x=src_i->goal.x;des_i->goal.y=src_i->goal.y;
		des_i->allocated_=src_i->allocated_;
		des_i->state_id=src_i->state_id;
		des_i->scenario_id=src_i->scenario_id;
		des_i->sizeX_=src_i->sizeX_; des_i->sizeY_=src_i->sizeY_;
		des_i->cells=src_i->cells;
		des_i->b_Extern_cells=true;

		atomicAdd(weight, des_i->weight);
	}
}
/*_global__ void AllocParticleCopy(int Cell_x,int Cell_y, int num_particles)
{
	int pos=threadIdx.x;
	//int y=threadIdx.y;
	if(pos==0)
		Dvc_particles_copy=(Dvc_UncNavigationState*)malloc(num_particles*sizeof(Dvc_UncNavigationState));

	__syncthreads();
	if(pos < num_particles)
	{
		Dvc_UncNavigationState* Dvc_i=static_cast<Dvc_UncNavigationState*>(Dvc_particles_copy)+pos;
		Dvc_i->cells=(bool*)malloc(Cell_x*Cell_y*sizeof(bool));
		Dvc_i->b_Extern_cells=false;
	}
}*/
void BaseUncNavigation::CreateMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;

	if(gpu_memory_pool_==NULL)
		gpu_memory_pool_=new GPU_MemoryPool<Dvc_UncNavigationState>;

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
void BaseUncNavigation::DestroyMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;
	switch(mode)
	{
		case 0:
			if(gpu_memory_pool_){delete gpu_memory_pool_;gpu_memory_pool_=NULL;}
			break;
		case 1:
			if(gpu_memory_pool_ ){ gpu_memory_pool_->ResetChuncks();};
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

Dvc_State* BaseUncNavigation::AllocGPUParticles(int numParticles, int alloc_mode) const
{//numParticles==num_Scenarios
	clock_t start=clock();
	dim3 grid((numParticles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);
	int num_threads=1;

	if(use_multi_thread_)
	{
		num_threads=NUM_THREADS;
	}
	switch(alloc_mode)
	{
	case 0:
		HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles, numParticles*sizeof(Dvc_UncNavigationState)));
		AllocCells<<<grid, threads>>>(Dvc_particles,size_, size_,numParticles);
		HANDLE_ERROR(cudaDeviceSynchronize());

		Dvc_tempCells=new bool*[num_threads];
		Hst_tempCells=new bool*[num_threads];

		for(int i=0;i<num_threads;i++)
		{
			HANDLE_ERROR(cudaMalloc((void**)&Dvc_tempCells[i],numParticles*size_*size_*sizeof(bool) ));
			HANDLE_ERROR(cudaHostAlloc((void**)&Hst_tempCells[i],numParticles*size_*size_*sizeof(bool),0 ));
		}
		HANDLE_ERROR(cudaHostAlloc((void**)&tmp,1*sizeof(Dvc_UncNavigationState),0));

		 //HANDLE_ERROR(cudaMalloc(&tmp_result, sizeof(float)));
		 Dvc_stepped_particles_all_a=new Dvc_State*[num_threads];
		//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particleIDs,numParticles*sizeof(int) ));
		for(int i=0;i<num_threads;i++)
			HANDLE_ERROR(cudaMalloc((void**)&Dvc_stepped_particles_all_a[i],
					NumActions()*numParticles*sizeof(Dvc_UncNavigationState)));

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

		return Dvc_particles;
	case 1:
		CreateMemoryPool(1);
		return Dvc_particles;
	case 2:
		//cout<<__FUNCTION__ <<endl;
		//cout<<"numParticles"<<numParticles<<endl;
		//cout<<"gpu_memory_pool_"<<gpu_memory_pool_<<endl;

		Dvc_UncNavigationState* tmp=gpu_memory_pool_->Allocate(numParticles);
		//Dvc_UncNavigationState* tmp;
		//HANDLE_ERROR(cudaMalloc((void**)&tmp, numParticles*sizeof(Dvc_UncNavigationState)));

		return tmp;
	};
	/*Dvc_particles_copy is an extern variable declared in GPUpolicy.h*/
	//HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles_copy, numParticles*sizeof(Dvc_UncNavigationState)));
	/*AllocParticleCopy<<<grid, threads>>>(size_, size_,numParticles);
	HANDLE_ERROR(cudaDeviceSynchronize());*/

	cout<<"GPU particles alloc time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl;
}


void BaseUncNavigation::CopyGPUParticles(Dvc_State* des,Dvc_State* src,int src_offset,
		int* IDs,int num_particles,bool interleave,
		Dvc_RandomStreams* streams, int stream_pos,
		void* CUDAstream, int shift) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);
	if(num_particles<THREADDIM)
	{
		grid.x=1;grid.y=1;threads.x=num_particles;
	}
	//float* tmp_result; HANDLE_ERROR(cudaMalloc(&tmp_result, sizeof(float)));

	int ThreadID=0;
	if(use_multi_thread_)
		ThreadID=MapThread(this_thread::get_id());
	//float particle_weight=0;
	if(CUDAstream)
	{
		CopyParticles<<<grid, threads,0, *(cudaStream_t*)CUDAstream>>>(static_cast<Dvc_UncNavigationState*>(des),
				static_cast<Dvc_UncNavigationState*>(src)+src_offset,tmp_result[(ThreadID+shift)%NUM_THREADS],
				IDs,num_particles, streams,stream_pos);
		//HANDLE_ERROR(cudaMemcpyAsync(temp_weight[(ThreadID+shift)%NUM_THREADS], tmp_result[(ThreadID+shift)%NUM_THREADS], sizeof(float),cudaMemcpyDeviceToHost, *(cudaStream_t*)CUDAstream));
		//if(num_particles>0)
		//	HANDLE_ERROR(cudaMemcpyAsync(temp_weight[ThreadID], tmp_result[ThreadID], sizeof(float),cudaMemcpyDeviceToHost, *(cudaStream_t*)CUDAstream));
		//else
		//	HANDLE_ERROR(cudaMemcpy(temp_weight[ThreadID], tmp_result[ThreadID], sizeof(float),cudaMemcpyDeviceToHost));
		if(!interleave)
			;//HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)CUDAstream));
	}
	else
	{
		CopyParticles<<<grid, threads,0, 0>>>(static_cast<Dvc_UncNavigationState*>(des),
				static_cast<Dvc_UncNavigationState*>(src)+src_offset,tmp_result[ThreadID],
				IDs,num_particles, streams,stream_pos);
		if(!interleave)
			HANDLE_ERROR(cudaDeviceSynchronize());
	}
	//HANDLE_ERROR(cudaMemcpy(&particle_weight, tmp_result, sizeof(float),cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaFree(tmp_result));

	//return particle_weight;
		//HANDLE_ERROR(cudaDeviceSynchronize());
}

void BaseUncNavigation::CopyGPUWeight1(void* cudaStream, int shift) const
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
float BaseUncNavigation::CopyGPUWeight2(void* cudaStream, int shift) const
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
		//HANDLE_ERROR(cudaMemcpyAsync(temp_weight[ThreadID], tmp_result[ThreadID], sizeof(float),cudaMemcpyDeviceToHost, *(cudaStream_t*)cudaStream));
		HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)cudaStream));
		if(*temp_weight[(ThreadID+shift)%NUM_THREADS]>1)
			Global_print_value(this_thread::get_id(),*temp_weight[(ThreadID+shift)%NUM_THREADS],"Wrong weight from CopyGPUWeight");
		return *temp_weight[(ThreadID+shift)%NUM_THREADS];
	}
}

Dvc_State* BaseUncNavigation::GetGPUParticles() const
{
	return Dvc_particles;
}
Dvc_State* BaseUncNavigation::CopyToGPU(const std::vector<State*>& particles, bool copy_cells) const
{
	//Dvc_UncNavigationState* Dvc_particles;
	//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particles, particles.size()*sizeof(Dvc_UncNavigationState)));
	//dvc_particles should be managed device memory
	auto start = Time::now();

	for (int i=0;i<particles.size();i++)
	{
		const UncNavigationState* src=static_cast<const UncNavigationState*>(particles[i]);
		//Dvc_particles[i].Assign(src);
		Dvc_UncNavigationState::CopyToGPU(Dvc_particles,src->scenario_id,src);
		//Dvc_UncNavigationState::CopyToGPU(NULL,src->scenario_id,src, false);//copy to Dvc_particles_copy, do not copy cells, leave it a NULL pointer
	}
	Dvc_UncNavigationState::CopyToGPU2(Dvc_particles,particles.size());

	//for (int i=0;i<particles.size();i++)
	//{
		//const UncNavigationState* src=static_cast<const UncNavigationState*>(particles[i]);
		//Dvc_particles[i].Assign(src);
		//Dvc_UncNavigationState::CopyToGPU2(Dvc_particles,src->scenario_id,src);
		//Dvc_UncNavigationState::CopyToGPU(NULL,src->scenario_id,src, false);//copy to Dvc_particles_copy, do not copy cells, leave it a NULL pointer
	//}
	//cudaDeviceSynchronize();
	cout<<"GPU particles copy time:"<<chrono::duration_cast<sec>(Time::now() - start).count()<<endl;

	return Dvc_particles;
}
void BaseUncNavigation::ReadBackToCPU(std::vector<State*>& particles ,const Dvc_State* dvc_particles,
			bool copycells) const
{
	auto start = Time::now();

	for (int i=0;i<particles.size();i++)
	{
		const Dvc_UncNavigationState* src=static_cast<const Dvc_UncNavigationState*>(dvc_particles)+i;
		UncNavigationState* des=static_cast<UncNavigationState*>(particles[i]);
		Dvc_UncNavigationState::ReadBackToCPU(src,des);
	}
	Dvc_UncNavigationState::ReadBackToCPU2(
			static_cast<const Dvc_UncNavigationState*>(dvc_particles),
			particles,true);
}
void BaseUncNavigation::CopyToGPU(const std::vector<int>& particleIDs, int* Dvc_ptr, void *CUDAstream) const
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

__global__ void FreeCells(Dvc_UncNavigationState* states, int num_particles)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;

	if (i < num_particles) {
		states[i].deleteCells();
	}
}

/*__global__ void FreeParticleCopy(int num_particles)
{
	int i=threadIdx.x;
	if (i < num_particles && Dvc_particles_copy!=NULL) {
		static_cast<Dvc_UncNavigationState*>(Dvc_particles_copy)[i].deleteCells();
	}

	__syncthreads();

	if(i==0)
	{free(Dvc_particles_copy); Dvc_particles_copy=NULL;}
}*/

void BaseUncNavigation::DeleteGPUParticles( int num_particles) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);

	FreeCells<<<grid, threads>>>(static_cast<Dvc_UncNavigationState*>(Dvc_particles),num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaFree(static_cast<Dvc_UncNavigationState*>(Dvc_particles)));

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
		cudaFree(Dvc_tempCells[i]);
		cudaFreeHost(Hst_tempCells[i]);
	}
	delete [] Dvc_tempCells;
	delete [] Hst_tempCells;
	cudaFreeHost(tmp);

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

	/*FreeParticleCopy<<<grid, threads>>>(num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());*/
	//HANDLE_ERROR(cudaFree(static_cast<Dvc_UncNavigationState*>(Dvc_particles_copy)));
}

void BaseUncNavigation::DeleteGPUParticles(Dvc_State* particles, int num_particles) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);
	FreeCells<<<grid, threads>>>(static_cast<Dvc_UncNavigationState*>(particles),num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaFree(static_cast<Dvc_UncNavigationState*>(particles)));
}


DEVICE float Dvc_UncNavigationParticleUpperBound1::Value(
		const Dvc_State* particles, int scenarioID, Dvc_History& history) {
	const Dvc_UncNavigationState* nav_state =
			static_cast<const Dvc_UncNavigationState*>(particles) + scenarioID;
	int count_x = abs(nav_state->rob.x - nav_state->goal.x);
	int count_y = abs(nav_state->rob.y - nav_state->goal.y);
	float value1 = 0;

	int min_xy=min(count_x,count_y);
	int diff_xy=abs(count_x-count_y);
	for (int i=0;i<min_xy;i++)
	{
		value1+=Dvc_Globals::Dvc_Discount(Dvc_config, i)*(-0.1);
	}
	for (int j=0;j<diff_xy;j++)
	{
		value1+=Dvc_Globals::Dvc_Discount(Dvc_config, min_xy+j)*(-0.1);
	}

	/*for (int i = 0; i < count_x; i++) {
		value1 += Dvc_Globals::Dvc_Discount(Dvc_config, i) * (-0.1);
	}
	for (int j = 0; j < count_y; j++) {
		value1 += Dvc_Globals::Dvc_Discount(Dvc_config, count_x + j) * (-0.1);
	}
	for (int j = 0; j < count_y; j++) {
		value2 += Dvc_Globals::Dvc_Discount(Dvc_config, j) * (-0.1);
	}
	for (int i = 0; i < count_x; i++) {
		value2 += Dvc_Globals::Dvc_Discount(Dvc_config, count_y + i) * (-0.1);
	}*/
	/*return max(value1, value2)
			+ Dvc_Globals::Dvc_Discount(Dvc_config, count_x + count_y - 1) * (10);*/
	return value1
				+ Dvc_Globals::Dvc_Discount(Dvc_config, min_xy + diff_xy - 1) * (/*10*/GOAL_REWARD);
}

void UncNavigationState::InitCells(int sizeX, int sizeY)
{
	sizeX_=sizeX; sizeY_=sizeY;
	if(cells==NULL)
	{
		/*if(use_multi_thread_)
		{
			cells=cell_memory_pool_->Allocate( sizeX_*sizeY_);
			//HANDLE_ERROR(cudaHostAlloc((void**)&cells, sizeX_*sizeY_*sizeof(bool),0));
		}
		else*/
		 cells=new bool[sizeX_*sizeY_];
	}
	memset((void*)cells,0, sizeX*sizeY*sizeof(bool));
}
UncNavigationState::~UncNavigationState()
{
	if(cells!=NULL)
	{
		/*if(use_multi_thread_)
			;//HANDLE_ERROR(cudaFreeHost(cells));
		else*/
			delete [] cells;
	}
}

/*HOST virtual void Dvc_UncNavigationState::assign(State* host_state)
{
	UncNavigationState* src=static_cast<UncNavigationState*>(host_state);
	rob.x=src->rob.x; rob.y=src->rob.y;
	goal=src->goal;
	state_id=src->state_id;
	scenario_id=src->scenario_id;
	weight=src->weight;

	InitCellsManaged(src->sizeX_,src->sizeY_);

	HANDLE_ERROR(cudaMemcpy((void*)cells, (const void*)src->cells, sizeX_*sizeY_*sizeof(bool), cudaMemcpyHostToDevice));
}*/

/*DEVICE string Dvc_UncNavigationState::text() const {
	return "id = " + to_string(state_id);
}*/



/*
DEVICE Dvc_BaseUncNavigation::Dvc_BaseUncNavigation(int size, int obstacles)// :
	//grid_(size, size),
	//size_(size),
	//num_obstacles_(obstacles)
{
	if (size == 4 && obstacles == 4) {
		Init_4_4();
	} else if (size == 5 && obstacles == 5) {
		Init_5_5();
	} else if (size == 5 && obstacles == 7) {
		Init_5_7();
	} else if (size == 7 && obstacles == 8) {
		Init_7_8();
	} else if (size == 11 && obstacles == 11) {
		Init_11_11();
	} else {
		InitGeneral();
	}
	// put obstacles in random positions
	//InitGeneral();
	// InitStates();

}
*/

/*

DEVICE void Dvc_BaseUncNavigation::RandGate(Dvc_UncNavigationState* nav_state) const
{
   DvcCoord pos;
   pos=nav_state->GateNorth();
   nav_state->GridOpen(pos) = (bool)Random::RANDOM.NextInt(2); // randomly put obstacles there
   pos=nav_state->GateEast();
   nav_state->GridOpen(pos) = (bool)Random::RANDOM.NextInt(2); // randomly put obstacles there
   pos=nav_state->GateWest();
   nav_state->GridOpen(pos) = (bool)Random::RANDOM.NextInt(2); // randomly put obstacles there
}

DEVICE void Dvc_BaseUncNavigation::RandMap(Dvc_UncNavigationState* nav_state, float ObstacleProb, int skip) const
{
	//assign obstacle with prob ObstacleProb at 1/skip of the map
	for (int x=0;x<nav_state->sizeX_;x+=skip)
		for (int y=0;y<nav_state->sizeY_;y+=skip)
		{
			DvcCoord pos(x,y);
			if(nav_state->Grid(pos)==false
					&& pos!=nav_state->goal )//ignore existing obstacles
				nav_state->GridOpen(pos)=Random::RANDOM.NextDouble()<ObstacleProb? true:false;
		}
}
DEVICE void Dvc_BaseUncNavigation::RandMap(Dvc_UncNavigationState* nav_state) const
{
	for(int i=0;i<NumObstacles;i++)
	{
		DvcCoord pos=obstacle_pos[i];
		nav_state->GridOpen(pos)=(bool)Random::RANDOM.NextInt(2);//put obstacle there
	}
}


DEVICE void Dvc_BaseUncNavigation::CalObstacles(float prob) const
{
	NumObstacles=prob*(float)(size_*size_);
	if(NumObstacles>0)
	{
		obstacle_pos=new DvcCoord[NumObstacles];
		int ExistingObs=0;
		//allocate a temporary state
		Dvc_UncNavigationState* nav_state = Dvc_memory_pool_.Allocate();
		nav_state->InitCells(size_,size_);
		//put the goal first
		nav_state->FixedGoal();
		//generate obstacles
		DvcCoord pos;
		do {
			do {
				pos = DvcCoord(Random::RANDOM.NextInt(size_),
					Random::RANDOM.NextInt(size_));
			} while (nav_state->Grid(pos) == true || pos==nav_state->goal);// check for random free map position
			nav_state->GridOpen(pos)=true;//put obstacle there
			obstacle_pos[ExistingObs]=pos;
			ExistingObs++;
		}while(ExistingObs<NumObstacles);
	}
	else
		obstacle_pos=NULL;
}

DEVICE void Dvc_BaseUncNavigation::FreeObstacles() const
{
	delete [] obstacle_pos;
}

DEVICE Dvc_State* Dvc_BaseUncNavigation::CreateStartState(string type) const {
	if(NewRound)
	{
		CalObstacles(0.0);
		NewRound=false;
	}
	//Dvc_UncNavigationState state(size_, size_);
	Dvc_UncNavigationState* startState = Dvc_memory_pool_.Allocate();
	startState->InitCells(size_,size_);
	//put the goal first
	startState->FixedGoal();
	// put obstacles in fixed positions
	DvcCoord pos;
	if (num_obstacles_>0)
	{
		pos.x=size_/4; pos.y=3*size_/4;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>1)
	{
		pos.x=2*size_/4; pos.y=2*size_/4;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>2)
	{
		pos.x=size_/2-2; pos.y=0;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>3)
	{
		pos.x=0; pos.y=1*size_/4;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>4)
	{
		pos.x=size_-2; pos.y=1*size_/4+1;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	//RandGate(startState);
	RandMap(startState);//Generate map using the obstacle positions specified in obstacle_pos

	// pick a random position for the robot, always do this in the last step
	UniformRobPos(startState);
	//AreaRobPos(startState,2);
	return startState;
}
DEVICE void Dvc_BaseUncNavigation::UniformRobPos(Dvc_UncNavigationState* startState) const
{
	DvcCoord pos;
	do {
		pos = DvcCoord(Random::RANDOM.NextInt(size_),
			Random::RANDOM.NextInt(size_));
	} while (startState->Grid(pos) == true || pos==startState->goal);// check for random free map position
	startState->rob=pos;//put robot there
}
DEVICE void Dvc_BaseUncNavigation::AreaRobPos(Dvc_UncNavigationState* startState, int area_size) const
{
	// robot start from top middle block of area_size
	DvcCoord Robpos(size_/2-area_size/2-1,size_-area_size);
	DvcCoord pos;

	bool has_slot=false;
	for(int x=Robpos.x;x<Robpos.x+area_size;x++)
		for(int y=Robpos.y;y<Robpos.y+area_size;y++)
		{
			if(startState->Grid(x,y) == false)
			{
				has_slot=true;
				break;
			}
		}

	if(has_slot)
	{
		do {
			pos = Robpos+DvcCoord(Random::RANDOM.NextInt(area_size),
				Random::RANDOM.NextInt(area_size));
		} while (pos==startState->goal);// check for random free map position
		startState->GridOpen(pos) = false ;
		startState->rob=pos;//put robot there
	}
	else
	{
		AreaRobPos(startState,area_size+2);//enlarge the area and re-assign
	}
}
*/

/*

DEVICE Belief* Dvc_BaseUncNavigation::InitialBelief(const Dvc_State* start, string type) const {
	int N = Globals::config.num_scenarios*10;


	vector<Dvc_State*> particles(N);
	for (int i = 0; i < N; i++) {
		particles[i] = CreateStartState();
		//particles[i] = CreateFixedStartState();
		particles[i]->weight = 1.0 / N;
	}
	FreeObstacles();
	NewRound=true;
	return new ParticleBelief(particles, this);
}

class Dvc_UncNavigationParticleUpperBound1: public Dvc_ParticleUpperBound {
protected:
	const Dvc_BaseUncNavigation* rs_model_;
public:
	DEVICE Dvc_UncNavigationParticleUpperBound1(const Dvc_BaseUncNavigation* model) :
		rs_model_(model) {
	}

	DEVICE double Value(const Dvc_State& state) const {
		const Dvc_UncNavigationState& nav_state =
			static_cast<const Dvc_UncNavigationState&>(state);
		int count_x=abs(nav_state.rob.x-nav_state.goal.x);
		int count_y=abs(nav_state.rob.y-nav_state.goal.y);
		double value1=0, value2=0;
		for (int i=0;i<count_x;i++)
		{
			value1+=Globals::Discount(i)*(-0.1);
		}
		for (int j=0;j<count_y;j++)
		{
			value1+=Globals::Discount(count_x+j)*(-0.1);
		}

		for (int j=0;j<count_y;j++)
		{
			value2+=Globals::Discount(j)*(-0.1);
		}
		for (int i=0;i<count_x;i++)
		{
			value2+=Globals::Discount(count_y+i)*(-0.1);
		}

		return max(value1,value2)+Globals::Discount(count_x+count_y-1)*(10);
	}
};
DEVICE Dvc_ScenarioUpperBound* Dvc_BaseUncNavigation::CreateScenarioUpperBound(string name,
	string particle_bound_name) const {
	Dvc_ScenarioUpperBound* bound = NULL;
	if (name == "UB1") {
		bound = new Dvc_UncNavigationParticleUpperBound1(this);
	} else if (name == "DEFAULT" || "TRIVIAL") {
		bound = new Dvc_TrivialParticleUpperBound(this);
	} else if (name == "UB2") {
		bound = new UncNavigationParticleUpperBound2(this);
	} else if (name == "DEFAULT" || name == "MDP") {
		bound = new UncNavigationMDPParticleUpperBound(this);
	} else if (name == "APPROX") {
		bound = new UncNavigationApproxParticleUpperBound(this);
	} else {
		cerr << "Unsupported scenario upper bound: " << name << endl;
		exit(0);
	}
	return bound;
}

DEVICE Dvc_ScenarioLowerBound* Dvc_BaseUncNavigation::CreateScenarioLowerBound(string name, string
	particle_bound_name) const {
	if (name == "TRIVIAL") {
		return new Dvc_TrivialParticleLowerBound(this);
	} else if (name == "DEFAULT" || name == "EAST") {
		// scenario_lower_bound_ = new BlindPolicy(this, Dvc_NavCompass::EAST);
		return new UncNavigationEastScenarioLowerBound(this);
	} else if (name == "DEFAULT" || name == "RANDOM") {
		return new RandomPolicy(this,
			CreateParticleLowerBound(particle_bound_name));
	} else if (name == "ENT") {
		return new UncNavigationENTScenarioLowerBound(this);
	} else if (name == "MMAP") {
		return new UncNavigationMMAPStateScenarioLowerBound(this);
	} else if (name == "MODE") {
		// scenario_lower_bound_ = new ModeStatePolicy(this);
		return NULL; // TODO
	} else {
		cerr << "Unsupported lower bound algorithm: " << name << endl;
		exit(0);
		return NULL;
	}
}

DEVICE void Dvc_BaseUncNavigation::PrintState(const Dvc_State& state, ostream& out) const {
	Dvc_UncNavigationState navstate=static_cast<const Dvc_UncNavigationState&>(state);
	int Width=7;
	int Prec=1;
	out << endl;
	for (int x = 0; x < size_ + 2; x++)
	{
		out.width(Width);out.precision(Prec);	out << "# ";
	}
	out << endl;
	for (int y = size_ - 1; y >= 0; y--) {
		out.width(Width);out.precision(Prec);	out << "# ";
		for (int x = 0; x < size_; x++) {
			DvcCoord pos(x, y);
			int obstacle = navstate.Grid(pos);
			if (navstate.goal == DvcCoord(x, y))
			{
				out.width(Width);out.precision(Prec);	out << "G ";
			}
			else if (GetRobPos(&state) == DvcCoord(x, y))
			{
				out.width(Width);out.precision(Prec);	out << "R ";
			}
			else if (obstacle ==true)
			{
				out.width(Width);out.precision(Prec);	out << "X ";
			}
			else
			{
				out.width(Width);out.precision(Prec);	out << ". ";
			}
		}
		out.width(Width);out.precision(Prec);	out << "#" << endl;
	}
	for (int x = 0; x < size_ + 2; x++)
	{
		out.width(Width);out.precision(Prec);	out << "# ";
	}
	out << endl;

	//PrintBelief(*belief_);
}
DEVICE void Dvc_BaseUncNavigation::PrintBeliefMap(float** Beliefmap, std::ostream& out) const
{
	ios::fmtflags old_settings = out.flags();
	//out.precision(5);
	//out.width(4);
	int Width=6;
	int Prec=1;
	out << endl;
	out.width(Width);out.precision(Prec);
	for (int x = 0; x < size_ + 2; x++)
	{
		out.width(Width);out.precision(Prec);	out << "# ";
	}
	out << endl;
	for (int y = size_ - 1; y >= 0; y--) {
		out.width(Width);out.precision(Prec); out << "# ";
		for (int x = 0; x < size_; x++) {
			out.width(Width);out.precision(Prec);
			if(Beliefmap[x][y]>0.2)
			{
				out.width(Width-3);
				out <<"<"<<Beliefmap[x][y]<<">"<<" ";
				out.width(Width);
			}
			else
				out <<Beliefmap[x][y]<<" ";
		}
		out.width(Width);out.precision(Prec); out << "#" << endl;
	}
	for (int x = 0; x < size_ + 2; x++)
		{out.width(Width);out.precision(Prec); out << "# ";}
	out << endl;
	out.flags(old_settings);
}
DEVICE void Dvc_BaseUncNavigation::AllocBeliefMap(float**& Beliefmap) const
{
	Beliefmap =new float*[size_];
	for(int i=0;i<size_;i++)
	{
		Beliefmap[i]=new float[size_];
		memset((void*)Beliefmap[i], 0 , size_*sizeof(float));
	}
}
DEVICE void Dvc_BaseUncNavigation::ClearBeliefMap(float**& Beliefmap) const
{
	for(int i=0;i<size_;i++)
	{
		delete [] Beliefmap[i];
	}
	delete [] Beliefmap;
}
DEVICE void Dvc_BaseUncNavigation::PrintBelief(const Belief& belief, ostream& out) const {
	const vector<Dvc_State*>& particles =
		static_cast<const ParticleBelief&>(belief).particles();
	out << "Robot position belief:";
	float** Beliefmap;float** ObsBeliefmap;
	AllocBeliefMap(Beliefmap);AllocBeliefMap(ObsBeliefmap);
	float GateState[3];

	memset((void*)GateState, 0 , 3*sizeof(float));
	for (int i = 0; i < particles.size(); i++) {

		const Dvc_UncNavigationState* navstate = static_cast<const Dvc_UncNavigationState*>(particles[i]);

		GateState[0]+=((int)navstate->Grid(navstate->GateWest()))*navstate->weight;
		GateState[1]+=((int)navstate->Grid(navstate->GateNorth()))*navstate->weight;
		GateState[2]+=((int)navstate->Grid(navstate->GateEast()))*navstate->weight;

		for (int x=0; x<size_; x++)
			for (int y=0; y<size_; y++)
			{
				if(navstate->rob.x==x && navstate->rob.y==y)
					Beliefmap[x][y]+=navstate->weight;
				DvcCoord pos(x,y);
				ObsBeliefmap[x][y]+=((int)navstate->Grid(pos))*navstate->weight;
					//out << "Weight=" << particles[i]->weight<<endl;
			}
	}

	PrintBeliefMap(Beliefmap,out);
	out << "Map belief:";
	PrintBeliefMap(ObsBeliefmap,out);
	out << "Gate obstacle belief:"<<endl;
	for (int i=0;i<3;i++)
		out<<GateState[i]<<" ";
	out<<endl;
	ClearBeliefMap(Beliefmap);ClearBeliefMap(ObsBeliefmap);
}

DEVICE void Dvc_BaseUncNavigation::PrintAction(int action, ostream& out) const {
	if (action < E_STAY)
		out << Dvc_NavCompass::CompassString[action] << endl;
	if (action == E_STAY)
		out << "Stay" << endl;
}
*/
/*
DEVICE Dvc_State* Dvc_BaseUncNavigation::Allocate(int state_id, double weight) const {
	//Dvc_UncNavigationState* state = Dvc_memory_pool_.Allocate();
	Dvc_UncNavigationState* state = new Dvc_UncNavigationState();
	state->state_id = state_id;
	state->weight = weight;

	return state;
}

DEVICE Dvc_State* Dvc_BaseUncNavigation::Dvc_Copy(const Dvc_State* particle) {
	//Dvc_UncNavigationState* state = Dvc_memory_pool_.Allocate();
	Dvc_UncNavigationState* state = new Dvc_UncNavigationState();
	*state = *static_cast<const Dvc_UncNavigationState*>(particle);
	state->SetAllocated();
	return state;
}

DEVICE void Dvc_BaseUncNavigation::Dvc_Free(Dvc_State* particle) {
	//Dvc_memory_pool_.Free(static_cast<Dvc_UncNavigationState*>(particle));
	delete particle;
}


DEVICE int Dvc_BaseUncNavigation::NumActiveParticles() const {
	return Dvc_memory_pool_.num_allocated();
}

DEVICE int Dvc_BaseUncNavigation::NumObservations() { // one dummy terminal state
	return 16;
}


DEVICE DvcCoord Dvc_BaseUncNavigation::GetRobPos(const Dvc_State* state) const {
	return static_cast<const Dvc_UncNavigationState*>(state)->rob;
}


DEVICE OBS_TYPE Dvc_BaseUncNavigation::GetObservation(double rand_num,
	const Dvc_UncNavigationState& nav_state) {
	OBS_TYPE obs;
	double TotalProb=0;
	for(int i=0;i<NumObservations();i++)//pick an obs according to the prob of each one
	{
		TotalProb+=ObsProb(i, nav_state, E_STAY);
		if(rand_num<=TotalProb)
		{	obs=i;	break;	}
	}
	return obs;
}*/


/*
DEVICE int Dvc_BaseUncNavigation::GetX(const Dvc_UncNavigationState* state) const {
	return state->rob.x;
}

DEVICE void Dvc_BaseUncNavigation::IncX(Dvc_UncNavigationState* state) const {
	state->rob.x+=1;
}

DEVICE void Dvc_BaseUncNavigation::DecX(Dvc_UncNavigationState* state) const {
	state->rob.x-=1;
}

DEVICE int Dvc_BaseUncNavigation::GetY(const Dvc_UncNavigationState* state) const {
	return state->rob.y;
}

DEVICE void Dvc_BaseUncNavigation::IncY(Dvc_UncNavigationState* state) const {
	state->rob.y+=1;
}

DEVICE void Dvc_BaseUncNavigation::DecY(Dvc_UncNavigationState* state) const {
	state->rob.y-=1;
}


DEVICE Dvc_UncNavigationState Dvc_BaseUncNavigation::NextState(Dvc_UncNavigationState& s, int a) const {
	if (s.rob==s.goal)// terminal state is an absorbing state
		return s;

    double Rand=Random::RANDOM.NextDouble();
	DvcCoord rob_pos = s.rob;
	Dvc_UncNavigationState newState(s);
	if (a < E_STAY) {//movement actions
	    if(Rand<0.8)// only succeed with 80% chance
	    	rob_pos += Dvc_NavCompass::DIRECTIONS[a];
		if (s.Inside(rob_pos) && s.CollisionCheck(rob_pos)==false) {
			newState.rob=rob_pos;//move the robot
		} else {
			;// don't move the robot
		}
		return newState;
	} else if (a == E_STAY) {//stay action
		return newState;
	} else //unsupported action
		return s;
}

DEVICE double Dvc_BaseUncNavigation::Reward(Dvc_UncNavigationState& s, int a) const {
	if (s.rob==s.goal)// at terminal state, no reward
		return 0;
	DvcCoord rob_pos = s.rob;
	if (a < E_STAY) {
	    double Rand=Random::RANDOM.NextDouble();
	    if(Rand<0.8)// only succeed with 80% chance
	    	rob_pos += Dvc_NavCompass::DIRECTIONS[a];
	    if(rob_pos==s.goal)// arrive goal
	    	return 10;
		if (s.Inside(rob_pos) && s.CollisionCheck(rob_pos)==false) {
			return -0.1;// small cost for each move
		} else
			return -1;//run into obstacles or walls
	} else if (a == E_STAY) {
		return -0.1;// small cost for each step
	} else //unsupported action
		return 0;
}
*/

} // namespace despot
