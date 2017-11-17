#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUutil/Dvc_memorypool.h>

namespace despot {
#define DIM 128
DEVICE static Dvc_MemoryPool<int>* dvc_action_pool_=NULL;
DEVICE static Dvc_MemoryPool<OBS_TYPE>* dvc_obs_pool_=NULL;

__global__ void CreateDvcMemoryPools1()
{
	if(dvc_action_pool_==NULL)
		dvc_action_pool_=new Dvc_MemoryPool<int>;
	if(dvc_obs_pool_==NULL)
		dvc_obs_pool_=new Dvc_MemoryPool<OBS_TYPE>;
	//printf("Create history pool\n");
	//printf("dvc_action_pool_=%0x, dvc_obs_pool_=%0x, num1=%d, num2=%d\n", dvc_action_pool_,dvc_obs_pool_,
	//		dvc_action_pool_->num_allocated_,dvc_obs_pool_->num_allocated_);
}

void Dvc_History::CreateMemoryPool(int mode) const
{
	if(mode==0)
	{
		CreateDvcMemoryPools1<<<1,1,1>>>();
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
}

__global__ void DestoryDvcMemoryPools1(int mode)
{
	//printf("Detory history pool\n");
	switch (mode)
	{
	case 0:
		if(dvc_action_pool_){delete dvc_action_pool_;dvc_action_pool_=NULL;}
		if(dvc_obs_pool_){delete dvc_obs_pool_;dvc_obs_pool_=NULL;}
		break;
	case 1:
		dvc_action_pool_->DeleteContents();
		dvc_obs_pool_->DeleteContents();
		break;
	};
}
void Dvc_History::DestroyMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;
	DestoryDvcMemoryPools1<<<1,1,1>>>(mode);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

__global__ void InitHistory(Dvc_History* Dvc_history, int length, int num_particles)
{
	//int SID=blockIdx.x*blockDim.x+threadIdx.x;

	//if(SID==0)
	//{
	//for(int SID=0;SID<num_particles;SID++)
	//{
	int SID=0;
	//printf("dvc_action_pool_=%0x, dvc_obs_pool_=%0x\n", dvc_action_pool_,dvc_obs_pool_);
		Dvc_history[SID].actions_=dvc_action_pool_->Allocate(length);
		Dvc_history[SID].observations_=dvc_obs_pool_->Allocate(length);
		//Dvc_history[SID].actions_=(int*)malloc(sizeof(int)*length);
		//Dvc_history[SID].observations_=(OBS_TYPE*)malloc(sizeof(OBS_TYPE)*length);
		Dvc_history[SID].currentSize_=0;
		//printf("Alloc history %0x with actionList=%0x, obsList=%0x, with length %d\n",
		//		Dvc_history+SID,Dvc_history[SID].actions_,Dvc_history[SID].observations_, length);
	//}
	//}
}

void Dvc_History::InitInGPU(int num_particles, Dvc_History* Dvc_history, int length,void* cuda_stream)
{
	dim3 grid(1,1);dim3 threads(1,1);
	if(cuda_stream!=NULL)
	{
		InitHistory<<<grid, threads,0,(*(cudaStream_t*)cuda_stream)>>>( Dvc_history, length,num_particles);
		//HANDLE_ERROR(cudaStreamSynchronize((*(cudaStream_t*)cuda_stream)));
	}
	else
	{
		InitHistory<<<grid, threads>>>( Dvc_history, length,num_particles);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
}

__global__ void CopyHistory(int* particleIDs, Dvc_History* des, int* src_action, OBS_TYPE* src_obs, int size, int num_particles)
{
	/*int TID=blockIdx.x*blockDim.x+threadIdx.x;
	int pos=blockIdx.y;

	if(TID<num_particles)
	{
		int SID=particleIDs[TID];
		//if(SID==0)
		//{*/
	int pos=threadIdx.x;
	int SID=0;
		if(size!=0)
		{
			des[SID].actions_[pos]=src_action[pos];
			des[SID].observations_[pos]=src_obs[pos];
		}

		if(pos==0)
			des[SID].currentSize_=size;
	/*}
	//}*/
}

void Dvc_History::CopyToGPU(int num_particles,int* particleIDs, Dvc_History* Dvc_history, History* history)
{
	int* tmp_actions=NULL;OBS_TYPE* tmp_obs=NULL;
	if(history->Size()>0)
	{
		HANDLE_ERROR(cudaMalloc((void**)&tmp_actions,history->Size()*sizeof(int)));
		HANDLE_ERROR(cudaMalloc((void**)&tmp_obs,history->Size()*sizeof(OBS_TYPE)));

		HANDLE_ERROR(cudaMemcpy(tmp_actions,history->Action(),(int)history->Size()*sizeof(int),cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(tmp_obs,history->Observation(),history->Size()*sizeof(OBS_TYPE), cudaMemcpyHostToDevice));
	}


	//dim3 grid((num_particles+DIM-1)/DIM,history->Size());dim3 threads(DIM,1);
	//if(history->Size()==0)grid.x=1;
	dim3 grid(1,1);dim3 threads(history->Size(),1);

	CopyHistory<<<grid, threads>>>(particleIDs, Dvc_history, tmp_actions,tmp_obs,history->Size(), num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());
	if(history->Size()>0)
	{
		HANDLE_ERROR(cudaFree(tmp_actions));
		HANDLE_ERROR(cudaFree(tmp_obs));
	}
}


__global__ void AddToBack(Dvc_History* Dvc_history,int action, OBS_TYPE obs)
{
	int SID=threadIdx.x;

	if(SID==0)
	{
		//printf("Dvc_history=%0x, currentSize=%d, actionPtr=%0x, obsPtr=%0x\n", Dvc_history,Dvc_history->currentSize_,Dvc_history->actions_, Dvc_history->observations_);
		Dvc_history->Add(action,obs);
	}
}

void Dvc_History::Dvc_Add(Dvc_History* Dvc_history,int action, OBS_TYPE obs, void* cudaStream)
{
	try{
	dim3 grid(1,1);dim3 threads(1,1);
	if(cudaStream==NULL)
	{
		AddToBack<<<grid, threads>>>(Dvc_history,action,obs);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	else
	{
		AddToBack<<<grid, threads,0, *(cudaStream_t*)cudaStream>>>(Dvc_history,action,obs);
	}
	}
	catch(...)
	{
		std::cout<<"Exeption in "<<__FUNCTION__<<" at "<<__LINE__<<std::endl;
		exit(-1);
	}
}

__global__ void GlbTruncate(Dvc_History* Dvc_history,int size)
{
	int SID=threadIdx.x;

	if(SID==0)
	{
		Dvc_history->Truncate(size);
	}
}

void Dvc_History::Dvc_Trunc(Dvc_History* Dvc_history, int size, void* cudaStream)
{
	dim3 grid(1,1);dim3 threads(1,1);
	if(cudaStream==NULL)
	{
		GlbTruncate<<<grid, threads>>>(Dvc_history,size);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	else
	{
		GlbTruncate<<<grid, threads,0, *(cudaStream_t*)cudaStream>>>(Dvc_history,size);
	}

}


}//namespace despot
