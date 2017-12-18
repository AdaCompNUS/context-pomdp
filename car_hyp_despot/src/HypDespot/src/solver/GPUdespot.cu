#include <despot/solver/GPUdespot.h>
//#include <despot/solver/pomcp.h>
#include <despot/GPUcore/GPUnode.h>
#include <despot/GPUutil/GPUmap.h>
#include <despot/GPUcore/thread_globals.h>
#include <despot/simple_tui.h>
#include <string.h>
#include <disabled_util.h>

using namespace std;
static atomic<double> InitBoundTime(0);
//static double BlockerCheckTime=0;
//static double TreeExpansionTime=0;
//static double PathTrackTime=0;
static atomic<double> AveRewardTime(0);
static atomic<double> MakeObsNodeTime(0);
//static double ModelStepTime=0;
//static double ParticleCopyTime=0;
//static double ObsCountTime=0;
static atomic<double> CopyParticleTime(0);
static atomic<double> CopyHistoryTime(0);
static atomic<double> MakePartitionTime(0);
static atomic<int> HitCount(0);
static atomic<double> AveNumParticles(0);
static atomic<double> TotalExpansionTime(0);

static atomic<double> BarrierWaitTime(0);
static atomic<double> DataBackTime(0);
//static cudaStream_t* cuda_streams = NULL;
static int stream_counter = 1;
int Obs_parallel_level = OBS_PARALLEL_NONE;
int Obs_type= OBS_LONG64;
namespace despot {

static int TotalNumParticles = 0;

static Dvc_RandomStreams** Dvc_streams = NULL;
static Dvc_History** Dvc_history = NULL;

static int** Dvc_particleIDs_long = NULL;
static Dvc_State** Dvc_particles_long = NULL;

static void** Dvc_MC_Data = NULL;
static float** Dvc_r_all_a = NULL;

static float ** Dvc_ub_all_a_p = NULL;
static float** Dvc_uub_all_a_p = NULL;
static Dvc_ValuedAction** Dvc_lb_all_a_p = NULL;
static Dvc_ValuedAction** Hst_lb_all_a_p = NULL;

/*Host memory to get dev_info*/
static void** Hst_MC_Data = NULL;
static int MC_DataSize=0;

static float** Hst_r_all_a = NULL;
static float ** Hst_ub_all_a_p = NULL;
static float** Hst_uub_all_a_p = NULL;

static int cudaCoreNum = 0;
static int asyncEngineCount = 0;

DEVICE bool (*DvcModelStep_)(Dvc_State&, float, int, float&, OBS_TYPE&)=NULL;
//for more complex observations
DEVICE bool (*DvcModelStepIntObs_)(Dvc_State&, float, int, float&, int*)=NULL;

DEVICE Dvc_State* (*DvcModelAlloc_)(int num)=NULL;
DEVICE Dvc_State* (*DvcModelCopy_)(const Dvc_State*, int pos)=NULL;
DEVICE void (*DvcModelCopyNoAlloc_)(Dvc_State*, const Dvc_State*, int pos,
		bool offset_des)=NULL;
DEVICE void (*DvcModelCopyToShared_)(Dvc_State*, const Dvc_State*, int pos,
		bool offset_des)=NULL;
DEVICE void (*DvcModelFree_)(Dvc_State*)=NULL;
DEVICE Dvc_ValuedAction (*DvcLowerBoundValue_)( Dvc_State *,
		Dvc_RandomStreams&, Dvc_History&)=NULL;
DEVICE float (*DvcUpperBoundValue_)(const Dvc_State*, int, Dvc_History&)=NULL;

__global__ void GPU_InitBounds(float* upper, float* utility_upper,
		Dvc_ValuedAction* default_move, int num_particles, Dvc_State* particles,
		const int* particleIDs, Dvc_RandomStreams* streams,
		Dvc_History* local_history, int action, OBS_TYPE* observations,
		int depth);
DEVICE Dvc_ValuedAction GPU_InitLowerBound(Dvc_State* particles,
		Dvc_RandomStreams& streams, Dvc_History& local_history, int depth);
DEVICE float GPU_InitUpperBound(int scenarioID, const Dvc_State* particles,
/*Dvc_RandomStreams& streams,*/Dvc_History& local_history, int depth);
//DEVICE void (*DvcModelFreeList_)(const Dvc_State*);
void AdvanceStreamCounter(int stride);

__global__ void AllocConfig() {
	Dvc_config = new Dvc_Config;
}
__global__ void PassConfig(Dvc_Config* src) {
	Dvc_config = new Dvc_Config;
	Dvc_config->search_depth = src->search_depth;
	Dvc_config->discount = src->discount;
	Dvc_config->root_seed = src->root_seed;
	Dvc_config->time_per_move = src->time_per_move;
	Dvc_config->num_scenarios = src->num_scenarios;
	Dvc_config->pruning_constant = src->pruning_constant;
	Dvc_config->xi = src->xi;
	Dvc_config->sim_len = src->sim_len;
	Dvc_config->max_policy_sim_len = src->max_policy_sim_len;
	Dvc_config->noise = src->noise;
	Dvc_config->silence = src->silence;
}
__global__ void InitRandom() {
	//Dvc_random->seed_=src->seed_;
	Dvc_random = new Dvc_Random;
}
/*
 __global__ void InitDepth(int TotalNumParticles)
 {
 int i=threadIdx.x;
 if(i==0)
 Dvc_initial_depth=(int*)malloc(TotalNumParticles*sizeof(int));
 __syncthreads();

 Dvc_initial_depth[i]=0;
 }
 */

void DESPOT::PrepareGPUMemory(const Config& config, int num_actions,
		int num_obs) {
	clock_t start = clock();

	//ClearGPUMemory();

	TotalNumParticles = config.num_scenarios;

	AllocConfig<<<1, 1, 1>>>();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Dvc_Config::CopyToGPU(&config);

	int num_copies =1;
	if(use_multi_thread_)num_copies=NUM_THREADS;

	if (NUM_THREADS > 1 && use_multi_thread_) {
		Dvc_streams = new Dvc_RandomStreams*[NUM_THREADS];
		Dvc_r_all_a = new float*[NUM_THREADS];
		Hst_r_all_a = new float*[NUM_THREADS];
		Dvc_obs_all_a_and_p = new OBS_TYPE*[NUM_THREADS];
		Hst_obs_all_a_and_p = new OBS_TYPE*[NUM_THREADS];
		Dvc_obs_int_all_a_and_p = new int*[NUM_THREADS];
		Hst_obs_int_all_a_and_p = new int*[NUM_THREADS];
		Dvc_term_all_a_and_p = new bool*[NUM_THREADS];
		Hst_term_all_a_and_p = new bool*[NUM_THREADS];
		Dvc_ub_all_a_p = new float*[NUM_THREADS];
		Dvc_uub_all_a_p = new float*[NUM_THREADS];
		Dvc_lb_all_a_p = new Dvc_ValuedAction*[NUM_THREADS];
		Hst_lb_all_a_p = new Dvc_ValuedAction*[NUM_THREADS];
		Hst_ub_all_a_p = new float*[NUM_THREADS];
		Hst_uub_all_a_p = new float*[NUM_THREADS];
		Dvc_particleIDs_long = new int*[NUM_THREADS];
		Dvc_particles_long = new Dvc_State*[NUM_THREADS];
		Dvc_MC_Data=new void*[NUM_THREADS];
		Hst_MC_Data=new void*[NUM_THREADS];
	} else {
		num_copies = 1;
		Dvc_streams = new Dvc_RandomStreams*;
		Dvc_r_all_a = new float*;
		Hst_r_all_a = new float*;
		//Dvc_qnode=new Dvc_QNode*;
		Dvc_obs_all_a_and_p = new OBS_TYPE*;
		Hst_obs_all_a_and_p = new OBS_TYPE*;
		Dvc_obs_int_all_a_and_p = new int*;
		Hst_obs_int_all_a_and_p = new int*;
		Dvc_term_all_a_and_p = new bool*;
		Hst_term_all_a_and_p = new bool*;
		Dvc_ub_all_a_p = new float*;
		Dvc_uub_all_a_p = new float*;
		Dvc_lb_all_a_p = new Dvc_ValuedAction*;
		Hst_lb_all_a_p = new Dvc_ValuedAction*;
		Hst_ub_all_a_p = new float*;
		Hst_uub_all_a_p = new float*;
		Dvc_particleIDs_long = new int*;
		Dvc_particles_long = new Dvc_State*;
		Dvc_MC_Data=new void*;
		Hst_MC_Data=new void*;
	}

	for (int i = 0; i < num_copies; i++) {
		HANDLE_ERROR(
				cudaMallocManaged((void** )&Dvc_streams[i],
						sizeof(Dvc_RandomStreams)));
		Dvc_RandomStreams::Init(Dvc_streams[i], TotalNumParticles,
				config.search_depth,(i==0)?true:false);
		int offset_obs=0;int offset_term=0;

		if(Obs_type==OBS_INT_ARRAY)
		{
			offset_obs=num_actions * sizeof(float);
			int blocksize=sizeof(int)*num_Obs_element_in_GPU;
			if(offset_obs%blocksize!=0)
				offset_obs=(offset_obs/blocksize+1)*blocksize;
			//cout<<"offset_obs= "<<offset_obs<<", sizeof(OBS_TYPE)="<<blocksize<<" bytes"<<endl;

			offset_term=offset_obs+num_actions * TotalNumParticles * blocksize;
			if(offset_term%sizeof(bool)!=0) offset_term=(offset_term/sizeof(bool)+1)*sizeof(bool);
			//cout<<"offset_term= "<<offset_term<<", sizeof(bool)="<<sizeof(bool)<<" bytes"<<endl;
		}
		else
		{
			offset_obs=num_actions * sizeof(float);
			if(offset_obs%sizeof(OBS_TYPE)!=0) offset_obs=(offset_obs/sizeof(OBS_TYPE)+1)*sizeof(OBS_TYPE);
			//cout<<"offset_obs= "<<offset_obs<<", sizeof(OBS_TYPE)="<<sizeof(OBS_TYPE)<<" bytes"<<endl;

			offset_term=offset_obs+num_actions * TotalNumParticles * sizeof(OBS_TYPE);
			if(offset_term%sizeof(bool)!=0) offset_term=(offset_term/sizeof(bool)+1)*sizeof(bool);
			//cout<<"offset_term= "<<offset_term<<", sizeof(bool)="<<sizeof(bool)<<" bytes"<<endl;
		}

		int offset_ub=offset_term+num_actions * TotalNumParticles * sizeof(bool);
		if(offset_ub%sizeof(float)!=0) offset_ub=(offset_ub/sizeof(float)+1)*sizeof(float);
		//cout<<"offset_ub= "<<offset_ub<<", sizeof(float)="<<sizeof(float)<<" bytes"<<endl;

		int offset_uub=offset_ub+num_actions * TotalNumParticles * sizeof(float);
		if(offset_uub%sizeof(float)!=0) offset_uub=(offset_uub/sizeof(float)+1)*sizeof(float);
		//cout<<"offset_uub= "<<offset_uub<<", sizeof(float)="<<sizeof(float)<<" bytes"<<endl;

		int offset_lb=offset_uub+num_actions * TotalNumParticles * sizeof(float);
		if(offset_lb%sizeof(Dvc_ValuedAction)!=0) offset_lb=(offset_lb/sizeof(Dvc_ValuedAction)+1)*sizeof(Dvc_ValuedAction);
		//cout<<"offset_lb= "<<offset_lb<<", sizeof(Dvc_ValuedAction)="<<sizeof(Dvc_ValuedAction)<<" bytes"<<endl;

		MC_DataSize=offset_lb+num_actions * TotalNumParticles * sizeof(Dvc_ValuedAction);

		HANDLE_ERROR(
				cudaMalloc((void** )&Dvc_MC_Data[i],MC_DataSize));
		HANDLE_ERROR(
				cudaHostAlloc((void** )&Hst_MC_Data[i],MC_DataSize	, 0));

		Dvc_r_all_a[i]=(float*)Dvc_MC_Data[i];
		Hst_r_all_a[i]=(float*)Hst_MC_Data[i];

		if(Obs_type==OBS_INT_ARRAY)
		{
			Dvc_obs_int_all_a_and_p[i]=(int*)(Dvc_MC_Data[i]+offset_obs);
			Hst_obs_int_all_a_and_p[i]=(int*)(Hst_MC_Data[i]+offset_obs);
		}
		else{
			Dvc_obs_all_a_and_p[i]=(OBS_TYPE*)(Dvc_MC_Data[i]+offset_obs);
			Hst_obs_all_a_and_p[i]=(OBS_TYPE*)(Hst_MC_Data[i]+offset_obs);
		}

		Dvc_term_all_a_and_p[i]=(bool*)(Dvc_MC_Data[i]+offset_term);
		Hst_term_all_a_and_p[i]=(bool*)(Hst_MC_Data[i]+offset_term);

		Dvc_ub_all_a_p[i]=(float*)(Dvc_MC_Data[i]+offset_ub);
		Hst_ub_all_a_p[i]=(float*)(Hst_MC_Data[i]+offset_ub);

		Dvc_uub_all_a_p[i]=(float*)(Dvc_MC_Data[i]+offset_uub);
		Hst_uub_all_a_p[i]=(float*)(Hst_MC_Data[i]+offset_uub);

		Dvc_lb_all_a_p[i]=(Dvc_ValuedAction*)(Dvc_MC_Data[i]+offset_lb);
		Hst_lb_all_a_p[i]=(Dvc_ValuedAction*)(Hst_MC_Data[i]+offset_lb);

		HANDLE_ERROR(
				cudaHostAlloc((void** )&Dvc_particleIDs_long[i],
						TotalNumParticles * sizeof(int), 0));
	}
	cout<<"GPUDespot ouput Data size: "<<MC_DataSize<<"*"<<num_copies<<" bytes"<<endl;

	//HANDLE_ERROR(cudaHostAlloc((void**)&Hst_lb_all_a_p, num_actions*TotalNumParticles*sizeof(Dvc_ValuedAction),0));

	//HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_partition_ptrs, num_actions*num_obs*sizeof(Dvc_State*)));

	//if(Globals::config.rollout_type!="BLIND")
	//{
	Dvc_Random::init(TotalNumParticles);
	InitRandom<<<1, 1, 1>>>();
	HANDLE_ERROR(cudaDeviceSynchronize());
	//}
	//InitDepth<<<1,TotalNumParticles>>>(TotalNumParticles);
	//HANDLE_ERROR(cudaDeviceSynchronize());

	//Dvc_streams->assign(&streams);
	/*AveRewardTime=0;
	 InitBoundTime=0;
	 CopyHistoryTime=0;
	 MakePartitionTime=0;
	 MakeObsNodeTime=0;
	 HitCount=0;
	 AveNumParticles=0;*/

	cout << "GPU memory init time:"
			<< (double) (clock() - start) / CLOCKS_PER_SEC << endl;

}

__global__ void ShareStreamData(Dvc_RandomStreams* des,
		Dvc_RandomStreams* src) {

	des->num_streams_ = src->num_streams_;
	des->length_ = src->length_;
	for (int i = 0; i < des->num_streams_; i++) {
		des->streams_[i] = src->streams_[i];
	}
	des->position_=0;
}
void DESPOT::PrepareGPUStreams(const RandomStreams& streams,
		const Config& config, int TotalNumParticlesForTree) {
	clock_t start = clock();

	Dvc_RandomStreams::CopyToGPU(Dvc_streams[0], &streams);

	if (use_multi_thread_) {
		for (int i = 1; i < NUM_THREADS; i++) {
			dim3 grid1(1, 1);
			dim3 threads1(1, 1);
			ShareStreamData<<<grid1, threads1, 0, cuda_streams[i]>>>(
					Dvc_streams[i], Dvc_streams[0]);
		}
	} else {
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	//cout << "GPU stream copy time:"
	//		<< (double) (clock() - start) / CLOCKS_PER_SEC << endl;
}


__global__ void FreeHistory(Dvc_History* history, int num_particles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < num_particles) {
		//if(history[i].actions_) free(history[i].actions_);
		//if(history[i].observations_) free(history[i].observations_);
		history[i].currentSize_ = 0;
	}
}

__global__ void clearConfig() {
	if (Dvc_config != NULL) {
		delete Dvc_config;
		Dvc_config = NULL;
	}
}
__global__ void freeRandom() {
	//Dvc_random->seed_=src->seed_;
	if (Dvc_random != NULL) {
		delete Dvc_random;
		Dvc_random = NULL;
	}
}

/*__global__ void freeDepth()
 {
 //Dvc_random->seed_=src->seed_;
 if(Dvc_initial_depth!=NULL){free(Dvc_initial_depth);Dvc_initial_depth=NULL;}
 }*/

void DESPOT::ClearGPUMemory() {
	int thread_count = 1;
	if (use_multi_thread_)
		thread_count = NUM_THREADS;

	for (int i = 0; i < thread_count; i++) {
		if (Dvc_streams[i] != NULL) {
			Dvc_RandomStreams::Clear(Dvc_streams[i]);
			HANDLE_ERROR(cudaFree(Dvc_streams[i]));
			Dvc_streams[i] = NULL;
		}
		if (Dvc_MC_Data[i] != NULL) {
			HANDLE_ERROR(cudaFree(Dvc_MC_Data[i]));
			Dvc_MC_Data[i] = NULL;
		}
		if (Hst_MC_Data[i] != NULL) {
			HANDLE_ERROR(cudaFreeHost(Hst_MC_Data[i]));
			Hst_MC_Data[i] = NULL;
		}

		if (Dvc_particleIDs_long[i] != NULL) {
			HANDLE_ERROR(cudaFree(Dvc_particleIDs_long[i]));
			Dvc_particleIDs_long[i] = NULL;
		}
	}

	if (NUM_THREADS > 1 && use_multi_thread_) {
		delete[] Dvc_streams;
		delete[] Dvc_r_all_a;
		delete[] Hst_r_all_a;
		//delete [] Dvc_qnode;
		delete[] Dvc_obs_all_a_and_p;
		delete[] Hst_obs_all_a_and_p;
		delete[] Dvc_obs_int_all_a_and_p;
		delete[] Hst_obs_int_all_a_and_p;
		delete[] Dvc_term_all_a_and_p;
		delete[] Hst_term_all_a_and_p;
		delete[] Dvc_ub_all_a_p;
		delete[] Dvc_uub_all_a_p;
		delete[] Hst_ub_all_a_p;
		delete[] Hst_uub_all_a_p;
		delete[] Dvc_lb_all_a_p;
		delete[] Hst_lb_all_a_p;
		delete[] Dvc_particles_long;
		delete[] Dvc_particleIDs_long;
		delete thread_barrier;
		delete[] Dvc_MC_Data;
		delete[] Hst_MC_Data;
	} else {
		delete Dvc_streams;
		delete Dvc_r_all_a;
		delete Hst_r_all_a;
		//delete Dvc_qnode;
		delete Dvc_obs_all_a_and_p;
		delete Hst_obs_all_a_and_p;
		delete Dvc_obs_int_all_a_and_p;
		delete Hst_obs_int_all_a_and_p;
		delete Dvc_term_all_a_and_p;
		delete Hst_term_all_a_and_p;
		delete Dvc_ub_all_a_p;
		delete Dvc_uub_all_a_p;
		delete Hst_ub_all_a_p;
		delete Hst_uub_all_a_p;
		delete Dvc_lb_all_a_p;
		delete Hst_lb_all_a_p;
		delete Dvc_particles_long;
		delete Dvc_particleIDs_long;
		delete Dvc_MC_Data;
		delete Hst_MC_Data;
	}

	clearConfig<<<1, 1, 1>>>();
	HANDLE_ERROR(cudaDeviceSynchronize());

	Dvc_Random::clear();
	freeRandom<<<1, 1, 1>>>();
	HANDLE_ERROR(cudaDeviceSynchronize());

	if (cuda_streams) {
		for (int i = 0; i < NUM_THREADS; i++)
			HANDLE_ERROR(cudaStreamDestroy(cuda_streams[i]));
		delete[] cuda_streams;
		cuda_streams = NULL;
	}
}

extern __shared__ int localParticles[];

__global__ void
//__launch_bounds__(64, 16)
Step_(int total_num_scenarios, int num_particles, Dvc_State* vnode_particles,
		const int* vnode_particleIDs, float* step_reward_all_a,
		//float* step_reward_all_a_p,
		OBS_TYPE* observations_all_a_p, Dvc_State* new_particles,
		Dvc_RandomStreams* streams, int num_obs, bool* terminal_all_a_p,
		int parent_action) {
	int action = blockIdx.x;
	int PID = (blockIdx.y * blockDim.y + threadIdx.y) % num_particles;
	int obs_i = threadIdx.x;
	//int offset=blockDim.x*action;
	//int pos=TID+offset;
	int parent_PID = -1;

	if (blockIdx.y == 0 && threadIdx.y == 0 && obs_i == 0)
		step_reward_all_a[action] = 0;

	/*Counter for observation groups*/
	/*__shared__ unsigned int* action_partition_counter;
	 if(TID==0)
	 {
	 action_partition_counter=partition_counter+action*num_obs;
	 for(int i=0;i<num_obs;i++)
	 action_partition_counter[i]=0;
	 }
	 __syncthreads();*/

	/*Step the particles*/
	Dvc_State* current_particle = NULL;

	bool error = false;
	parent_PID = vnode_particleIDs[PID];

	/*make a local copy of the particle*/
	//int* localParticleMem=;
	//int localParticleMem[20];
	if (obs_i == 0) {
		DvcModelCopyNoAlloc_(
				(Dvc_State*) ((int*) localParticles + 20 * threadIdx.y),
				vnode_particles, PID % num_particles, false);
	}
	current_particle = (Dvc_State*) ((int*) localParticles + 20 * threadIdx.y);

	int terminal = false;
	OBS_TYPE obs = (OBS_TYPE) (-1);
	float reward = 0;
	//if(obs_i==0)
	//{
	/*step the local particle, get obs and reward*/
	if(parent_action>=0)
	{
		terminal = DvcModelStep_(*current_particle, streams->Entry(current_particle->scenario_id, streams->position_-1),
				parent_action, reward, obs);
		if (blockIdx.y * blockDim.y + threadIdx.y < num_particles) {
			/*Record stepped particles from parent as particles in this node*/
			if (obs_i == 0 && action==0) {
				Dvc_State* temp = DvcModelGet_(vnode_particles, PID % num_particles);
				DvcModelCopyNoAlloc_(temp, current_particle,0, false);
			}
		}
	}

	terminal = DvcModelStep_(*current_particle, streams->Entry(current_particle->scenario_id),
				action, reward, obs);
	if (abs(reward) < 0.1 && obs_i == 0 && terminal != true)
		error = true;
	//bool terminal=false; reward=-0.1; obs=0;
	reward = reward * current_particle->weight;
	//}

	if (blockIdx.y * blockDim.y + threadIdx.y < num_particles) {
		/*Record stepped particles*/
		int global_list_pos = action * total_num_scenarios + parent_PID;
		if (obs_i == 0) {
			//if(current_particle->weight!=0.002||global_scenario_pos>(action+1)*total_num_scenarios-1)
			//	error=true;
			Dvc_State* temp = DvcModelGet_(new_particles, global_list_pos);
			DvcModelCopyNoAlloc_(temp, current_particle, 0, false);

			//if(temp->weight!=0.002)
			//	error=true;

			/*Record all observations for CPU usage*/
			if (!terminal) {
				observations_all_a_p[global_list_pos] = obs;
			} else {
				observations_all_a_p[global_list_pos] = (OBS_TYPE) (-1);
			}
			if (obs < 0 || obs >= num_obs) {
				if (reward != 0 || terminal != true)
					error = true;			//error here
			} else {
				//atomicAdd(action_partition_counter+obs,1);//update obs group counter
			}

			/*Accumulate rewards of all particles from the v-node for CPU usage*/
			atomicAdd(step_reward_all_a + action, reward);
			//step_reward_all_a_p[global_scenario_pos]=reward;
		}
		if (obs_i == 0)
			terminal_all_a_p[global_list_pos] = terminal;
	}

}

__global__ void
//__launch_bounds__(64, 16)
Step_1(int total_num_scenarios, int num_particles, Dvc_State* vnode_particles,
		const int* vnode_particleIDs, float* step_reward_all_a,
		//float* step_reward_all_a_p,
		OBS_TYPE* observations_all_a_p, Dvc_State* new_particles,
		Dvc_RandomStreams* streams, int num_obs, bool* terminal_all_a_p
		, int parent_action) {

	if (blockIdx.y * blockDim.x + threadIdx.x < num_particles) {

		int action = blockIdx.x;
		int PID = (blockIdx.y * blockDim.x + threadIdx.x) % num_particles;
		int obs_i = threadIdx.y;
		int parent_PID = -1;

		if (blockIdx.y == 0 && threadIdx.x == 0 && obs_i == 0)
			step_reward_all_a[action] = 0;

		/*Step the particles*/
		Dvc_State* current_particle = NULL;

		bool error = false;
		parent_PID = vnode_particleIDs[PID];

		/*make a local copy of the particle in shared memory*/
		if (obs_i == 0) {
			DvcModelCopyToShared_(
					(Dvc_State*) ((int*) localParticles + 60 * threadIdx.x),
					vnode_particles, PID % num_particles, false);
		}
		current_particle = (Dvc_State*) ((int*) localParticles + 60 * threadIdx.x);

		int terminal = false;
		OBS_TYPE obs = (OBS_TYPE) (-1);
		float reward = 0;

		/*step the local particle, get obs and reward*/

		if(parent_action>=0)
		{
			terminal = DvcModelStep_(*current_particle, streams->Entry(current_particle->scenario_id, streams->position_-1),
					parent_action, reward, obs);

			if (blockIdx.y * blockDim.y + threadIdx.y < num_particles) {
				/*Record stepped particles from parent as particles in this node*/
				if (obs_i == 0 && action==0) {
					Dvc_State* temp = DvcModelGet_(vnode_particles, PID % num_particles);
					DvcModelCopyNoAlloc_(temp, current_particle,0, false);
				}
			}
		}

		terminal = DvcModelStep_(*current_particle, streams->Entry(current_particle->scenario_id),
				action, reward, obs);

		if (abs(reward) < 0.1 && obs_i == 0 && terminal != true)
			error = true;
		//bool terminal=false; reward=-0.1; obs=0;
		reward = reward * current_particle->weight;


		/*Record stepped particles*/
		int global_list_pos = action * total_num_scenarios + parent_PID;
		if (obs_i == 0) {
			//if(current_particle->weight!=0.002||global_scenario_pos>(action+1)*total_num_scenarios-1)
			//	error=true;
			Dvc_State* temp = DvcModelGet_(new_particles, global_list_pos);
			DvcModelCopyNoAlloc_(temp, current_particle, 0, false);

			//if(temp->weight!=0.002)
			//	error=true;

			/*Record all observations for CPU usage*/
			if (!terminal) {
				observations_all_a_p[global_list_pos] = obs;
			} else {
				observations_all_a_p[global_list_pos] = (OBS_TYPE) (-1);
			}
			if (obs < 0 || obs >= num_obs) {
				if (reward != 0 || terminal != true)
					error = true;			//error here
			} else {
				//atomicAdd(action_partition_counter+obs,1);//update obs group counter
			}

			/*Accumulate rewards of all particles from the v-node for CPU usage*/
			atomicAdd(step_reward_all_a + action, reward);
			//step_reward_all_a_p[global_scenario_pos]=reward;
		}
		if (obs_i == 0)
			terminal_all_a_p[global_list_pos] = terminal;
	}
}



__global__ void
//__launch_bounds__(64, 16)
Step_IntObs(int total_num_scenarios, int num_particles, Dvc_State* vnode_particles,
		const int* vnode_particleIDs, float* step_reward_all_a,
		//float* step_reward_all_a_p,
		int* observations_all_a_p,const int num_obs_elements,
		Dvc_State* new_particles,
		Dvc_RandomStreams* streams, bool* terminal_all_a_p
		, int parent_action,
		int Shared_mem_per_particle) {

	//if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
			printf("line %s\n", __LINE__);

	if (blockIdx.y * blockDim.x + threadIdx.x < num_particles) {
		__shared__ int Intobs[32*60];

		int action = blockIdx.x;
		int PID = (blockIdx.y * blockDim.x + threadIdx.x) % num_particles;
		int obs_i = threadIdx.y;
		int parent_PID = -1;

		if (blockIdx.y == 0 && threadIdx.x == 0 && obs_i == 0)
			step_reward_all_a[action] = 0;

		/*Step the particles*/
		Dvc_State* current_particle = NULL;

		bool error = false;
		parent_PID = vnode_particleIDs[PID];

		/*make a local copy of the particle in shared memory*/
		if (obs_i == 0) {
			DvcModelCopyToShared_(
					(Dvc_State*) ((int*) localParticles + Shared_mem_per_particle * threadIdx.x),
					vnode_particles, PID % num_particles, false);
			/*if(GPUDoPrint)current_particle = (Dvc_State*) ((int*) localParticles + Shared_mem_per_particle * threadIdx.x);
			if(GPUDoPrint && current_particle->scenario_id==GPUPrintPID && action==0){
				int sid=current_particle->scenario_id;
				printf("load vnode particle pid %d scenario %d\n",PID, sid);
			}*/
		}
		current_particle = (Dvc_State*) ((int*) localParticles + Shared_mem_per_particle * threadIdx.x);
		__syncthreads();

		if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
			printf("line 685\n");


		int terminal = false;
		float reward = 0;

		/*step the local particle, get obs and reward*/

		if(parent_action>=0)
		{
			if(DvcModelStepIntObs_)
			{

				terminal = DvcModelStepIntObs_(*current_particle, streams->Entry(current_particle->scenario_id, streams->position_-1),
							parent_action, reward, Intobs+threadIdx.x*num_obs_elements);
				if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
				printf("line 701\n");
			}
			else
			{
				printf("Undefined DvcModelStepIntObs_!\n");
			}
			__syncthreads();

			if (blockIdx.y * blockDim.y + threadIdx.y < num_particles) {
				/*Record stepped particles from parent as particles in this node*/
				if (obs_i == 0 && action==0) {
					/*if( current_particle->scenario_id==226){
						printf("vnode_particle address=%#0x \n", vnode_particles);
					}*/

					Dvc_State* temp = DvcModelGet_(vnode_particles, PID % num_particles);
					DvcModelCopyNoAlloc_(temp, current_particle,0, false);

					/*if(GPUDoPrint && current_particle->scenario_id==GPUPrintPID){
						int sid=current_particle->scenario_id;
						Dvc_State* address=temp;
						int pid=PID;
						printf("save vnode particle %d ",address);
						printf("in pid %d scenario %d\n", pid, sid);
					}*/
				}
			}
			__syncthreads();

		}

		if(DvcModelStepIntObs_)
		{
			terminal = DvcModelStepIntObs_(*current_particle, streams->Entry(current_particle->scenario_id),
					action, reward, Intobs+threadIdx.x*num_obs_elements);

			if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
				printf("line %s\n", __LINE__);
			/*if(current_particle->scenario_id==0 && blockIdx.x+threadIdx.y==0)
				printf("\nFinish step with action %d, get reward %f obs\n",action, reward);*/

		}
		else
		{
			printf("Undefined DvcModelStepIntObs_!\n");
		}

		if (abs(reward) < 0.1 && obs_i == 0 && terminal != true)
			error = true;
		//bool terminal=false; reward=-0.1; obs=0;
		reward = reward * current_particle->weight;


		/*Record stepped particles*/
		int global_list_pos = action * total_num_scenarios + parent_PID;


		if (obs_i == 0) {
			/*if(GPUDoPrint && current_particle->scenario_id==GPUPrintPID && action==0){
				printf("check global_pos %d action %d parent PID %d num_scena %d\n",
						global_list_pos,action,parent_PID,total_num_scenarios);
			}*/
			//if(current_particle->weight!=0.002||global_scenario_pos>(action+1)*total_num_scenarios-1)
			//	error=true;
			Dvc_State* temp = DvcModelGet_(new_particles, global_list_pos);
			DvcModelCopyNoAlloc_(temp, current_particle, 0, false);
			/*if(GPUDoPrint && current_particle->scenario_id==GPUPrintPID && action==0){
				printf("[2] save new particle %d ", temp);
				printf("for global_pos %d\n",global_list_pos);
			}*/
			//if(temp->weight!=0.002)
			//	error=true;
			/*Record all observations for CPU usage*/
			if (!terminal) {
				for(int i=0;i<num_obs_elements;i++)
					observations_all_a_p[global_list_pos*num_obs_elements+i] = Intobs[threadIdx.x*num_obs_elements+i];
			} else {
				observations_all_a_p[global_list_pos*num_obs_elements] = 0;//no content in obs list
			}

			/*Accumulate rewards of all particles from the v-node for CPU usage*/
			atomicAdd(step_reward_all_a + action, reward);
			//step_reward_all_a_p[global_scenario_pos]=reward;

			if (obs_i == 0)
				terminal_all_a_p[global_list_pos] = terminal;
			//if(threadIdx.x+threadIdx.y+blockIdx.x+blockIdx.y==0)
			//	streams->position_++;
		}
	}
}
__global__ void
//__launch_bounds__(64, 16)
_InitBounds(int total_num_scenarios, int num_particles,
		Dvc_State* new_particles, const int* vnode_particleIDs,
		float* upper_all_a_p, float* utility_upper_all_a_p,
		Dvc_ValuedAction* default_move_all_a_p, Dvc_RandomStreams* streams,
		int depth, int hist_size) {
	int action = blockIdx.x;
	int PID = (blockIdx.y * blockDim.y + threadIdx.y) % num_particles;
	int obs_i = threadIdx.x;
	//int offset=blockDim.x*action;
	//int pos=TID+offset;
	int parent_PID = -1;
	Dvc_State* current_particle = NULL;
	//if(PID<num_particles)
	//{
	parent_PID = vnode_particleIDs[PID];
	current_particle = (Dvc_State*) ((int*) localParticles + 20 * threadIdx.y);

	int global_list_pos = action * total_num_scenarios + parent_PID;

	if (obs_i == 0) {
		Dvc_State* temp = DvcModelGet_(new_particles, global_list_pos);
		DvcModelCopyNoAlloc_(current_particle, temp, 0, false);
	}

	/*Do roll-out using the updated particle*/
	//local_history[scenarioID].Add(action, observations[scenarioID]);
	Dvc_History local_history;
	local_history.currentSize_ = hist_size;
	local_history.actions_ = NULL;
	local_history.observations_ = NULL;
	Dvc_RandomStreams local_streams(streams->num_streams_, streams->length_,
			streams->streams_,
			(hist_size>0)?streams->position_+1:streams->position_);

	//Dvc_ValuedAction local_lower(0,-1);
	/*Upper Bound*/
	float local_upper;
	if (obs_i == 0 && (blockIdx.y * blockDim.y + threadIdx.y) < num_particles) {
		//double local_upper = 10;
		//local_upper = GPU_InitUpperBound(0/*already at right pos*/,
		//		current_particle,
		//		/*local_streams, */local_history, depth);
		local_upper = DvcUpperBoundValue_(current_particle, 0, local_history);
		local_upper *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
	}
	//Dvc_ValuedAction local_lower = GPU_InitLowerBound(
	//		current_particle,
	//		local_streams, local_history, depth);
	/*Lower bound*/
	local_streams.position(depth);
	Dvc_ValuedAction local_lower = DvcLowerBoundValue_( current_particle, local_streams,
				local_history);
	local_lower.value *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
	local_streams.position(depth);

	//local_lower.action=0;local_lower.value=-1;//DEBUG: test baseline
	if (obs_i == 0 && (blockIdx.y * blockDim.y + threadIdx.y) < num_particles) {
		//double local_upper = 10;
		//float local_upper = GPU_InitUpperBound(0/*already at right pos*/,current_particle,
		//		/*local_streams, */local_history, depth);
		global_list_pos=action * total_num_scenarios + PID;

		local_lower.value = local_lower.value * current_particle->weight;
		local_upper = local_upper * current_particle->weight;
		//local_upper=10;//DEBUG: test baseline
		utility_upper_all_a_p[global_list_pos] = local_upper;

		//__syncthreads();
		/*local_upper = local_upper
				- Dvc_config->pruning_constant / num_particles;
		if (local_upper < local_lower.value
		// close gap because no more search can be done on leaf node
				|| depth == Dvc_config->search_depth - 1) {
			local_upper = local_lower.value;
		}*/
		upper_all_a_p[global_list_pos] = local_upper;
		default_move_all_a_p[global_list_pos] = local_lower;
		//local_history[scenarioID].RemoveLast();
	}
	//}
	//__syncthreads();
}

__global__ void
//__launch_bounds__(64, 16)
_InitBounds1(int total_num_scenarios, int num_particles,
		Dvc_State* new_particles, const int* vnode_particleIDs,
		float* upper_all_a_p, float* utility_upper_all_a_p,
		Dvc_ValuedAction* default_move_all_a_p, OBS_TYPE* observations_all_a_p,
		Dvc_RandomStreams* streams, Dvc_History* history, int depth,
		int hist_size) {

	int action = blockIdx.x;
	int PID = (blockIdx.y * blockDim.x + threadIdx.x) % num_particles;
	int obs_i = threadIdx.y;
	int parent_PID = -1;
	Dvc_State* current_particle = NULL;

	parent_PID = vnode_particleIDs[PID];
	current_particle = (Dvc_State*) ((int*) localParticles + 60 * threadIdx.x);

	int global_list_pos = action * total_num_scenarios + parent_PID;

	if (obs_i == 0) {
		Dvc_State* temp = DvcModelGet_(new_particles, global_list_pos);
		if(DvcModelCopyToShared_)
			DvcModelCopyToShared_(current_particle, temp, 0, false);
		else
			printf("InitBound kernel: DvcModelCopyToShared_ has not been defined!\n");
	}
	__syncthreads();

	/*Do roll-out using the updated particle*/
	//local_history[scenarioID].Add(action, observations[scenarioID]);
	Dvc_History local_history;
	local_history.currentSize_ = hist_size;
	local_history.actions_ = history->actions_;
	local_history.observations_ = history->observations_;
	if(hist_size>0)
	{
		local_history.actions_[hist_size - 1] = blockIdx.x;
		local_history.observations_[hist_size - 1] =
				observations_all_a_p[global_list_pos];
	}
	Dvc_RandomStreams local_streams(streams->num_streams_, streams->length_,
			streams->streams_,
			(hist_size>0)?streams->position_+1:streams->position_);

	float local_upper;
	if (obs_i == 0 && (blockIdx.y * blockDim.x + threadIdx.x) < num_particles) {
		//local_upper = GPU_InitUpperBound(0/*already at right pos*/,
		//		current_particle,
		//		/*local_streams, */local_history, depth);
		local_upper = DvcUpperBoundValue_(current_particle, 0, local_history);
		local_upper *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
	}
	//Dvc_ValuedAction local_lower = GPU_InitLowerBound(
	//		 current_particle,
	//		local_streams, local_history, depth);
	/*Lower bound*/
	local_streams.position(depth);
	Dvc_ValuedAction local_lower = DvcLowerBoundValue_( current_particle, local_streams,
				local_history);
	local_lower.value *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
	local_streams.position(depth);
	if (obs_i == 0 && (blockIdx.y * blockDim.x + threadIdx.x) < num_particles) {
		global_list_pos=action * total_num_scenarios + PID;
		local_lower.value = local_lower.value * current_particle->weight;
		local_upper = local_upper * current_particle->weight;
		utility_upper_all_a_p[global_list_pos] = local_upper;

		/*local_upper = local_upper
				- Dvc_config->pruning_constant / num_particles;
		if (local_upper < local_lower.value
		// close gap because no more search can be done on leaf node
				|| depth == Dvc_config->search_depth - 1) {
			local_upper = local_lower.value;
		}*/
		upper_all_a_p[global_list_pos] = local_upper;
		default_move_all_a_p[global_list_pos] = local_lower;
		//local_history[scenarioID].RemoveLast();
	}
}


__global__ void
//__launch_bounds__(64, 16)
_InitBounds_IntObs(int total_num_scenarios, int num_particles,
		Dvc_State* new_particles, const int* vnode_particleIDs,
		float* upper_all_a_p, float* utility_upper_all_a_p,
		Dvc_ValuedAction* default_move_all_a_p,
		Dvc_RandomStreams* streams, Dvc_History* history, int depth,
		int hist_size,int Shared_mem_per_particle) {
	if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
		printf("line %s\n", __LINE__);
	int action = blockIdx.x;

	/*if (blockIdx.y * blockDim.x + threadIdx.x < num_particles) {
		if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
			printf("line %s\n", __LINE__);

		int PID = (blockIdx.y * blockDim.x + threadIdx.x) % num_particles;
		int obs_i = threadIdx.y;
		int parent_PID = -1;
		Dvc_State* current_particle = NULL;

		parent_PID = vnode_particleIDs[PID];
		current_particle = (Dvc_State*) ((int*) localParticles + Shared_mem_per_particle * threadIdx.x);

		int global_list_pos = action * total_num_scenarios + parent_PID;

		if (obs_i == 0) {
			Dvc_State* temp = DvcModelGet_(new_particles, global_list_pos);
			if(DvcModelCopyToShared_)
				DvcModelCopyToShared_(current_particle, temp, 0, false);
			else
				printf("InitBound kernel: DvcModelCopyToShared_ has not been defined!\n");
		}
		__syncthreads();
		if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
			printf("line %s\n", __LINE__);

		//Do roll-out using the updated particle
		Dvc_History local_history;
		local_history.currentSize_ = hist_size;
		local_history.actions_ = history->actions_;
		local_history.observations_ = history->observations_;

		Dvc_RandomStreams local_streams(streams->num_streams_, streams->length_,
				streams->streams_,
				(hist_size>0)?streams->position_+1:streams->position_);

		float local_upper;
		if (obs_i == 0 && (blockIdx.y * blockDim.x + threadIdx.x) < num_particles) {

			local_upper = DvcUpperBoundValue_(current_particle, 0, local_history);
			local_upper *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
		}
		if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
			printf("line %s\n", __LINE__);

		//Lower bound
		local_streams.position(depth);
		Dvc_ValuedAction local_lower = DvcLowerBoundValue_( current_particle, local_streams,
					local_history);
		local_lower.value *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
		local_streams.position(depth);
		if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
			printf("line %s\n", __LINE__);
		if (obs_i == 0 && (blockIdx.y * blockDim.x + threadIdx.x) < num_particles) {
			global_list_pos=action * total_num_scenarios + PID;
			local_lower.value = local_lower.value * current_particle->weight;
			local_upper = local_upper * current_particle->weight;
			utility_upper_all_a_p[global_list_pos] = local_upper;

			upper_all_a_p[global_list_pos] = local_upper;
			default_move_all_a_p[global_list_pos] = local_lower;
			if(threadIdx.x+threadIdx.y+blockIdx.y+blockIdx.x==0 )
				printf("line %s\n", __LINE__);
		}
	}*/
}
void DESPOT::Validate_GPU(int line) {
	if (false) {
		int ThreadID = 0;
		if (use_multi_thread_)
			ThreadID = MapThread(this_thread::get_id());
		HANDLE_ERROR(cudaDeviceSynchronize());
		cout << "Thread " << ThreadID << "@" << line << ": "
				<< Dvc_streams[ThreadID]->position_ << endl;
	}
}

float DESPOT::CalExplorationValue(int depth) {
	//return 0.01 * Globals::config.exploration_constant * pow(sqrt(1.2), max(depth, 5)/3);
	return /*2*Globals::config.exploration_constant*//*0.05*//*1*/1*Initial_root_gap/*0*/;
}
void DESPOT::CalExplorationValue(Shared_QNode* node) {
	if(Globals::config.exploration_constant>0)
	{
		node->exploration_bonus= Globals::config.exploration_constant *
			sqrt(log(static_cast<Shared_VNode*>(((QNode*)node)->parent())->visit_count_*
					max(((QNode*)node)->parent()->Weight()*
					Globals::config.num_scenarios,1.1))
			/(node->visit_count_*max(((QNode*)node)->Weight()*
					Globals::config.num_scenarios,1.1)));

		node->exploration_bonus*=((QNode*)node)->Weight();
		/*Global_print_value(this_thread::get_id(),
				((QNode*)node)->Weight()*Globals::config.num_scenarios, "QNode num of particles");
		Global_print_value(this_thread::get_id(),
							static_cast<Shared_VNode*>(((QNode*)node)->parent())->visit_count_
								, "QNode parent visit count");
		Global_print_value(this_thread::get_id(),
							node->visit_count_, "QNode visit count");
		Global_print_value(this_thread::get_id(),
								node->exploration_bonus/node->Weight(), "QNode explortion bonus");*/
	}
}
void DESPOT::CalExplorationValue(Shared_VNode* node) {
	node->exploration_bonus= Globals::config.exploration_constant *
		sqrt(log(static_cast<Shared_QNode*>(((VNode*)node)->parent())->visit_count_*
				max(((VNode*)node)->parent()->Weight()*
				Globals::config.num_scenarios,1.0))
		/(node->visit_count_*max(((VNode*)node)->Weight()*
				Globals::config.num_scenarios,1.0)));

	//Global_print_value(this_thread::get_id(),
	//		static_cast<Shared_QNode*>(((VNode*)node)->parent())->visit_count_
	//							, "VNode parent visit count");
	//Global_print_value(this_thread::get_id(),
	//					node->visit_count_, "VNode visit count");
}
/*__global__ void UpdateStream(Dvc_RandomStreams *streams, int pos) {
	streams->position_ = pos;
}*/
void DESPOT::DataReadBack(VNode* vnode,int ThreadID)
{
	const std::vector<State*>& particles = vnode->particles();

}

//#define RECORD_TIME 1
void DESPOT::UpdateData(VNode* vnode,int ThreadID,const DSPOMDP* model,RandomStreams& streams)
{
#ifdef RECORD_TIME
	auto start = Time::now();
#endif
	streams.position(vnode->depth());
	//Copy streams to GPU
	/*if (use_multi_thread_) {
		UpdateStream<<<dim3(1, 1), dim3(1, 1), 0, cuda_streams[ThreadID]>>>(
				Dvc_streams[ThreadID], streams.position());
		//HANDLE_ERROR(cudaStreamSynchronize(cuda_streams[ThreadID]));//Managed memory cannot be used when kernel os running
	} else
		Dvc_streams[ThreadID]->position_ = streams.position();*/

	const std::vector<State*>& particles = vnode->particles();
	const std::vector<int>& particleIDs = vnode->particleIDs();
	if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
	model->CopyToGPU(particleIDs, Dvc_particleIDs_long[ThreadID],
			cuda_streams + ThreadID);
	if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
	if(vnode->parent()!=NULL)
	{
		/*Create GPU particles for the new v-node*/
		Dvc_State* new_particles = model->AllocGPUParticles(
				particleIDs.size(), 2);
	if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
		//model->CopyToGPU(particleIDs, Dvc_particleIDs_long[ThreadID],
		//		cuda_streams + ThreadID);

		/*Copy stepped particles to new memory*/

		/*if(FIX_SCENARIO==1)
			cout<<"Reset stream pos to vnode->depth="<<vnode->depth()<<endl;*/
		model->CopyGPUParticles(new_particles,
				vnode->parent()->parent()->GetGPUparticles(),
				0, Dvc_particleIDs_long[ThreadID],
				particleIDs.size(),true,
				Dvc_streams[ThreadID], streams.position(),
				cuda_streams + ThreadID);
		if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
		/*Interleaving with CPU*/
		vnode->AssignGPUparticles(new_particles,
				particleIDs.size());
		if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
		vnode->weight_=particleIDs.size()/((float)Globals::config.num_scenarios);
	}

#ifdef RECORD_TIME

	double oldValue=CopyParticleTime.load();
	CopyParticleTime.compare_exchange_weak(oldValue,oldValue+
							chrono::duration_cast < ns
					> (Time::now() - start).count()/1000000000.0f);
#endif
}

void DESPOT::MCSimulation(VNode* vnode, int ThreadID,
		const DSPOMDP* model, RandomStreams& streams,History& history, bool Do_rollout)
{
	/*if(FIX_SCENARIO==1 && vnode->edge()==11220829450167354192)
	{GPUDoPrint=true;GPUPrintPID=372;}*/
#ifdef RECORD_TIME
	auto start = Time::now();
#endif
	if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<endl;

	int thready, blocky;
	dim3 GridDim;
	dim3 ThreadDim;
	int NumActions = model->NumActions();
	int NumObs = model->NumObservations();
	int NumScenarios = Globals::config.num_scenarios;
	int NumParticles=vnode->num_GPU_particles_;

	int ParalllelisminStep = model->ParallelismInStep();
	int Shared_mem_per_particle=CalSharedMemSize();

	if (Obs_parallel_level == OBS_PARALLEL_X) {

		thready =
				(MC_DIM % ParalllelisminStep == 0) ?
						MC_DIM / ParalllelisminStep : MC_DIM / ParalllelisminStep + 1;
		blocky =
				(NumParticles % thready == 0) ?
						NumParticles / thready : NumParticles / thready + 1;
		GridDim.x = NumActions;
		GridDim.y = blocky;
		ThreadDim.x = ParalllelisminStep;
		ThreadDim.y = thready;
	}
	if(use_multi_thread_)
		{
	#ifdef RECORD_TIME
			start = Time::now();
	#endif

			//static_cast<Shared_VNode*>(vnode)->is_waiting_=true;
			;//thread_barrier->count_down_and_wait();//Make threads wait for each other
			//static_cast<Shared_VNode*>(vnode)->is_waiting_=false;
	#ifdef RECORD_TIME
			oldValue=BarrierWaitTime.load();
			BarrierWaitTime.compare_exchange_weak(oldValue,oldValue+
							chrono::duration_cast < ns
							> (Time::now() - start).count()/1000000000.0f);
	#endif
		}
	int threadx = 32;
	if (Obs_parallel_level == OBS_PARALLEL_Y) {
		blocky =
				(NumParticles % threadx == 0) ?
						NumParticles / threadx : NumParticles / threadx + 1;
		GridDim.x = NumActions;
		GridDim.y = blocky;
		ThreadDim.x = threadx;
		ThreadDim.y = model->ParallelismInStep();

		if(Obs_type==OBS_INT_ARRAY/*false*/)
		{
			if(GPUDoPrint){
				printf("pre-step particle %d\n", Dvc_particles_long[ThreadID]);
			}

			int num_Obs_element=num_Obs_element_in_GPU;
			if (use_multi_thread_)
				Step_IntObs<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int),
						cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
						NumParticles, Dvc_particles_long[ThreadID],
						Dvc_particleIDs_long[ThreadID], Dvc_r_all_a[ThreadID],
						//Dvc_r_all_a_p,
						Dvc_obs_int_all_a_and_p[ThreadID],num_Obs_element,
						Dvc_stepped_particles_all_a[ThreadID],
						Dvc_streams[ThreadID],
						Dvc_term_all_a_and_p[ThreadID],
						(vnode->parent()==NULL)?-1:vnode->parent()->edge(),
						Shared_mem_per_particle);
			else
				Step_IntObs<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int)>>>(
						Globals::config.num_scenarios, NumParticles,
						Dvc_particles_long[ThreadID],
						Dvc_particleIDs_long[ThreadID], Dvc_r_all_a[ThreadID],
						//Dvc_r_all_a_p,
						Dvc_obs_int_all_a_and_p[ThreadID],num_Obs_element,
						Dvc_stepped_particles_all_a[ThreadID],
						Dvc_streams[ThreadID],
						Dvc_term_all_a_and_p[ThreadID],
						(vnode->parent()==NULL)?-1:vnode->parent()->edge(),
						Shared_mem_per_particle);
		}
		else
		{
			if (use_multi_thread_)
				Step_1<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int),
						cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
						NumParticles, Dvc_particles_long[ThreadID],
						Dvc_particleIDs_long[ThreadID], Dvc_r_all_a[ThreadID],
						//Dvc_r_all_a_p,
						Dvc_obs_all_a_and_p[ThreadID],
						Dvc_stepped_particles_all_a[ThreadID],
						Dvc_streams[ThreadID], NumObs,
						Dvc_term_all_a_and_p[ThreadID],
						(vnode->parent()==NULL)?-1:vnode->parent()->edge());
			else
				Step_1<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int)>>>(
						Globals::config.num_scenarios, NumParticles,
						Dvc_particles_long[ThreadID],
						Dvc_particleIDs_long[ThreadID], Dvc_r_all_a[ThreadID],
						//Dvc_r_all_a_p,
						Dvc_obs_all_a_and_p[ThreadID],
						Dvc_stepped_particles_all_a[ThreadID],
						Dvc_streams[ThreadID], NumObs,
						Dvc_term_all_a_and_p[ThreadID],
						(vnode->parent()==NULL)?-1:vnode->parent()->edge());
		}
	} else {
		if (use_multi_thread_)
			Step_<<<GridDim, ThreadDim, thready * Shared_mem_per_particle * sizeof(int),
					cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
					NumParticles, Dvc_particles_long[ThreadID],
					Dvc_particleIDs_long[ThreadID], Dvc_r_all_a[ThreadID],
					//Dvc_r_all_a_p,
					Dvc_obs_all_a_and_p[ThreadID],
					Dvc_stepped_particles_all_a[ThreadID],
					Dvc_streams[ThreadID], NumObs,
					Dvc_term_all_a_and_p[ThreadID],
					(vnode->parent()==NULL)?-1:vnode->parent()->edge());
		else
			Step_<<<GridDim, ThreadDim, thready * Shared_mem_per_particle * sizeof(int)>>>(
					Globals::config.num_scenarios, NumParticles,
					Dvc_particles_long[ThreadID],
					Dvc_particleIDs_long[ThreadID], Dvc_r_all_a[ThreadID],
					//Dvc_r_all_a_p,
					Dvc_obs_all_a_and_p[ThreadID],
					Dvc_stepped_particles_all_a[ThreadID],
					Dvc_streams[ThreadID], NumObs,
					Dvc_term_all_a_and_p[ThreadID],
					(vnode->parent()==NULL)?-1:vnode->parent()->edge());
	}

	/*if (use_multi_thread_)
		HANDLE_ERROR(cudaStreamSynchronize(cuda_streams[ThreadID]));
	else
		HANDLE_ERROR(cudaDeviceSynchronize());*/
	//AveRewardTime += chrono::duration_cast < sec
	//		> (Time::now() - start).count();
#ifdef RECORD_TIME
	double oldValue=AveRewardTime.load();
	AveRewardTime.compare_exchange_weak(oldValue,oldValue+
					chrono::duration_cast < ns
					> (Time::now() - start).count()/1000000000.0f);
#endif

	if(Do_rollout)
	{
		//Global_print_message(this_thread::get_id(),"After step");
		if(use_multi_thread_)
		{
	#ifdef RECORD_TIME
			start = Time::now();
	#endif

			//static_cast<Shared_VNode*>(vnode)->is_waiting_=true;
			;//thread_barrier->count_down_and_wait();//Make threads wait for each other
			//static_cast<Shared_VNode*>(vnode)->is_waiting_=false;
	#ifdef RECORD_TIME
			oldValue=BarrierWaitTime.load();
			BarrierWaitTime.compare_exchange_weak(oldValue,oldValue+
							chrono::duration_cast < ns
							> (Time::now() - start).count()/1000000000.0f);
	#endif
		}

	#ifdef RECORD_TIME
		start = Time::now();
	#endif

		if (Obs_parallel_level == OBS_PARALLEL_Y) {

			if(Obs_type==OBS_INT_ARRAY)
			{
				if (use_multi_thread_)
					_InitBounds_IntObs<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int),
							cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
							NumParticles, Dvc_stepped_particles_all_a[ThreadID],
							Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
							Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
							 Dvc_streams[ThreadID],
							Dvc_history[ThreadID], vnode->depth() + 1,
							history.Size() + 1,Shared_mem_per_particle);
				else
					_InitBounds_IntObs<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int)>>>(
							Globals::config.num_scenarios, NumParticles,
							Dvc_stepped_particles_all_a[ThreadID],
							Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
							Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
							Dvc_streams[ThreadID],
							Dvc_history[ThreadID], vnode->depth() + 1,
							history.Size() + 1,Shared_mem_per_particle);
			}
			else
			{
				if (use_multi_thread_)
					_InitBounds1<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int),
							cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
							NumParticles, Dvc_stepped_particles_all_a[ThreadID],
							Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
							Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
							Dvc_obs_all_a_and_p[ThreadID], Dvc_streams[ThreadID],
							Dvc_history[ThreadID], vnode->depth() + 1,
							history.Size() + 1);
				else
					_InitBounds1<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int)>>>(
							Globals::config.num_scenarios, NumParticles,
							Dvc_stepped_particles_all_a[ThreadID],
							Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
							Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
							Dvc_obs_all_a_and_p[ThreadID], Dvc_streams[ThreadID],
							Dvc_history[ThreadID], vnode->depth() + 1,
							history.Size() + 1);
			}

		} else {
			if (use_multi_thread_)
				_InitBounds<<<GridDim, ThreadDim, thready * Shared_mem_per_particle * sizeof(int),
						cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
						NumParticles, Dvc_stepped_particles_all_a[ThreadID],
						Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
						Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
						Dvc_streams[ThreadID], vnode->depth() + 1,
						history.Size() + 1);
			else
				_InitBounds<<<GridDim, ThreadDim, thready * Shared_mem_per_particle * sizeof(int)>>>(
						Globals::config.num_scenarios, NumParticles,
						Dvc_stepped_particles_all_a[ThreadID],
						Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
						Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
						Dvc_streams[ThreadID], vnode->depth() + 1,
						history.Size() + 1);
		}

	#ifdef RECORD_TIME
		oldValue=InitBoundTime.load();
		InitBoundTime.compare_exchange_weak(oldValue,oldValue+
				chrono::duration_cast < ns
				> (Time::now() - start).count()/1000000000.0f);
		start = Time::now();
	#endif

		ReadBackData(ThreadID);
	#ifdef RECORD_TIME
		oldValue=DataBackTime.load();
		DataBackTime.compare_exchange_weak(oldValue,oldValue+
					chrono::duration_cast < ns
					> (Time::now() - start).count()/1000000000.0f);
	#endif
	}

	/*if(FIX_SCENARIO==1 && vnode->edge()==11220829450167354192)
	{
		HANDLE_ERROR(cudaDeviceSynchronize());
		GPUDoPrint=false;
	}*/
}

void DESPOT::GPU_Expand_Action(VNode* vnode, ScenarioLowerBound* lb,
		ScenarioUpperBound* ub, const DSPOMDP* model, RandomStreams& streams,
		History& history) {
	//if(FIX_SCENARIO==1) cout<<__FUNCTION__<<endl;

	int ThreadID = 0;
	if (use_multi_thread_)
		ThreadID = MapThread(this_thread::get_id());
	int NumActions = model->NumActions();
	int NumObs = model->NumObservations();
	int NumScenarios = Globals::config.num_scenarios;

	Global_print_expand(this_thread::get_id(), vnode, vnode->depth(), vnode->edge());

	if(use_multi_thread_)
		static_cast<Shared_VNode*>(vnode)->is_waiting_=true;

	HitCount++;
	AddExpanded();
	auto start_total = Time::now();

	/*Update streams, history, and particles into GPU*/
	UpdateData(vnode, ThreadID,model, streams);

	Dvc_particles_long[ThreadID] = vnode->GetGPUparticles();//get the GPU particles of the parent v-node



	if(use_multi_thread_)
	{
#ifdef RECORD_TIME
		auto start = Time::now();
#endif
		;//thread_barrier->count_down_and_wait();//Make threads wait for each other
#ifdef RECORD_TIME
		double oldValue=BarrierWaitTime.load();
		BarrierWaitTime.compare_exchange_weak(oldValue,oldValue+
				chrono::duration_cast < ns
				> (Time::now() - start).count()/1000000000.0f);
#endif
	}

	int NumParticles = vnode->particleIDs().size();

	AveNumParticles = AveNumParticles * (HitCount - 1) / HitCount
			+ NumActions * NumParticles / HitCount;

	/*Run Monte Carlo simulations in GPU: update particles and perform rollouts*/
	MCSimulation(vnode, ThreadID,model, streams,history,true);


	/*Debug lb*/
	if(false)
	{
		for(int i=0;i<NumParticles;i++)
		{
			cout.precision(3);
			cout<<Hst_lb_all_a_p[ThreadID][i].action<<"/"<<Hst_lb_all_a_p[ThreadID][i].value<<" ";
			cout<<Hst_ub_all_a_p[ThreadID][i]<<" ";
		}
		cout<<endl;
	}
	/*Debug lb*/

	std::vector<int> particleIDs=vnode->particleIDs();
	/*Expand v-node*/
	for (int action = 0; action < NumActions; action++) {
		/*Partition particles by observation*/
#ifdef RECORD_TIME
		auto start = Time::now();
#endif
		std::map<OBS_TYPE, std::vector<State*> > partitions;
		std::map<OBS_TYPE, std::vector<int> > partitions_ID;
		for (int i = 0; i < NumParticles; i++) {
			int parent_PID = particleIDs[i];
			OBS_TYPE obs;

			if(Obs_type==OBS_INT_ARRAY)
			{
				std::vector<int> tempobs;
				int* Int_obs_list = &Hst_obs_int_all_a_and_p[ThreadID][(action * NumScenarios
									+ parent_PID)*num_Obs_element_in_GPU];
				int num_obs_elements=Int_obs_list[0];
				tempobs.resize(num_obs_elements);
				//cout<<"obs ("<< num_obs_elements << " elements)"<<endl;
				for(int i=0;i<num_obs_elements;i++)
				{
					tempobs[i]=Int_obs_list[i+1];
					//cout<<tempobs[i]<<" ";
				}
				//cout<<endl;
				std::hash<std::vector<int>> myhash;
				obs=myhash(tempobs);
			}
			else
			{
				obs = Hst_obs_all_a_and_p[ThreadID][action * NumScenarios
					+ parent_PID];
			}
			/*Debug lb*/
			if(false){
				if(i</*NumParticles-*/20) cout<<"("<<parent_PID<<","<<obs<<") ";
			}
			/*Debug lb*/

			if (/*obs >= 0 && obs < NumObs
					&&*/ Hst_term_all_a_and_p[ThreadID][action * NumScenarios
							+ parent_PID] == false) {
				partitions[obs].push_back(NULL);
				partitions_ID[obs].push_back(/*scenarioID*/i);
			}
		}
		/*Debug lb*/
		//cout<<endl;
		/*Debug lb*/
#ifdef RECORD_TIME
		double oldValue=MakePartitionTime.load();
		MakePartitionTime.compare_exchange_weak(oldValue,oldValue+
					chrono::duration_cast < ns
					> (Time::now() - start).count()/1000000000.0f);
		/*Create new v-nodes for partitions, calculate the bounds*/
		auto nodestart = Time::now();
#endif

		float init_bound_hst_t = 0;
		QNode* qnode = vnode->Child(action);

		if(use_multi_thread_ && Globals::config.exploration_mode==UCT)
			static_cast<Shared_QNode*>(qnode)->visit_count_=1.1;

		if (partitions.size() == 0 && false) {
			cout<<"[Qnode] depth="<<vnode->depth()+1<<" obs="<< vnode->edge()<<" qnode "<<action<<" all particle termination: reward="<<Hst_r_all_a[action];
			cout<<" parent lb:"<<qnode->parent()->lower_bound()<<endl;
		} else {
		}

		double lower_bound = 0, upper_bound = 0;
		Hst_r_all_a[ThreadID][action] = Globals::Discount(vnode->depth())
				* Hst_r_all_a[ThreadID][action]
				- Globals::config.pruning_constant; //pruning_constant is used for regularization
		lower_bound = (Hst_r_all_a[ThreadID][action]);
		upper_bound = (Hst_r_all_a[ThreadID][action]);

		bool DoPrint=true;
		if (FIX_SCENARIO == 1 && DoPrint) {
			cout.precision(10);
			if(action==0) cout<<endl;
			cout << "step reward (d= " << vnode->depth() + 1 << " ): "
					<< Hst_r_all_a[ThreadID][action] / (1.0f/Globals::config.num_scenarios * NumParticles)
					<< endl;
		}

		/*if (lower_bound > 100)
			cout << "[1] Invalid step value " << lower_bound << " for action "
					<< action << endl;*/

		std::map<OBS_TYPE, VNode*>& children = qnode->children();
		for (std::map<OBS_TYPE, std::vector<State*> >::iterator it =
				partitions.begin(); it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			logd << " Creating node for obs " << obs << endl;
			//cout << "partition with obs: "<<obs<<endl;

			VNode* child_vnode;

			if (use_multi_thread_)
			{
				child_vnode = new Shared_VNode(partitions[obs],
						partitions_ID[obs], vnode->depth() + 1,
						static_cast<Shared_QNode*>(qnode), obs);

				if(Globals::config.exploration_mode==UCT)
					static_cast<Shared_VNode*>(child_vnode)->visit_count_=1.1;
			}
			else
				child_vnode = new VNode(partitions[obs], partitions_ID[obs],
						vnode->depth() + 1, qnode, obs);
#ifdef RECORD_TIME
			start = Time::now();
#endif

			/*Create GPU particles for the new v-node*/
			/*Dvc_State* new_particles = model->AllocGPUParticles(
					partitions_ID[obs].size(), 2);

			model->CopyToGPU(partitions_ID[obs], Dvc_particleIDs_long[ThreadID],
					cuda_streams + ThreadID);*/

			/*Copy stepped particles to new memory*/

			/*model->CopyGPUParticles(new_particles,
					Dvc_stepped_particles_all_a[ThreadID],
					action * NumScenarios, Dvc_particleIDs_long[ThreadID],
					partitions_ID[obs].size(), true, cuda_streams + ThreadID);*/
			/*Interleaving with CPU*/
			/*child_vnode->AssignGPUparticles(new_particles,
					partitions_ID[obs].size());*/
			//model->CopyGPUWeight1(cuda_streams + ThreadID);
			child_vnode->weight_=partitions[obs].size()/((float)NumScenarios);

			logd << " New node created!" << endl;
			children[obs] = child_vnode;

			/*Calculate initial bounds*/
			double vnode_lower_bound = 0;
			double vnode_upper_bound = 0;
			double vnode_utility_upper = 0;

			for (int i = 0; i < child_vnode->particleIDs().size(); i++) {
				int parent_PID = child_vnode->particleIDs()[i];

				vnode_lower_bound += Hst_lb_all_a_p[ThreadID][action
						* NumScenarios + parent_PID].value;
				vnode_upper_bound += Hst_ub_all_a_p[ThreadID][action
						* NumScenarios + parent_PID];				//*weight;
				vnode_utility_upper += Hst_uub_all_a_p[ThreadID][action
						* NumScenarios + parent_PID];				//*weight;
			}

			child_vnode->lower_bound(vnode_lower_bound);
			child_vnode->upper_bound(vnode_upper_bound-Globals::config.pruning_constant);
			child_vnode->utility_upper_bound(vnode_utility_upper);
			int first_particle = action * NumScenarios
					+ child_vnode->particleIDs()[0];
			child_vnode->default_move(
					ValuedAction(
							Hst_lb_all_a_p[ThreadID][first_particle].action,
							vnode_lower_bound));
			logd << " New node's bounds: (" << child_vnode->lower_bound()
					<< ", " << child_vnode->upper_bound() << ")" << endl;

			if (child_vnode->upper_bound() < child_vnode->lower_bound()
			// close gap because no more search can be done on leaf node
					|| child_vnode->depth() == Globals::config.search_depth - 1) {
				child_vnode->upper_bound(child_vnode->lower_bound());
			}
			//child_vnode->weight_ = model->CopyGPUWeight2(
			//		cuda_streams + ThreadID);

#ifdef RECORD_TIME
			init_bound_hst_t += chrono::duration_cast < ns
								> (Time::now() - start).count()/1000000000.0f;
#endif

			if (FIX_SCENARIO == 1 && DoPrint) {
				cout.precision(10);
				cout << " [GPU Vnode] New node's bounds: (d= "
						<< child_vnode->depth() << " ,obs=" << obs << " , lb= "
						<< child_vnode->lower_bound() / child_vnode->weight_
						<< " ,ub= "
						<< child_vnode->upper_bound() / child_vnode->weight_
						<< " ,uub= "
						<< child_vnode->utility_upper_bound()
								/ child_vnode->weight_ << " ,weight= "
						<< child_vnode->weight_ << " )";
				if(child_vnode->Weight()==1.0/Globals::config.num_scenarios) cout<<", particle_id="<< child_vnode->particles()[0]->scenario_id;
					cout<<", WEU="<<WEU(child_vnode);
				cout  << endl;
				/*if(obs==(OBS_TYPE)6325482029477897646)
					cout<<"ScenarioID="<<child_vnode->particleIDs()[0]<<endl;*/
			}

			lower_bound += child_vnode->lower_bound();
			upper_bound += child_vnode->upper_bound();

		}
#ifdef RECORD_TIME
		//CopyParticleTime += init_bound_hst_t;
		oldValue=CopyParticleTime.load();
		CopyParticleTime.compare_exchange_weak(oldValue,oldValue+init_bound_hst_t);
#endif
		qnode->step_reward = Hst_r_all_a[ThreadID][action];

		qnode->lower_bound(lower_bound);
		qnode->upper_bound(upper_bound);
		qnode->utility_upper_bound(
				upper_bound + Globals::config.pruning_constant);
		qnode->default_value = lower_bound; // for debugging
		/*if(vnode->depth()==0)
			cout<<"call weight() from "<< __FUNCTION__<<endl;*/
		qnode->Weight();
		if (FIX_SCENARIO == 1 && DoPrint) {
			cout.precision(10);
			cout << " [GPU Qnode] New qnode's bounds: (d= " << vnode->depth() + 1
					<< " ,action=" << action << ", lb= "
					<< qnode->lower_bound() / qnode->Weight() << " ,ub= "
					<< qnode->upper_bound() / qnode->Weight() << " ,uub= "
					<< qnode->utility_upper_bound() / qnode->Weight()
					<< " ,weight= " << qnode->Weight() << " )" << endl;
		}

		//MakeObsNodeTime += chrono::duration_cast < sec
		//		> (Time::now() - nodestart).count() - init_bound_hst_t;
#ifdef RECORD_TIME
		oldValue=MakeObsNodeTime.load();
		MakeObsNodeTime.compare_exchange_weak(oldValue,oldValue+
					chrono::duration_cast < ns
					> (Time::now() - nodestart).count()/1000000000.0f - init_bound_hst_t);
#endif
	}

	if(use_multi_thread_)
		static_cast<Shared_VNode*>(vnode)->is_waiting_=false;

	double oldValue=TotalExpansionTime.load();
	TotalExpansionTime.compare_exchange_weak(oldValue,oldValue+
						chrono::duration_cast < ns
						> (Time::now() - start_total).count()/1000000000.0f);
}

void DESPOT::ReadBackData(int ThreadID) {
	if (use_multi_thread_) {
		HANDLE_ERROR(
				cudaMemcpyAsync(Hst_MC_Data[ThreadID], Dvc_MC_Data[ThreadID],
						MC_DataSize, cudaMemcpyDeviceToHost,
						cuda_streams[ThreadID]));
		HANDLE_ERROR(cudaStreamSynchronize(cuda_streams[ThreadID]));
	} else {
		HANDLE_ERROR(
				cudaMemcpy(Hst_MC_Data[ThreadID], Dvc_MC_Data[ThreadID],
						MC_DataSize, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
}

int DESPOT::CalSharedMemSize() {
	int Shared_mem_per_particle;
	switch (Obs_parallel_level) {
	case OBS_PARALLEL_X:
		Shared_mem_per_particle = 20;
		break;
	case OBS_PARALLEL_Y:
		if (Obs_type == OBS_INT_ARRAY)
			Shared_mem_per_particle = 200;
		else
			Shared_mem_per_particle = 60;

		break;
	};
	return Shared_mem_per_particle;
}

void DESPOT::GPU_InitBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound,const DSPOMDP* model, RandomStreams& streams,
		History& history) {
	/*InitLowerBound(vnode, lower_bound, streams, history);
	InitUpperBound(vnode, upper_bound, streams, history);
	if (vnode->upper_bound() < vnode->lower_bound()
	// close gap because no more search can be done on leaf node
			|| vnode->depth() == Globals::config.search_depth - 1) {
		vnode->upper_bound(vnode->lower_bound());
	}*/
	if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
	int ThreadID = 0;
	//if (use_multi_thread_)
	//	ThreadID = MapThread(this_thread::get_id());
	UpdateData(vnode, ThreadID,model, streams);
	//Dvc_streams[ThreadID]->position_=vnode->depth();
	int thready, blocky;
	dim3 GridDim;
	dim3 ThreadDim;
	int NumScenarios = Globals::config.num_scenarios;
	int NumParticles=vnode->num_GPU_particles_;

	int ParalllelisminStep = model->ParallelismInStep();

	int Shared_mem_per_particle = CalSharedMemSize();
	if (Obs_parallel_level == OBS_PARALLEL_X) {

		thready =
				(MC_DIM % ParalllelisminStep == 0) ?
						MC_DIM / ParalllelisminStep : MC_DIM / ParalllelisminStep + 1;
		blocky =
				(NumParticles % thready == 0) ?
						NumParticles / thready : NumParticles / thready + 1;
		GridDim.x = 1;
		GridDim.y = blocky;
		ThreadDim.x = ParalllelisminStep;
		ThreadDim.y = thready;
	}

	int threadx = 32;
	if(true/*FIX_SCENARIO==1*/) cout<<__FUNCTION__<<": "<<__LINE__<<endl;
	if (Obs_parallel_level == OBS_PARALLEL_Y) {
		blocky =(NumParticles % threadx == 0) ?
				NumParticles / threadx : NumParticles / threadx + 1;
		GridDim.x = 1;
		GridDim.y = blocky;
		ThreadDim.x = threadx;
		ThreadDim.y = model->ParallelismInStep();

		if(Obs_type==OBS_INT_ARRAY)
		{
			if(true/*FIX_SCENARIO==1*/) {
				cout<<__FUNCTION__<<": "<<__LINE__<<endl;
				cout<<__FUNCTION__<<": ThreadID="<<ThreadID<<endl;
				cout<<__FUNCTION__<<": GridDim="<<GridDim.x<<","<<GridDim.y<<endl;
				cout<<__FUNCTION__<<": ThreadDim="<<ThreadDim.x<<","<<ThreadDim.y<<endl;
				cout<<__FUNCTION__<<": Globals::config.num_scenarios="<<Globals::config.num_scenarios<<endl;
				cout<<__FUNCTION__<<": NumParticles="<<NumParticles<<endl;
				cout<<__FUNCTION__<<": vnode->GetGPUparticles()="<<vnode->GetGPUparticles()<<endl;
				cout<<__FUNCTION__<<": Dvc_particleIDs_long[ThreadID]="<<Dvc_particleIDs_long[ThreadID]<<endl;
				cout<<__FUNCTION__<<": Dvc_ub_all_a_p[ThreadID]="<<Dvc_ub_all_a_p[ThreadID]<<endl;
				cout<<__FUNCTION__<<": Dvc_uub_all_a_p[ThreadID]="<<Dvc_uub_all_a_p[ThreadID]<<endl;
				cout<<__FUNCTION__<<": Dvc_lb_all_a_p[ThreadID]="<<Dvc_lb_all_a_p[ThreadID]<<endl;
				cout<<__FUNCTION__<<": Dvc_streams[ThreadID]="<<Dvc_streams[ThreadID]<<endl;
				cout<<__FUNCTION__<<": Dvc_streams[ThreadID]->position_="<<Dvc_streams[ThreadID]->position_<<endl;
				cout<<__FUNCTION__<<": Dvc_streams[ThreadID]->streams_="<<Dvc_streams[ThreadID]->streams_<<endl;
				cout<<__FUNCTION__<<": Dvc_history[ThreadID]="<<Dvc_history[ThreadID]<<endl;
				cout<<__FUNCTION__<<": Dvc_history[ThreadID]->actions_="<<Dvc_history[ThreadID]->actions_<<endl;

				cout<<__FUNCTION__<<": vnode->depth()="<<vnode->depth()<<endl;
				cout<<__FUNCTION__<<": history.Size()="<<history.Size()<<endl;
				cout<<__FUNCTION__<<": Shared_mem_per_particle="<<Shared_mem_per_particle<<endl;
				cout<<__FUNCTION__<<": use_multi_thread_="<<use_multi_thread_<<endl;
				
			}
			if (use_multi_thread_)
				_InitBounds_IntObs<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int),
						cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
						NumParticles, vnode->GetGPUparticles(),
						Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
						Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
						 Dvc_streams[ThreadID],
						Dvc_history[ThreadID], vnode->depth(),
						history.Size(),Shared_mem_per_particle);
			else
				_InitBounds_IntObs<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int)>>>(
						Globals::config.num_scenarios, NumParticles,
						vnode->GetGPUparticles(),
						Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
						Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
						Dvc_streams[ThreadID],
						Dvc_history[ThreadID], vnode->depth(),
						history.Size(),Shared_mem_per_particle);
		}
		else
		{
			if (use_multi_thread_)
				_InitBounds1<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int),
						cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
						NumParticles, vnode->GetGPUparticles(),
						Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
						Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
						Dvc_obs_all_a_and_p[ThreadID], Dvc_streams[ThreadID],
						Dvc_history[ThreadID], vnode->depth(),
						history.Size());
			else
				_InitBounds1<<<GridDim, ThreadDim, threadx * Shared_mem_per_particle * sizeof(int)>>>(
						Globals::config.num_scenarios, NumParticles,
						vnode->GetGPUparticles(),
						Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
						Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
						Dvc_obs_all_a_and_p[ThreadID], Dvc_streams[ThreadID],
						Dvc_history[ThreadID], vnode->depth(),
						history.Size());
		}
	} else {
		if (use_multi_thread_)
			_InitBounds<<<GridDim, ThreadDim, thready * Shared_mem_per_particle * sizeof(int),
					cuda_streams[ThreadID]>>>(Globals::config.num_scenarios,
					NumParticles, vnode->GetGPUparticles(),
					Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
					Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
					Dvc_streams[ThreadID], vnode->depth(),
					history.Size());
		else
			_InitBounds<<<GridDim, ThreadDim, thready * Shared_mem_per_particle * sizeof(int)>>>(
					Globals::config.num_scenarios, NumParticles,
					vnode->GetGPUparticles(),
					Dvc_particleIDs_long[ThreadID], Dvc_ub_all_a_p[ThreadID],
					Dvc_uub_all_a_p[ThreadID], Dvc_lb_all_a_p[ThreadID],
					Dvc_streams[ThreadID], vnode->depth(),
					history.Size());
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	ReadBackData(ThreadID);

	double vnode_lower_bound = 0;
	double vnode_upper_bound = 0;
	double vnode_utility_upper = 0;

	for (int i = 0; i < vnode->particleIDs().size(); i++) {
		int parent_PID = vnode->particleIDs()[i];

		vnode_lower_bound += Hst_lb_all_a_p[ThreadID][0
				* NumScenarios + parent_PID].value;
		vnode_upper_bound += Hst_ub_all_a_p[ThreadID][0
				* NumScenarios + parent_PID];				//*weight;
		vnode_utility_upper += Hst_uub_all_a_p[ThreadID][0
				* NumScenarios + parent_PID];				//*weight;
	}

	vnode->lower_bound(vnode_lower_bound);
	vnode->upper_bound(vnode_upper_bound-Globals::config.pruning_constant);
	vnode->utility_upper_bound(vnode_utility_upper);
	int first_particle = 0 * NumScenarios
			+ vnode->particleIDs()[0];
	vnode->default_move(ValuedAction(
			Hst_lb_all_a_p[ThreadID][first_particle].action,
			vnode_lower_bound));

	if (vnode->upper_bound() < vnode->lower_bound()
	// close gap because no more search can be done on leaf node
			|| vnode->depth() == Globals::config.search_depth - 1) {
		vnode->upper_bound(vnode->lower_bound());
	}
}


void DESPOT::GPU_UpdateParticles(VNode* vnode, ScenarioLowerBound* lb,
		ScenarioUpperBound* ub, const DSPOMDP* model, RandomStreams& streams,
		History& history) {

	int ThreadID = 0;
	if (use_multi_thread_)
		ThreadID = MapThread(this_thread::get_id());
	//if(FIX_SCENARIO==1 && ThreadID == 0) cout<<__FUNCTION__<<endl;

	int NumActions = model->NumActions();
	int NumObs = model->NumObservations();
	int NumScenarios = Globals::config.num_scenarios;

	if(use_multi_thread_)
		static_cast<Shared_VNode*>(vnode)->is_waiting_=true;

	auto start_total = Time::now();

	/*Update streams, history, and particles into GPU*/
	UpdateData(vnode, ThreadID,model, streams);

	Dvc_particles_long[ThreadID] = vnode->GetGPUparticles();//get the GPU particles of the parent v-node
	/*if(ThreadID==0)
		printf("Updating address=%#0x \n", vnode->GetGPUparticles());//Debugging*/
	/*Run Monte Carlo simulations in GPU: update particles and perform rollouts*/
	/*if(FIX_SCENARIO==1 && vnode->edge()==8796560787575389481)
		{GPUDoPrint=true;GPUPrintPID=1;}*/
	MCSimulation(vnode, ThreadID,model, streams,history,false);
	/*if(FIX_SCENARIO==1 && vnode->edge()==8796560787575389481)
	{
		HANDLE_ERROR(cudaDeviceSynchronize());
		GPUDoPrint=false;
	}*/
}
DEVICE Dvc_ValuedAction GPU_InitLowerBound( Dvc_State* particles,
		Dvc_RandomStreams& streams, Dvc_History& local_history, int depth) {
	streams.position(depth);

	Dvc_ValuedAction move = DvcLowerBoundValue_( particles, streams,
			local_history);
	//Dvc_ValuedAction move(0,-1);
	move.value *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);
	streams.position(depth);

	return move;
	//vnode->default_move_=move;
	//vnode->lower_bound_=move.value;
}

DEVICE float GPU_InitUpperBound(int scenarioID, const Dvc_State* particles,
/*Dvc_RandomStreams& streams, */Dvc_History& local_history, int depth) {
	float upper = DvcUpperBoundValue_(particles, scenarioID, local_history);
	//double upper=10;
	upper *= Dvc_Globals::Dvc_Discount(Dvc_config, depth);

	//vnode->upper_bound_=upper;
	return upper;
}


void DESPOT::GPUHistoryAdd(int action, OBS_TYPE obs, bool for_all_threads) {
	/*int ThreadID = 0;

	if (!for_all_threads || !use_multi_thread_) {
		if (use_multi_thread_)
			ThreadID = MapThread(this_thread::get_id());
		Dvc_History::Dvc_Add(Dvc_history[ThreadID], action, obs);
	} else if (use_multi_thread_) {
		for (int i = 0; i < NUM_THREADS; i++)
			Dvc_History::Dvc_Add(Dvc_history[i], action, obs);
	}
*/
}
void DESPOT::GPUHistoryTrunc(int size, bool for_all_threads) {
	/*int ThreadID = 0;

	if (!for_all_threads || !use_multi_thread_) {
		if (use_multi_thread_)
			ThreadID = MapThread(this_thread::get_id());
		Dvc_History::Dvc_Trunc(Dvc_history[ThreadID], size);
	} else if (use_multi_thread_) {
		for (int i = 0; i < NUM_THREADS; i++)
			Dvc_History::Dvc_Trunc(Dvc_history[i], size);
	}*/
}
void DESPOT::initGPUHistory() {
	int thread_count = 1;
	if (use_multi_thread_)
		thread_count = NUM_THREADS;
	TotalNumParticles = Globals::config.num_scenarios;

	Dvc_history = new Dvc_History*[thread_count];
	for (int i = 0; i < thread_count; i++) {
		HANDLE_ERROR(
				cudaMallocManaged((void** )&Dvc_history[i],
						1 * sizeof(Dvc_History)));
	}
	Dvc_history[0]->CreateMemoryPool(0);
	cout<<"Globals::config.search_depth="<<Globals::config.search_depth<<endl;
	for (int i = 0; i < thread_count; i++) {
		if (use_multi_thread_)
			Dvc_History::InitInGPU(TotalNumParticles, Dvc_history[i],
					Globals::config.search_depth/*, cuda_streams + i*/);
		else
			Dvc_History::InitInGPU(TotalNumParticles, Dvc_history[i],
					Globals::config.search_depth);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}
void DESPOT::clearGPUHistory() {
	int thread_count = 1;
	if (use_multi_thread_)
		thread_count = NUM_THREADS;
	for (int i = 0; i < thread_count; i++) {
		if (Dvc_history[i] != NULL) {
			dim3 grid((TotalNumParticles + MC_DIM - 1) / MC_DIM, 1);
			dim3 threads(MC_DIM, 1);
			FreeHistory<<<1, 1,1>>>(Dvc_history[i], TotalNumParticles);
			HANDLE_ERROR(cudaDeviceSynchronize());
			//HANDLE_ERROR(cudaFree(Dvc_history));Dvc_history=NULL;
		}
	}
	Dvc_history[0]->DestroyMemoryPool(0);

	if (Dvc_history)
	{
		delete[] Dvc_history;
		Dvc_history = NULL;
	}
}
void DESPOT::PrepareGPUMemory_root(const DSPOMDP* model,
		const std::vector<int>& particleIDs, std::vector<State*>& particles,
		VNode* node) {
	//cout<<__FUNCTION__<<endl;
	int thread_count = 1;
	if (use_multi_thread_)
		thread_count = NUM_THREADS;

	Dvc_State* new_particles = model->AllocGPUParticles(particleIDs.size(), 2);

	//for(int i=0;i<thread_count;i++)
	model->CopyToGPU(particleIDs, Dvc_particleIDs_long[0]);
	if(FIX_SCENARIO==1)
		cout<<"Reset stream pos to "<<0<<endl;
	model->CopyGPUParticles(new_particles, model->GetGPUParticles(), 0,
			Dvc_particleIDs_long[0], particles.size(),
			Dvc_streams[0], 0,
			false);

	//model->CopyGPUWeight1();
	//node->weight_ = model->CopyGPUWeight2();
	node->weight_ =particleIDs.size()/((float)Globals::config.num_scenarios);
	node->AssignGPUparticles(new_particles, particles.size());
}

void DESPOT::PrintGPUData(int num_searches) {
	cout.precision(5);
	if (use_multi_thread_)
		cout << "ExpansionCount (total/per-search)=" << CountExpanded() << "/"
				<< CountExpanded() / num_searches << endl;
	else
		cout << "ExpansionCount (total/per-search)=" << CountExpanded() << "/"
				<< CountExpanded() / num_searches << endl;
	cout.precision(3);
	//cout << "TotalExpansionTime=" << TotalExpansionTime / num_searches << "/"
	//		<< TotalExpansionTime / CountExpanded() << endl;
	/*cout << "AveRewardTime=" << AveRewardTime / num_searches << "/"
			<< AveRewardTime / CountExpanded() << "/"
			<< AveRewardTime / TotalExpansionTime * 100 << "%" << endl;
	cout << "CopyHistoryTime=" << CopyHistoryTime / num_searches << "/"
			<< CopyHistoryTime / CountExpanded() << "/"
			<< CopyHistoryTime / TotalExpansionTime * 100 << "%" << endl;
	cout << "CopyParticleTime=" << CopyParticleTime / num_searches << "/"
			<< CopyParticleTime / CountExpanded() << "/"
			<< CopyParticleTime / TotalExpansionTime * 100 << "%" << endl;

	cout << "InitBoundTime=" << InitBoundTime / num_searches << "/"
			<< InitBoundTime / CountExpanded() << "/"
			<< InitBoundTime / TotalExpansionTime * 100 << "%" << endl;
	cout << "MakePartitionTime=" << MakePartitionTime / num_searches << "/"
			<< MakePartitionTime / CountExpanded() << "/"
			<< MakePartitionTime / TotalExpansionTime * 100 << "%" << endl;
	cout << "MakeObsNodeTime=" << MakeObsNodeTime / num_searches << "/"
			<< MakeObsNodeTime / CountExpanded() << "/"
			<< MakeObsNodeTime / TotalExpansionTime * 100 << "%" << endl;
	cout << "BarrierWaitTime=" << BarrierWaitTime / num_searches << "/"
			<< BarrierWaitTime / CountExpanded() << "/"
			<< BarrierWaitTime / TotalExpansionTime * 100 << "%" << endl;
	cout << "DataBackTime=" << DataBackTime / num_searches << "/"
			<< DataBackTime / CountExpanded() << "/"
			<< DataBackTime / TotalExpansionTime * 100 << "%" << endl;*/
	//cout << "AveNumParticles=" << AveNumParticles << endl;
}
void DESPOT::ValidateGPU(const char* file, int line){
	int deviceIndex;
	HANDLE_ERROR(cudaGetDevice(&deviceIndex));
	cout<<"device@"<<file<<"_line"<<line<<":"<<deviceIndex<<endl;
}

void DESPOT::ChooseGPUForThread(){
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	int deviceIndex=/*devicesCount-1*/Globals::config.GPUid;
	cudaSetDevice(deviceIndex);
}

/*bool DESPOT::PassGPUThreshold(VNode* vnode){
	return (vnode->particleIDs().size()>2 || vnode->depth()<=1);
}*/

int getSPcores(cudaDeviceProp devProp) {
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1)
			cores = mp * 48;
		else
			cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1)
			cores = mp * 128;
		else if (devProp.minor == 0)
			cores = mp * 64;
		else
			printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}
void SimpleTUI::SetupGPU() {
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	int deviceIndex = /*devicesCount-1*/Globals::config.GPUid;

	cudaSetDevice(deviceIndex);
	cudaGetDevice(&deviceIndex);

	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, deviceIndex);
	//HANDLE_ERROR( cudaGetDeviceProperties(&deviceProperties, deviceIndex));
	if (deviceProperties.major >= 2 && deviceProperties.minor >= 0) {
		//HANDLE_ERROR( cudaSetDevice(deviceIndex));
		//cudaSetDevice(deviceIndex);
		cout << "Device:" << "(" << deviceIndex << ")" << deviceProperties.name
				<< endl;
		cout << "Multi-processors:" << deviceProperties.multiProcessorCount
				<< endl;
		//cout << "Wrap size:" << deviceProperties.warpSize << endl;
		//cout << "Global memory:" << deviceProperties.totalGlobalMem << " bytes"
		//		<< endl;
		//cout << "Support managed memory:" << deviceProperties.managedMemory
		//		<< endl;
		//cout << "Shared memory/block:" << deviceProperties.sharedMemPerBlock
		//		<< " bytes" << endl;
		//cout << "Register memory/block:" << deviceProperties.regsPerBlock
		//		<< " 32 bits" << endl;

		//cout << "Clock rate:" << deviceProperties.clockRate << " kHZ" << endl;
		//cout << "Memory clock rate:" << deviceProperties.memoryClockRate
		//		<< " kHZ" << endl;
		//cout << "Memory bus:" << deviceProperties.memoryBusWidth << " bits"
		//		<< endl;
		//cout << "# Async engine:" << deviceProperties.asyncEngineCount << " "
		//		<< endl;
		size_t heapsize;
		cudaDeviceGetLimit(&heapsize, cudaLimitMallocHeapSize);

		cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize * 10);
		cudaDeviceGetLimit(&heapsize, cudaLimitMallocHeapSize);
		//cout << "Heap size limit:" << heapsize << " " << endl;
		cudaCoreNum = getSPcores(deviceProperties);
		cout << "Number of cores:" << cudaCoreNum << endl;

		asyncEngineCount = 32;

		if (asyncEngineCount >= 2) {
			if (use_multi_thread_) {
				cuda_streams = new cudaStream_t[NUM_THREADS];

				for (int i = 0; i < NUM_THREADS; i++)
					HANDLE_ERROR(cudaStreamCreate(&cuda_streams[i]));

				thread_barrier=new spinlock_barrier(NUM_THREADS);
			} else {
				cuda_streams = NULL;
			}
		} else
			cout << "The current GPU no enough asyncEngine (<2)" << endl;
	}
	std::memset((void*)&sa, 0, sizeof(struct sigaction));
	//sa.sa_handler = segfault_sigaction;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = segfault_sigaction;
	sa.sa_flags   = SA_SIGINFO;

	sigaction(SIGSEGV, &sa, NULL);
}

void AdvanceStreamCounter(int stride) {
	stream_counter += stride;
	if (stream_counter >= asyncEngineCount)
		stream_counter = 1;
}

} // namespace despot
