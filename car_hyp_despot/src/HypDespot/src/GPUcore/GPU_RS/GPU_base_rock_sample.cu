#include "GPU_base_rock_sample.h"

#include <base/base_rock_sample.h>
#include "GPU_rock_sample.h"
#include "despot/GPUutil/GPUutil.h"
#include "despot/solver/GPUdespot.h"

using namespace std;

namespace despot {

#define THREADDIM 128

DEVICE int map_size_=NULL;
DEVICE int num_rocks_=NULL;
DEVICE double half_efficiency_distance_=NULL;
DEVICE int* grid_=NULL;/*A flattened pointer of a 2D map*/
DEVICE DvcCoord* rock_pos_=NULL;
DEVICE Dvc_RockSample* rs_model_=NULL;
DEVICE int Dvc_policy_size_=0;
DEVICE Dvc_ValuedAction* Dvc_policy_=NULL;

static GPU_MemoryPool<Dvc_RockSampleState>* gpu_memory_pool_=NULL;
static Dvc_RockSampleState* Dvc_particles=NULL;
//static int* Dvc_particleIDs=NULL;
static Dvc_RockSampleState* tmp=NULL;
static float** tmp_result=NULL;
static int** tempHostID;
static float** temp_weight;

/* ==============================================================================
 * RockSampleState class
 * ==============================================================================*/

DEVICE Dvc_RockSampleState::Dvc_RockSampleState() {
}

HOST void Dvc_RockSampleState::CopyToGPU(Dvc_RockSampleState* Dvc, int scenarioID, const RockSampleState* Hst, bool copy_cells)
{
	Dvc[scenarioID].weight=Hst->weight;
	Dvc[scenarioID].allocated_=Hst->allocated_;
	Dvc[scenarioID].state_id=Hst->state_id;
	Dvc[scenarioID].scenario_id=Hst->scenario_id;
}


DEVICE DvcCoord Dvc_RockSample::GetCoord(int index)
{
	assert(index >= 0 && index < map_size_ * map_size_);
	return DvcCoord(index % map_size_, index / map_size_);
}
DEVICE DvcCoord Dvc_RockSample::GetRobPos(const Dvc_State* state) {
	return GetCoord(state->state_id >> num_rocks_);
}

DEVICE bool Dvc_RockSample::GetRock(const Dvc_State* state, int rock) {
	return Dvc_CheckFlag(state->state_id, rock);
}

DEVICE int Dvc_RockSample::GetX(const Dvc_State* state) {
	return (state->state_id >> num_rocks_) % map_size_;
}

DEVICE int Dvc_RockSample::GetY(const Dvc_State* state) {
	return (state->state_id >> num_rocks_) / map_size_;
}

DEVICE void Dvc_RockSample::IncX(Dvc_State* state) {
	state->state_id += (1 << num_rocks_);
}

DEVICE void Dvc_RockSample::DecX(Dvc_State* state) {
	state->state_id -= (1 << num_rocks_);
}

DEVICE void Dvc_RockSample::IncY(Dvc_State* state) {
	state->state_id += (1 << num_rocks_) * map_size_;
}

DEVICE void Dvc_RockSample::DecY(Dvc_State* state) {
	state->state_id -= (1 << num_rocks_) * map_size_;
}

DEVICE int Dvc_RockSample::GetRobPosIndex(const Dvc_State* state) {
	return state->state_id >> num_rocks_;
}

DEVICE void Dvc_RockSample::SampleRock(Dvc_State* state, int rock) {
	Dvc_UnsetFlag(state->state_id, rock);
}

/*
class RockSampleParticleUpperBound1: public ParticleUpperBound {
protected:
	const BaseRockSample* rs_model_;
public:
	RockSampleParticleUpperBound1(const BaseRockSample* model) :
		rs_model_(model) {
	}

	double Value(const State& state) const {
		const RockSampleState& rockstate =
			static_cast<const RockSampleState&>(state);
		int count = 0;
		for (int rock = 0; rock < num_rocks_; rock++)
			count += rs_model_->GetRock(&rockstate, rock);
		return 10.0 * (1 - Dvc_Globals::Dvc_Discount(Dvc_config,count + 1)) / (1 - Dvc_Globals::Dvc_Discount(Dvc_config,));
	}
};

class RockSampleParticleUpperBound2: public ParticleUpperBound {
protected:
	const BaseRockSample* rs_model_;
public:
	RockSampleParticleUpperBound2(const BaseRockSample* model) :
		rs_model_(model) {
	}

	double Value(const State& state) const {
		const RockSampleState& rockstate =
			static_cast<const RockSampleState&>(state);
		double value = 0;
		for (int rock = 0; rock < num_rocks_; rock++)
			value += 10.0 * rs_model_->GetRock(&rockstate, rock)
				* Dvc_Globals::Dvc_Discount(Dvc_config,
					DvcCoord::ManhattanDistance(rs_model_->GetRobPos(&rockstate),
						rock_pos_[rock]));
		value += 10.0
			* Dvc_Globals::Dvc_Discount(Dvc_config,map_size_ - rs_model_->GetX(&rockstate));
		return value;
	}
};

class RockSampleMDPParticleUpperBound: public ParticleUpperBound {
protected:
	const BaseRockSample* rs_model_;
	vector<Dvc_ValuedAction> policy_;
public:
	RockSampleMDPParticleUpperBound(const BaseRockSample* model) :
		rs_model_(model) {
		policy_ = rs_model_->ComputeOptimalSamplingPolicy();
	}

	double Value(const State& state) const {
		return policy_[state.state_id].value;
	}
};*/

DEVICE float Dvc_RockSampleApproxParticleUpperBound::Value(const Dvc_State* particles, int scenarioID, Dvc_History& history)
{
	const Dvc_RockSampleState* rs_state =
		static_cast<const Dvc_RockSampleState*>(particles) + scenarioID;
	float value = 0;
	float discount = 1.0;
	DvcCoord rob_pos = rs_model_->GetRobPos(rs_state);
	__shared__ bool visited[NUM_ROCKS];
	//vector<bool> visited(num_rocks_);
	while (true) {
		// Move to the nearest valuable rock and sample
		int shortest = 2 * map_size_;
		int id = -1;
		DvcCoord rock_pos(-1, -1);
		for (int rock = 0; rock < num_rocks_; rock++) {
			int dist = DvcCoord::ManhattanDistance(rob_pos,
				rock_pos_[rock]);
			if (Dvc_CheckFlag(rs_state->state_id, rock) && dist < shortest
				&& !visited[rock]) {
				shortest = dist;
				id = rock;
				rock_pos = rock_pos_[rock];
			}
		}

		if (id == -1)
			break;

		discount *= Dvc_Globals::Dvc_Discount(Dvc_config, DvcCoord::ManhattanDistance(rock_pos, rob_pos));
		value += discount * 10.0;
		visited[id] = true;
		rob_pos = rock_pos;
	}

	value += 10.0 * discount
		* Dvc_Globals::Dvc_Discount(Dvc_config,map_size_ - rs_model_->GetX(rs_state));
	return value;
}
DEVICE float Dvc_RockSampleMDPParticleUpperBound::Value(const Dvc_State* particles, int scenarioID, Dvc_History& history) {
	const Dvc_RockSampleState* rs_state =
		static_cast<const Dvc_RockSampleState*>(particles) + scenarioID;
	return Dvc_policy_[rs_state->state_id].value;
}
DEVICE Dvc_ValuedAction Dvc_RockSampleEastScenarioLowerBound::Value(
		Dvc_State* particles,
		Dvc_RandomStreams& streams,
		Dvc_History& history) {
	const Dvc_RockSampleState* rs_state =
			static_cast<const Dvc_RockSampleState*>(particles);
		return Dvc_ValuedAction(Dvc_Compass::EAST,
			10/* * rs_state->weight*/
				* Dvc_Globals::Dvc_Discount(Dvc_config, map_size_ - rs_model_->GetX(rs_state) - 1));
	}
/*
class RockSampleEastScenarioLowerBound : public ScenarioLowerBound {
private:
	const BaseRockSample* rs_model_;
	const Grid<int>& grid_;

public:
	RockSampleEastScenarioLowerBound(const DSPOMDP* model) :
		ScenarioLowerBound(model),
		rs_model_(static_cast<const BaseRockSample*>(model)),
		grid_(rs_model_->grid_) {
	}

	Dvc_ValuedAction Value(const vector<State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const {
		return Dvc_ValuedAction(Compass::EAST,
			10 * State::Weight(particles)
				* Dvc_Globals::Dvc_Discount(Dvc_config,grid_.xsize() - rs_model_->GetX(particles[0]) - 1));
	}
};

class RockSampleMMAPStateScenarioLowerBound : public ScenarioLowerBound {
private:
	const BaseRockSample* rs_model_;
	const Grid<int>& grid_;
	vector<vector<int> > rock_order_;

public:
	RockSampleMMAPStateScenarioLowerBound(const DSPOMDP* model) :
		ScenarioLowerBound(model),
		rs_model_(static_cast<const BaseRockSample*>(model)),
		grid_(rs_model_->grid_) {
		const vector<Dvc_ValuedAction> policy =
			rs_model_->ComputeOptimalSamplingPolicy();
		rock_order_ = vector<vector<int> >(policy.size());
		for (int s = 0; s < policy.size(); s++) {
			int cur = s;
			while (cur != policy.size() - 1) {
				int action = policy[cur].action;
				if (action == rs_model_->E_SAMPLE) {
					int rock = grid_(
						rs_model_->IndexToCoord(cur >> num_rocks_));
					if (rock < 0)
						exit(0);
					rock_order_[s].push_back(rock);
				}
				cur = rs_model_->NextState(cur, action);
			}
		}
	}

	Dvc_ValuedAction Value(const vector<State*>& particles, Dvc_RandomStreams& streams,
		Dvc_History& history) const {
		vector<double> expected_sampling_value = vector<double>(
			num_rocks_);
		int state = 0;
		DvcCoord rob_pos(-1, -1);
		double total_weight = 0;
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
			state = (1 << num_rocks_)
				* rs_model_->GetRobPosIndex(particle);
			rob_pos = rs_model_->GetRobPos(particle);
			for (int r = 0; r < num_rocks_; r++) {
				expected_sampling_value[r] += particle->weight
					* (Dvc_CheckFlag(particle->state_id, r) ? 10 : -10);
			}
			total_weight += particle->weight;
		}

		for (int rock = 0; rock < num_rocks_; rock++)
			if (expected_sampling_value[rock] / total_weight > 0.0)
				SetFlag(state, rock);

		int action = -1;
		double value = 0;
		double discount = 1.0;
		for (int r = 0; r < rock_order_[state].size(); r++) {
			int rock = rock_order_[state][r];
			DvcCoord rock_pos = rock_pos_[rock];

			if (action == -1) {
				if (rock_pos.x < rob_pos.x)
					action = Compass::WEST;
				else if (rock_pos.x > rob_pos.x)
					action = Compass::EAST;
				else if (rock_pos.y < rob_pos.y)
					action = Compass::SOUTH;
				else if (rock_pos.y > rob_pos.y)
					action = Compass::NORTH;
				else
					action = rs_model_->E_SAMPLE;
			}

			discount *= Dvc_Globals::Dvc_Discount(Dvc_config,DvcCoord::ManhattanDistance(rock_pos, rob_pos));
			value += discount * expected_sampling_value[rock];
			rob_pos = rock_pos;
		}

		if (action == -1)
			action = Compass::EAST;

		value += 10 * total_weight * discount
			* Dvc_Globals::Dvc_Discount(Dvc_config,grid_.xsize() - rob_pos.x);

		return Dvc_ValuedAction(action, value);
	}
};
*/

Dvc_State* BaseRockSample::AllocGPUParticles(int numParticles,int alloc_mode) const
{//numParticles==num_Scenarios
	clock_t start=clock();

	int threadCount=1;
	if(use_multi_thread_)
		threadCount=NUM_THREADS;
	switch(alloc_mode)
	{
	case 0:
		HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles, numParticles*sizeof(Dvc_RockSampleState)));

		HANDLE_ERROR(cudaMallocManaged((void**)&tmp,sizeof(Dvc_RockSampleState)));
		//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particleIDs,numParticles*sizeof(int) ));

		Dvc_stepped_particles_all_a=new Dvc_State*[threadCount];
		for(int i=0;i<threadCount;i++)
		HANDLE_ERROR(cudaMalloc((void**)&Dvc_stepped_particles_all_a[i], NumActions()*numParticles*sizeof(Dvc_RockSampleState)));

		tempHostID=new int*[threadCount];
		for(int i=0;i<threadCount;i++)
		{
			cudaHostAlloc(&tempHostID[i],numParticles*sizeof(int),0);
		}

		temp_weight=new float*[threadCount];
		for(int i=0;i<threadCount;i++)
			cudaHostAlloc(&temp_weight[i],1*sizeof(float),0);

		tmp_result=new float*[threadCount];
		for(int i=0;i<threadCount;i++)
			HANDLE_ERROR(cudaMalloc(&tmp_result[i], sizeof(float)));

		return Dvc_particles;
	case 1:
		CreateMemoryPool(0);
		return Dvc_particles;
	case 2:
		//cout<<__FUNCTION__ <<endl;
		//cout<<"numParticles"<<numParticles<<endl;
		//cout<<"gpu_memory_pool_"<<gpu_memory_pool_<<endl;
		Dvc_RockSampleState* tmp=gpu_memory_pool_->Allocate(numParticles);
		//Dvc_RockSampleState* tmp;
		//HANDLE_ERROR(cudaMalloc((void**)&tmp, numParticles*sizeof(Dvc_RockSampleState)));

		return tmp;
	};
	/*Dvc_particles_copy is an extern variable declared in GPUpolicy.h*/
	//HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_particles_copy, numParticles*sizeof(Dvc_RockSampleState)));
	/*AllocParticleCopy<<<grid, threads>>>(size_, size_,numParticles);
	HANDLE_ERROR(cudaDeviceSynchronize());*/

	cout<<"GPU particles alloc time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl;
}
__global__ void CopyParticles(Dvc_RockSampleState* des,Dvc_RockSampleState* src,float* weight,
		int* IDs,int num_particles,Dvc_RandomStreams* streams, int stream_pos)
{
	int pos=blockIdx.x*blockDim.x+threadIdx.x;
	//int y=threadIdx.y;

	if(pos==0)
	{
		weight[0]=0;
		if(streams) streams->position_ = stream_pos;
	}
	__syncthreads();
	if(pos < num_particles)
	{
		bool error=false;
		int src_pos=IDs[pos];
		Dvc_RockSampleState* src_i=src+src_pos;//src is a full length array for all particles
		Dvc_RockSampleState* des_i=des+pos;//des is short, only for the new partition

		des_i->weight=src_i->weight;
		des_i->allocated_=src_i->allocated_;
		des_i->state_id=src_i->state_id;
		des_i->scenario_id=src_i->scenario_id;
		//IDs[pos]=src_i->scenario_id;//Important!! Change the IDs to scenario IDs for later usage
		if(des_i->weight>=0.000201 || des_i->weight<=0.000199)
		{
			error=true;//error here
		}
		else
		{
			error=false;
		}

		atomicAdd(weight, des_i->weight);

		pos=error;
	}
}

void BaseRockSample::CopyGPUParticles(Dvc_State* des,Dvc_State* src,int src_offset,int* IDs,
		int num_particles,bool interleave,
		Dvc_RandomStreams* streams, int stream_pos,
		void* CUDAstream, int shift) const
{
	//dim3 grid(1,1); dim3 threads(num_particles,1);
	dim3 grid((num_particles+THREADDIM-1)/THREADDIM,1); dim3 threads(THREADDIM,1);


	int ThreadID=0;
	if(use_multi_thread_)
		ThreadID=MapThread(this_thread::get_id());
	if(CUDAstream)
	{
		CopyParticles<<<grid, threads,0, *(cudaStream_t*)CUDAstream>>>(static_cast<Dvc_RockSampleState*>(des),
				static_cast<Dvc_RockSampleState*>(src)+src_offset,tmp_result[ThreadID],
				IDs,num_particles, streams, stream_pos);
		if(!interleave)
			;//HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)CUDAstream));
	}
	else
	{
		CopyParticles<<<grid, threads,0, 0>>>(static_cast<Dvc_RockSampleState*>(des),
				static_cast<Dvc_RockSampleState*>(src)+src_offset,tmp_result[ThreadID],
				IDs,num_particles, streams, stream_pos);
		if(!interleave)
			HANDLE_ERROR(cudaDeviceSynchronize());
	}


		//HANDLE_ERROR(cudaDeviceSynchronize());
}
void BaseRockSample::CopyGPUWeight1(void* cudaStream, int shift) const
{
	;
}
float BaseRockSample::CopyGPUWeight2(void* cudaStream, int shift) const
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
		HANDLE_ERROR(cudaMemcpyAsync(temp_weight[ThreadID], tmp_result[ThreadID], sizeof(float),cudaMemcpyDeviceToHost, *(cudaStream_t*)cudaStream));
		if(*temp_weight[ThreadID]>1)
			Global_print_value(this_thread::get_id(),*temp_weight[ThreadID],"Wrong weight from CopyGPUWeight");
		HANDLE_ERROR(cudaStreamSynchronize(*(cudaStream_t*)cudaStream));
		return *temp_weight[ThreadID];
	}
}
void BaseRockSample::DeleteGPUParticles( int num_particles) const
{
	HANDLE_ERROR(cudaFree(static_cast<Dvc_RockSampleState*>(Dvc_particles)));

	//HANDLE_ERROR(cudaFree(Dvc_particleIDs));

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

	cudaFree(tmp);


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
}

void BaseRockSample::DeleteGPUParticles(Dvc_State* particles, int num_particles) const
{
	HANDLE_ERROR(cudaFree(static_cast<Dvc_RockSampleState*>(particles)));
}

Dvc_State* BaseRockSample::GetGPUParticles() const
{
	return Dvc_particles;
}
Dvc_State* BaseRockSample::CopyToGPU(const std::vector<State*>& particles, bool copy_cells) const
{
	//Dvc_RockSampleState* Dvc_particles;
	//HANDLE_ERROR(cudaMalloc((void**)&Dvc_particles, particles.size()*sizeof(Dvc_RockSampleState)));
	//dvc_particles should be managed device memory
	clock_t start=clock();

	for (int i=0;i<particles.size();i++)
	{
		const RockSampleState* src=static_cast<const RockSampleState*>(particles[i]);
		//Dvc_particles[i].Assign(src);
		Dvc_RockSampleState::CopyToGPU(Dvc_particles,src->scenario_id,src);
		//Dvc_RockSampleState::CopyToGPU(NULL,src->scenario_id,src, false);//copy to Dvc_particles_copy, do not copy cells, leave it a NULL pointer
	}
	//cout<<"GPU particles copy time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl;

	return Dvc_particles;
}
void BaseRockSample::CopyToGPU(const std::vector<int>& particleIDs, int* Dvc_ptr, void* CUDAstream) const
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

void BaseRockSample::CreateMemoryPool(int chunk_size) const
{
	//cout<<__FUNCTION__<<endl;
	if(gpu_memory_pool_==NULL)
		gpu_memory_pool_=new GPU_MemoryPool<Dvc_RockSampleState>;
	//gpu_memory_pool_->SetChunkSize(chunk_size);
}
void BaseRockSample::DestroyMemoryPool(int mode) const
{
	//cout<<__FUNCTION__<<endl;
	if(gpu_memory_pool_){delete gpu_memory_pool_;gpu_memory_pool_=NULL;}
}

__global__ void PassPolicy(int policy_size, Dvc_ValuedAction* policy)
{
	Dvc_policy_=policy;
	Dvc_policy_size_=policy_size;
}

void RockSampleMDPParticleUpperBound::InitializeinGPU()
{
	if(Globals::config.useGPU)
	{
		  Dvc_ValuedAction* tmp_policy;
		  cout<<"Loading MDP policy of size "<< policy_.size()<<" into GPU memory..."<<flush;

		  HANDLE_ERROR(cudaMallocManaged((void**)&tmp_policy, policy_.size()*sizeof(Dvc_ValuedAction)));
		  for(int i=0;i<policy_.size();i++)
		  {
			  tmp_policy[i].action=policy_[i].action;
			  tmp_policy[i].value=policy_[i].value;
		  }
		  PassPolicy<<<1,1,1>>>(policy_.size(),tmp_policy);
		  HANDLE_ERROR(cudaDeviceSynchronize());
		  cout<<"done"<<endl;
	}

  //HANDLE_ERROR(cudaFree(tmp_policy));
}
void RockSampleMDPParticleUpperBound::ImportOptimalPolicy()
{
	ifstream fin;fin.open("OptimalPolicy.txt", ios::in);
	if (fin.good())
	{
		cout<<"Start importing optimal policy from OptimalPolicy.txt"<<endl;
		string str;
		getline(fin, str);
		istringstream ss(str);

		int policysize;
		ss>>policysize;
		cout<<"Allocating "<<policysize*(sizeof(int)+sizeof(double))<<" bytes..."<<flush;
		policy_.resize(policysize);
		cout<<"done"<<endl;

		getline(fin, str);
		istringstream ss1(str);
		int block=policysize/10;
		for (int j = 0; j < policysize; j++)
		{
			ss1 >> policy_[j].action >> policy_[j].value;
			if(j%block==0)
				cout<<j/block<<"/10 completed"<<endl;
		}
	}
	else
	{
		cout<<__FUNCTION__<<": Empty Optimal policy file!"<<endl;
		exit(-1);
	}
	fin.close();
}
void RockSampleMDPParticleUpperBound::ExportOptimalPolicy(string filename)
{
	ofstream fout;fout.open(filename.c_str(), ios::trunc);
	fout<<policy_.size()<<endl;
	for (int j = 0; j < policy_.size(); j++)
	{
		fout<<policy_[j].action <<" "<< policy_[j].value<<" ";
	}

	fout.close();
}

} // namespace despot
