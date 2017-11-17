#include <despot/GPUcore/GPUpolicy_graph.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <unistd.h>

#include <despot/GPUutil/GPUmap.h>

using namespace std;

namespace despot {
DEVICE int graph_size_=0;
DEVICE int num_edges_per_node_=0;
DEVICE int* action_nodes_=NULL;
DEVICE int* obs_edges_=NULL;

__device__ int graph_entry_node=0;
/*

DEVICE Dvc_ValuedAction Dvc_PolicyGraph::Value(int scenarioID,Dvc_State* particles,
	Dvc_RandomStreams& streams, Dvc_History& Local_history) {
	Dvc_State* particle = particles;
	__shared__ int current_node_[DIM];
	__shared__ int all_terminated[DIM];
	__shared__ int action[DIM];

	if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
	{
		if(FIX_SCENARIO)
			graph_entry_node=0;//Debug
		else
			graph_entry_node=Dvc_random->NextInt(graph_size_, 0 );
	}
	__syncthreads();
	if(threadIdx.x==0)
		current_node_[threadIdx.y]=graph_entry_node;
	//__syncthreads();

	//int action;
	float Accum_Value=0;

	int init_depth=Local_history.currentSize_;


	int MaxDepth=min(Dvc_config->max_policy_sim_len+init_depth,streams.Length());
	//int terminal = false;
	int depth;
	int Action_decision=action_nodes_[graph_entry_node];
	int terminal;

	for(depth=init_depth;depth<MaxDepth;depth++)
	{
		if(threadIdx.x==0)
		{
			all_terminated[threadIdx.y]=true;
		}

		if(threadIdx.x==0)
		{
			action[threadIdx.y] = action_nodes_[current_node_[threadIdx.y]];
			//if(depth==init_depth)
				//Action_decision=action[threadIdx.y];
		}
		__syncthreads();

		OBS_TYPE obs;
		float reward;
		terminal = DvcModelStep_(*particle, streams.Entry(scenarioID), action[threadIdx.y], reward, obs);

		if(threadIdx.x==0)
		{
			atomicAnd(&all_terminated[threadIdx.y],terminal);

			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+0)* reward;//particle->weight;
		}
		streams.Advance();

		if(threadIdx.x==0)
			current_node_[threadIdx.y]=obs_edges_[obs*graph_size_+current_node_[threadIdx.y]];
		__syncthreads();
		if(all_terminated[threadIdx.y])
		{
			break;
		}

	}
	//use default value for leaf positions
	if(threadIdx.x==0)
	{
		if(!terminal)
		{
			Dvc_ValuedAction va = DvcParticleLowerBound_Value_(0,particle);
			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+1) * va.value;
		}
	}

	//the value returned here need to be weighted summed to get the real value of the action
	return Dvc_ValuedAction(Action_decision, Accum_Value);
}
*/

DEVICE Dvc_ValuedAction Dvc_PolicyGraph::Value(Dvc_State* particles,
	Dvc_RandomStreams& streams, Dvc_History& Local_history) {
	Dvc_State* particle = particles;
	__shared__ int current_node_[MC_DIM];
	__shared__ int all_terminated[MC_DIM];
	__shared__ int action[MC_DIM];

	if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
	{
		if(FIX_SCENARIO)
			graph_entry_node=0;//Debug
		else
			graph_entry_node=Dvc_random->NextInt(graph_size_, 0 );
	}
	__syncthreads();
	if(threadIdx.y==0)
		current_node_[threadIdx.x]=graph_entry_node;
	//__syncthreads();

	//int action;
	float Accum_Value=0;

	int init_depth=Local_history.currentSize_;


	int MaxDepth=min(Dvc_config->max_policy_sim_len+init_depth,streams.Length());
	//int terminal = false;
	int depth;
	int Action_decision=action_nodes_[graph_entry_node];
	int terminal;

	for(depth=init_depth;depth<MaxDepth;depth++)
	{
		if(threadIdx.y==0)
		{
			all_terminated[threadIdx.x]=true;
		}

		if(threadIdx.y==0)
		{
			action[threadIdx.x] = action_nodes_[current_node_[threadIdx.x]];
			//if(depth==init_depth)
				//Action_decision=action[threadIdx.y];
		}
		__syncthreads();

		OBS_TYPE obs;
		float reward;

		terminal = DvcModelStep_(*particle, streams.Entry(particle->scenario_id), action[threadIdx.x], reward, obs);
		if(threadIdx.y==0)
		{
			atomicAnd(&all_terminated[threadIdx.x],terminal);

			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+0)* reward;//particle->weight;
		}
		streams.Advance();

		if(threadIdx.y==0)
			current_node_[threadIdx.x]=obs_edges_[obs*graph_size_+current_node_[threadIdx.x]];
		__syncthreads();
		if(all_terminated[threadIdx.x])
		{
			break;
		}

	}
	//use default value for leaf positions
	if(threadIdx.y==0)
	{
		if(!terminal)
		{
			Dvc_ValuedAction va = DvcParticleLowerBound_Value_(0,particle);
			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+1) * va.value;
		}
	}

	//the value returned here need to be weighted summed to get the real value of the action
	return Dvc_ValuedAction(Action_decision, Accum_Value);
}

} // namespace despot
