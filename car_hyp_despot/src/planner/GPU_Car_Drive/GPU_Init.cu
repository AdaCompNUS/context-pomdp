#include "GPU_Init.h"

#include <despot/GPUconfig.h>
#include <despot/GPUcore/CudaInclude.h>
#include <despot/GPUcore/GPUglobals.h>
#include <despot/GPUcore/GPUpolicy_graph.h>

#include "GPU_Car_Drive.h"
#include "GPU_CarUpperBound.h"
#include "GPU_LowerBoundPolicy.h"

#include <despot/solver/GPUdespot.h>

#include <ped_pomdp.h>
#include <vector>
#include <simulator_hyp.h>
using namespace despot;
using namespace std;

static Dvc_PedPomdp* Dvc=NULL;
static Dvc_Policy* inde_lowerbound=NULL;
static Dvc_PedPomdpSmartPolicyGraph* graph_lowerbound=NULL;
static Dvc_TrivialParticleLowerBound* b_lowerbound=NULL;
static Dvc_PedPomdpParticleLowerBound* b_smart_lowerbound=NULL;
static Dvc_PedPomdpParticleUpperBound1* upperbound=NULL;
static Dvc_PedPomdpSmartPolicy* smart_lowerbound=NULL;

static Dvc_COORD* tempGoals=NULL;
static Dvc_COORD* tempPath=NULL;

__global__ void PassPolicyGraph(int graph_size, int num_edges_per_node, int* action_nodes, int* obs_edges)
{
	graph_size_=graph_size;
	num_edges_per_node_=num_edges_per_node;
	action_nodes_=action_nodes;
	obs_edges_=obs_edges;
}

void Simulator::InitializeGPUPolicyGraph(PolicyGraph* hostGraph)
{
	  int* tmp_node_list; int* tmp_edge_list;
	  HANDLE_ERROR(cudaMalloc((void**)&tmp_node_list, hostGraph->graph_size_*sizeof(int)));
	  HANDLE_ERROR(cudaMalloc((void**)&tmp_edge_list, hostGraph->graph_size_*hostGraph->num_edges_per_node_*sizeof(int)));

	  HANDLE_ERROR(cudaMemcpy(tmp_node_list, hostGraph->action_nodes_.data(), hostGraph->graph_size_*sizeof(int), cudaMemcpyHostToDevice));

	  for (int i = 0; i < hostGraph->num_edges_per_node_; i++)
	  {
		  HANDLE_ERROR(cudaMemcpy(tmp_edge_list+i*hostGraph->graph_size_, hostGraph->obs_edges_[(OBS_TYPE)i].data(), hostGraph->graph_size_*sizeof(int), cudaMemcpyHostToDevice));
	  }

	  PassPolicyGraph<<<1,1,1>>>(hostGraph->graph_size_,hostGraph->num_edges_per_node_,
			  tmp_node_list,tmp_edge_list );
	  HANDLE_ERROR(cudaDeviceSynchronize());
}

void Simulator::InitializeGPUGlobals()
{
	  HANDLE_ERROR(cudaMallocManaged((void**)&Dvc_Globals::config, sizeof(Dvc_Config)));
	  Dvc_Globals::config->search_depth=Globals::config.search_depth;
	  Dvc_Globals::config->discount=Globals::config.discount;
	  Dvc_Globals::config->root_seed=Globals::config.root_seed;
	  Dvc_Globals::config->time_per_move=Globals::config.time_per_move;  // CPU time available to construct the search tree
	  Dvc_Globals::config->num_scenarios=Globals::config.num_scenarios;
	  Dvc_Globals::config->pruning_constant=Globals::config.pruning_constant;
	  Dvc_Globals::config->xi=Globals::config.xi; // xi * gap(root) is the target uncertainty at the root.
	  Dvc_Globals::config->sim_len=Globals::config.sim_len; // Number of steps to run the simulation for.
	  Dvc_Globals::config->max_policy_sim_len=Globals::config.max_policy_sim_len; // Maximum number of steps for simulating the default policy
	  Dvc_Globals::config->noise=Globals::config.noise;
	  Dvc_Globals::config->silence=Globals::config.silence;

}


void Simulator::DeleteGPUModel()
{
	  HANDLE_ERROR(cudaFree(Dvc));
	  if(inde_lowerbound)HANDLE_ERROR(cudaFree(inde_lowerbound));
	  if(graph_lowerbound)HANDLE_ERROR(cudaFree(graph_lowerbound));
	  if(smart_lowerbound)HANDLE_ERROR(cudaFree(smart_lowerbound));

	  if(b_lowerbound)HANDLE_ERROR(cudaFree(b_lowerbound));
	  if(b_smart_lowerbound)HANDLE_ERROR(cudaFree(b_smart_lowerbound));
	  HANDLE_ERROR(cudaFree(upperbound));
	  if(tempGoals)HANDLE_ERROR(cudaFree(tempGoals));
	  if(tempPath)HANDLE_ERROR(cudaFree(tempPath));
}

void Simulator::DeleteGPUGlobals()
{
    HANDLE_ERROR(cudaFree(Dvc_Globals::config));
}


__global__ void PassModelFuncs(Dvc_PedPomdp* model,
		double _in_front_angle_cos, double _freq,
		Dvc_COORD* _goals, Dvc_COORD* _path, int pathsize)
{
	DvcModelStepIntObs_=&(model->Dvc_Step);
	DvcModelCopyNoAlloc_=&(model->Dvc_Copy_NoAlloc);
	DvcModelCopyToShared_=&(model->Dvc_Copy_ToShared);
	DvcModelGet_=&(model->Dvc_Get);
	DvcModelGetMinRewardAction_=&(model->Dvc_GetMinRewardAction);
	in_front_angle_cos=_in_front_angle_cos;
	freq=_freq;
	goals=_goals;
	if(path == NULL) path=new Dvc_Path();
	printf("pass model to gpu\n");
	 
}
__global__ void UpdatePathKernel(Dvc_COORD* _path, int pathsize)
{
	if(path) {delete path; path=new Dvc_Path();}
	if(path==NULL) 	path=new Dvc_Path();

	path->size_=pathsize;
	path->pos_=0;
	path->way_points_=_path;
	printf("pass path to gpu %d\n", path);
}

__global__ void UpdateGoalKernel(Dvc_COORD* _goals)
{
	goals=_goals;
}

__global__ void PassActionValueFuncs(Dvc_PedPomdp* model, Dvc_PedPomdpSmartPolicy* lowerbound,
		/*Dvc_TrivialParticleLowerBound*/Dvc_PedPomdpParticleLowerBound* b_lowerbound,
		Dvc_PedPomdpParticleUpperBound1* upperbound)
{
	DvcPolicyAction_=&(lowerbound->Action);

	DvcLowerBoundValue_=&(lowerbound->Value);//DvcRandomPolicy.Value
	DvcUpperBoundValue_=&(upperbound->Value);//DvcUncNavigationParticleUpperBound1.Value

	DvcParticleLowerBound_Value_=&(b_lowerbound->Value);//DvcTrivialParticleLowerBound.Value

}

__global__ void PassValueFuncs(Dvc_PedPomdpSmartPolicyGraph* lowerbound,
		/*Dvc_TrivialParticleLowerBound*/Dvc_PedPomdpParticleLowerBound* b_lowerbound,
		Dvc_PedPomdpParticleUpperBound1* upperbound)
{
	DvcLowerBoundValue_=&(lowerbound->Value);//DvcRandomPolicy.Value
	DvcUpperBoundValue_=&(upperbound->Value);//DvcUncNavigationParticleUpperBound1.Value

	DvcParticleLowerBound_Value_=&(b_lowerbound->Value);//DvcTrivialParticleLowerBound.Value

}

/*__global__ void PassValueFuncs(Dvc_PedPomdpSmartPolicy* lowerbound,
		Dvc_TrivialParticleLowerBound* b_lowerbound,
		Dvc_PedPomdpParticleUpperBound1* upperbound)
{
	DvcLowerBoundValue_=&(lowerbound->Value);//DvcRandomPolicy.Value
	DvcUpperBoundValue_=&(upperbound->Value);//DvcUncNavigationParticleUpperBound1.Value

	DvcParticleLowerBound_Value_=&(b_lowerbound->Value);//DvcTrivialParticleLowerBound.Value

}*/

__global__ void PassModelParameters(
		double GOAL_TRAVELLED,
		int N_PED_IN,
		int N_PED_WORLD,
		double VEL_MAX,
		double NOISE_GOAL_ANGLE,
		double CRASH_PENALTY,
		double REWARD_FACTOR_VEL,
		double REWARD_BASE_CRASH_VEL,
		double BELIEF_SMOOTHING,
		double NOISE_ROBVEL,
		double COLLISION_DISTANCE,
		double IN_FRONT_ANGLE_DEG,
		double LASER_RANGE,
		double pos_rln, // position resolution
		double vel_rln, // velocity resolution
		double PATH_STEP,
		double GOAL_TOLERANCE,
		double PED_SPEED,
		bool debug,
		double control_freq,
		double AccSpeed,
		double GOAL_REWARD){
	Dvc_ModelParams::GOAL_TRAVELLED =  GOAL_TRAVELLED ;
	Dvc_ModelParams::N_PED_IN = N_PED_IN  ;
	Dvc_ModelParams::N_PED_WORLD = N_PED_WORLD  ;
	Dvc_ModelParams::VEL_MAX =  VEL_MAX ;
	Dvc_ModelParams::NOISE_GOAL_ANGLE = NOISE_GOAL_ANGLE  ;
	Dvc_ModelParams::CRASH_PENALTY =  CRASH_PENALTY ;
	Dvc_ModelParams::REWARD_FACTOR_VEL = REWARD_FACTOR_VEL  ;
	Dvc_ModelParams::REWARD_BASE_CRASH_VEL =  REWARD_BASE_CRASH_VEL ;
	Dvc_ModelParams::BELIEF_SMOOTHING =   BELIEF_SMOOTHING;
	Dvc_ModelParams::NOISE_ROBVEL =NOISE_ROBVEL;
	Dvc_ModelParams::COLLISION_DISTANCE =  COLLISION_DISTANCE ;
	Dvc_ModelParams::IN_FRONT_ANGLE_DEG = IN_FRONT_ANGLE_DEG  ;
	Dvc_ModelParams::LASER_RANGE =  LASER_RANGE ;
	Dvc_ModelParams::pos_rln =  pos_rln ; // position resolution
	Dvc_ModelParams::vel_rln =  vel_rln ; // velocity resolution
	Dvc_ModelParams::PATH_STEP =  PATH_STEP ;
	Dvc_ModelParams::GOAL_TOLERANCE =  GOAL_TOLERANCE ;
	Dvc_ModelParams::PED_SPEED = PED_SPEED  ;
	Dvc_ModelParams::debug =  debug ;
	Dvc_ModelParams::control_freq =  control_freq ;
	Dvc_ModelParams::AccSpeed =  AccSpeed ;
	Dvc_ModelParams::GOAL_REWARD = GOAL_REWARD  ;


	/*printf("Dvc_ModelParams::GOAL_TRAVELLED=%f\n", Dvc_ModelParams::GOAL_TRAVELLED);
	printf("Dvc_ModelParams::N_PED_IN=%d\n", Dvc_ModelParams::N_PED_IN);
	printf("Dvc_ModelParams::N_PED_WORLD=%d\n", Dvc_ModelParams::N_PED_WORLD);
	printf("Dvc_ModelParams::VEL_MAX=%f\n", Dvc_ModelParams::VEL_MAX);
	printf("Dvc_ModelParams::NOISE_GOAL_ANGLE=%f\n", Dvc_ModelParams::NOISE_GOAL_ANGLE);
	printf("Dvc_ModelParams::CRASH_PENALTY=%f\n", Dvc_ModelParams::CRASH_PENALTY);
	printf("Dvc_ModelParams::REWARD_FACTOR_VEL=%f\n", Dvc_ModelParams::REWARD_FACTOR_VEL);
	printf("Dvc_ModelParams::REWARD_BASE_CRASH_VEL=%f\n", Dvc_ModelParams::REWARD_BASE_CRASH_VEL);
	printf("Dvc_ModelParams::BELIEF_SMOOTHING=%f\n", Dvc_ModelParams::BELIEF_SMOOTHING);
	printf("Dvc_ModelParams::NOISE_ROBVEL=%f\n", Dvc_ModelParams::NOISE_ROBVEL);
	printf("Dvc_ModelParams::COLLISION_DISTANCE=%f\n", Dvc_ModelParams::COLLISION_DISTANCE);
	printf("Dvc_ModelParams::IN_FRONT_ANGLE_DEG=%f\n", Dvc_ModelParams::IN_FRONT_ANGLE_DEG);
	printf("Dvc_ModelParams::LASER_RANGE= %f\n", Dvc_ModelParams::LASER_RANGE);
	printf("Dvc_ModelParams::pos_rln=%f\n", Dvc_ModelParams::pos_rln); // position resolution
	printf("Dvc_ModelParams::vel_rln=%f\n", Dvc_ModelParams::vel_rln); // velocity resolution
	printf("Dvc_ModelParams::PATH_STEP=%f\n", Dvc_ModelParams::PATH_STEP);
	printf("Dvc_ModelParams::GOAL_TOLERANCE=%f\n", Dvc_ModelParams::GOAL_TOLERANCE);
	printf("Dvc_ModelParams::PED_SPEED=%f\n", Dvc_ModelParams::PED_SPEED);
	printf("Dvc_ModelParams::debug=%d\n", Dvc_ModelParams::debug);
	printf("Dvc_ModelParams::control_freq=%f\n", Dvc_ModelParams::control_freq);
	printf("Dvc_ModelParams::AccSpeed=%f\n", Dvc_ModelParams::AccSpeed);
	printf("Dvc_ModelParams::GOAL_REWARD=%f\n", Dvc_ModelParams::GOAL_REWARD);*/
}

void Simulator::UpdateGPUPath(DSPOMDP* Hst_model)
{
	PedPomdp* Hst =static_cast<PedPomdp*>(Hst_model);

	if(tempPath)HANDLE_ERROR(cudaFree(tempPath));
	HANDLE_ERROR(cudaMallocManaged((void**)&tempPath, Hst->world.path.size()*sizeof(Dvc_COORD)));

	for(int i=0;i<Hst->world.path.size();i++){
		tempPath[i].x=Hst->world.path[i].x;
		tempPath[i].y=Hst->world.path[i].y;
	}

	UpdatePathKernel<<<1,1,1>>>(tempPath,Hst->world.path.size());
	HANDLE_ERROR(cudaDeviceSynchronize());
	//exit(-1);
}

void Simulator::UpdateGPUGoals(DSPOMDP* Hst_model)
{
	PedPomdp* Hst =static_cast<PedPomdp*>(Hst_model);
	if(tempGoals)HANDLE_ERROR(cudaFree(tempGoals));
	HANDLE_ERROR(cudaMallocManaged((void**)&tempGoals,  Hst->world.goals.size()*sizeof(Dvc_COORD)));


	for(int i=0;i<Hst->world.goals.size();i++){
		tempGoals[i].x=Hst->world.goals[i].x;
		tempGoals[i].y=Hst->world.goals[i].y;
	}
	UpdateGoalKernel<<<1,1,1>>>(tempGoals);
	HANDLE_ERROR(cudaDeviceSynchronize());

}


void Simulator::InitializedGPUModel(std::string rollout_type, DSPOMDP* Hst_model)
{
	PedPomdp* Hst =static_cast<PedPomdp*>(Hst_model);
	HANDLE_ERROR(cudaMalloc((void**)&Dvc, sizeof(Dvc_PedPomdp)));
	if(rollout_type=="INDEPENDENT")
		  HANDLE_ERROR(cudaMalloc((void**)&smart_lowerbound, sizeof(Dvc_PedPomdpSmartPolicy)));
	if(rollout_type=="GRAPH")
	  HANDLE_ERROR(cudaMalloc((void**)&graph_lowerbound, sizeof(Dvc_PedPomdpSmartPolicyGraph)));

	HANDLE_ERROR(cudaMalloc((void**)&b_lowerbound, sizeof(Dvc_TrivialParticleLowerBound)));
	HANDLE_ERROR(cudaMalloc((void**)&b_smart_lowerbound, sizeof(Dvc_PedPomdpParticleLowerBound)));

	HANDLE_ERROR(cudaMalloc((void**)&upperbound, sizeof(Dvc_PedPomdpParticleUpperBound1)));

	if(tempPath==NULL && Hst->world.path.size()>0)
	HANDLE_ERROR(cudaMallocManaged((void**)&tempPath, Hst->world.path.size()*sizeof(Dvc_COORD)));
	if(tempGoals==NULL && Hst->world.goals.size()>0)
	HANDLE_ERROR(cudaMallocManaged((void**)&tempGoals,  Hst->world.goals.size()*sizeof(Dvc_COORD)));

	
/*	HANDLE_ERROR(cudaMemcpy(tempGoals, Hst->world.goals.data(),
			Hst->world.goals.size()*sizeof(Dvc_COORD), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(tempPath, Hst->world.path.data(),
			Hst->world.path.size()*sizeof(Dvc_COORD), cudaMemcpyHostToDevice));*/
	PassModelParameters<<<1,1,1>>>(
		ModelParams::GOAL_TRAVELLED,
		ModelParams::N_PED_IN,
		ModelParams::N_PED_WORLD,
		ModelParams::VEL_MAX,
		ModelParams::NOISE_GOAL_ANGLE,
		ModelParams::CRASH_PENALTY,
		ModelParams::REWARD_FACTOR_VEL,
		ModelParams::REWARD_BASE_CRASH_VEL,
		ModelParams::BELIEF_SMOOTHING,
		ModelParams::NOISE_ROBVEL,
		ModelParams::COLLISION_DISTANCE,
		ModelParams::IN_FRONT_ANGLE_DEG,
		ModelParams::LASER_RANGE,
		ModelParams::pos_rln, // position resolution
		ModelParams::vel_rln, // velocity resolution
		ModelParams::PATH_STEP,
		ModelParams::GOAL_TOLERANCE,
		ModelParams::PED_SPEED,
		ModelParams::debug,
		ModelParams::control_freq,
		ModelParams::AccSpeed,
		ModelParams::GOAL_REWARD);

	PassModelFuncs<<<1,1,1>>>(Dvc,Hst->world.in_front_angle_cos, Hst->world.freq, tempGoals, tempPath,Hst->world.path.size());
	if(rollout_type=="INDEPENDENT")
	  PassActionValueFuncs<<<1,1,1>>>(Dvc,static_cast<Dvc_PedPomdpSmartPolicy*>(smart_lowerbound),/*b_lowerbound*/b_smart_lowerbound,upperbound);
	if(rollout_type=="GRAPH")
	  PassValueFuncs<<<1,1,1>>>(graph_lowerbound,/*b_lowerbound*/b_smart_lowerbound,upperbound);

	UpdateGPUGoals(Hst_model);

	HANDLE_ERROR(cudaDeviceSynchronize());
	//exit(-1);
}


DSPOMDP* Simulator::InitializeModel(option::Option* options) {

}
void Simulator::DeleteGPUPolicyGraph()
{
}

void Simulator::InitializeDefaultParameters() {

	//Globals::config.time_per_move=0.1;
    //Globals::config.time_per_move = 10;//(1.0/ModelParams::control_freq) * 0.9;
    Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
	Globals::config.num_scenarios=50;
	Globals::config.discount=/*0.983*/0.95/*0.966*/;
	Globals::config.sim_len=1/*180*//*10*/;
	Globals::config.pruning_constant=0.001;

	Globals::config.max_policy_sim_len=/*Globals::config.sim_len+30*/25;

	Globals::config.GPUid=1;//default GPU
	Globals::config.useGPU=true;
	Globals::config.use_multi_thread_=false;
	Globals::config.NUM_THREADS=5;

	Globals::config.exploration_mode=UCT;
	Globals::config.exploration_constant=/*0.095*//*0.1*/0.1;

	Globals::config.silence=true;
	Obs_parallel_level=OBS_PARALLEL_Y;
	Obs_type=OBS_INT_ARRAY;
	DESPOT::num_Obs_element_in_GPU=1+ModelParams::N_PED_IN*2+2;
	switch(FIX_SCENARIO){
	case 0:		Load_Graph=false; break;
	case 1:     Load_Graph=true; break;
	case 2:     Load_Graph=false; break;
	}
	cout<<"Load_Graph="<<Load_Graph<<endl;
}

