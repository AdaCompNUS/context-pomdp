#include "GPU_LowerBoundPolicy.h"

#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUrandom_streams.h>
#include <despot/GPUutil/GPUcoord.h>
#include <despot/GPUutil/GPUlimits.h>
#include <despot/GPUcore/GPUpolicy_graph.h>
#include "GPU_Car_Drive.h"
using archaeopteryx::util::numeric_limits;
using despot::Dvc_History;
using despot::Dvc_RandomStreams;
using despot::Dvc_State;

#include <despot/GPUcore/GPUpomdp.h>

DEVICE int Dvc_PedPomdpSmartPolicy::Action(
		int scenarioID, const Dvc_State* particles,
				Dvc_RandomStreams& streams,
				Dvc_History& history)
{
	const Dvc_PomdpState &state=static_cast<const Dvc_PomdpState&>(particles[0]);
	__shared__ int mindist[32];
	//float mindist = numeric_limits<float>::infinity();
	auto& carpos = path->way_points_[state.car.pos];
	float carvel = state.car.vel;
	// Closest pedestrian in front
	//for (int i=0; i<state.num; i++) {
	//infront[threadIdx.x]=false;
	mindist[threadIdx.x]=__float_as_int(numeric_limits<float>::infinity());
	__syncthreads();
	if (threadIdx.y<state.num) {
		auto& p = state.peds[threadIdx.y];
		bool infront=false;
		//bool is_infront=false;

		if(Dvc_ModelParams::IN_FRONT_ANGLE_DEG >= 180.0) {
			// inFront check is disabled in this case
			infront=true;
		}
		else
		{
			const Dvc_COORD& car_pos = path->way_points_[state.car.pos];
			const Dvc_COORD& forward_pos = path->way_points_[path->forward(state.car.pos, 1.0)];
			float d0 = Dvc_COORD::EuclideanDistance(car_pos, p.pos);
			//if(d0<=0) return true;
			//if(p.vel<1e-5 && d0>1.0)
			//	infront=false;//don't consider non-moving peds
			/*else*/ if(d0 <= 0.7/*3.5*/)
				infront=true;
			else
			{
				float d1 = Dvc_COORD::EuclideanDistance(car_pos, forward_pos);
				if(d1<=0)
					infront=true;
				else
				{
					float dot = Dvc_Vector::DotProduct(forward_pos.x - car_pos.x, forward_pos.y - car_pos.y,
							p.pos.x - car_pos.x, p.pos.y - car_pos.y);
					float cosa = dot / (d0 * d1);
					if(cosa > 1.0 + 1E-6 || cosa < -1.0 - 1E-6)
					{
						/*printf("cosa=%f\n", cosa);
						int pos=state.car.pos;
						printf("car pox= %d ",pos);
						printf("dist=%f\n",state.car.dist_travelled);
						printf("car vel= %f\n",state.car.vel);

						for(int i=0;i<state.num;i++)
						{
							printf("ped %d pox_x= %f pos_y=%f\n",i,
									state.peds[i].pos.x,state.peds[i].pos.y);
						}*/
						printf("%s=%f\n", "cosa", cosa);
					}
					//assert(cosa <= 1.0 + 1E-8 && cosa >= -1.0 - 1E-8);
					infront=cosa > in_front_angle_cos;
				}
			}
		}

		if(infront) {
			float d = Dvc_COORD::EuclideanDistance(carpos, p.pos);
			// cout << d << " " << carpos.x << " " << carpos.y << " "<< p.pos.x << " " << p.pos.y << endl;
			atomicMin(mindist+threadIdx.x, __float_as_int(d));
			//if (d >= 0 && d < mindist)
			//	mindist = d;
		}
	}
	__syncthreads();

	// cout << "min_dist = " << mindist << endl;
	/*if(GPUDoPrint && state.scenario_id==GPUPrintPID && blockIdx.x==0 && threadIdx.y==0){
		float dist=__int_as_float(mindist[threadIdx.x]);
		printf("mindist, carvel= %f %f\n",dist,carvel);
	}*/
	// TODO set as a param
	if (__int_as_float(mindist[threadIdx.x]) < 2/*3.5*/) {
		return (carvel <= 0.01) ? 0 : 2;
	}

	if (__int_as_float(mindist[threadIdx.x]) < 4/*5*/) {
		if (carvel > 1.0+1e-4) return 2;
		else if (carvel < 0.5-1e-4) return 1;
		else return 0;
	}
	return carvel >= Dvc_ModelParams::VEL_MAX-1e-4 ? 0 : 1;
}

enum {
		CLOSE_STATIC,
		CLOSE_MOVING,
		MEDIUM_FAST,
		MEDIUM_MED,
		MEDIUM_SLOW,
		FAR_MAX,
		FAR_NOMAX,
	};
__device__ int graph_entry_node=0;

DEVICE Dvc_ValuedAction Dvc_PedPomdpSmartPolicyGraph::Value(Dvc_State* particles,
	Dvc_RandomStreams& streams, Dvc_History& Local_history) {
	//Dvc_State* particle = particles;
	const Dvc_PomdpState *particle=static_cast<const Dvc_PomdpState*>(particles);
	__shared__ int mindist[32];
	__shared__ int current_node_[32];
	__shared__ int terminated[32];
	__shared__ int action[32];

	if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
	{
		if(FIX_SCENARIO)
			graph_entry_node=0;//Debug
		else
			graph_entry_node=0;
	}
	__syncthreads();
	if(threadIdx.y==0)
	{
		current_node_[threadIdx.x]=graph_entry_node;
		//mindist[threadIdx.x]=__float_as_int(numeric_limits<float>::infinity());
	}
	float Accum_Value=0;

	int init_depth=Local_history.currentSize_;


	int MaxDepth=min(Dvc_config->max_policy_sim_len+init_depth,streams.Length());
	int depth;
	int Action_decision=action_nodes_[graph_entry_node];
	int terminal=false;
	/*if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){
		printf("Initial Action_decision=%d",Action_decision);
	}*/
	for(depth=init_depth-1;depth<MaxDepth;depth++)
	{
		if(threadIdx.y==0)
		{
			terminated[threadIdx.x]=false;
		}
		terminal=false;
		if(threadIdx.y==0)
		{
			action[threadIdx.x] = action_nodes_[current_node_[threadIdx.x]];
			mindist[threadIdx.x]=__float_as_int(numeric_limits<float>::infinity());

			if(depth==init_depth)
				Action_decision=action[threadIdx.x];
		}


		__syncthreads();


		if(depth>=init_depth)
		{
			float reward;

			terminal = DvcModelStepIntObs_(*(Dvc_State*)particle, streams.Entry(particle->scenario_id),
					action[threadIdx.x], reward, NULL);

			if(threadIdx.y==0)
			{
				atomicOr(&terminated[threadIdx.x],terminal);

				Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+0)* reward;//particle->weight;

				/*if(blockIdx.x==0 && particle->scenario_id==418 )
					printf("terminal,reward, value, depth, init_depth=%d %f %f %d %d \n"
						,terminal,reward, Accum_Value,depth,init_depth);*/
			}
			streams.Advance();

			/*if(particle->scenario_id==418 && blockIdx.x+threadIdx.y==0){
				int act=action[threadIdx.x];
				printf("Act at depth %d = %d, get reward %f\n",depth,act,reward);
			}*/
		}

		auto& carpos = path->way_points_[particle->car.pos];
		float carvel = particle->car.vel;
		if (threadIdx.y<particle->num) {
			auto& p = particle->peds[threadIdx.y];
			bool infront=false;
			//bool is_infront=false;
			if(Dvc_ModelParams::IN_FRONT_ANGLE_DEG >= 180.0) {
				// inFront check is disabled
				infront=true;
			}
			else
			{
				const Dvc_COORD& car_pos = path->way_points_[particle->car.pos];
				const Dvc_COORD& forward_pos = path->way_points_[path->forward(particle->car.pos, 1.0)];
				float d0 = Dvc_COORD::EuclideanDistance(car_pos, p.pos);
				//if(d0<=0) return true;
				//if(p.vel<1e-5 && d0>1.0)
				//	infront=false;//don't consider non-moving peds
				/*else*/ if(d0 <= /*0.7*/3.5)
					infront=true;
				else
				{
					float d1 = Dvc_COORD::EuclideanDistance(car_pos, forward_pos);
					if(d1<=0)
						infront=true;
					else
					{
						float dot = Dvc_Vector::DotProduct(forward_pos.x - car_pos.x, forward_pos.y - car_pos.y,
								p.pos.x - car_pos.x, p.pos.y - car_pos.y);
						float cosa = dot / (d0 * d1);
						assert(cosa <= 1.0 + 1E-8 && cosa >= -1.0 - 1E-8);
						infront=cosa > in_front_angle_cos;
					}
				}
			}

			if(infront) {
				float d = Dvc_COORD::EuclideanDistance(carpos, p.pos);
				atomicMin(mindist+threadIdx.x, __float_as_int(d));
			}
		}
		__syncthreads();

		if(threadIdx.y==0)
		{
			int edge;
			// TODO set as a param
			if (__int_as_float(mindist[threadIdx.x]) < /*2*/3.5) {
				edge= (carvel <= 0.01) ? CLOSE_STATIC : CLOSE_MOVING;
			}
			else if (__int_as_float(mindist[threadIdx.x]) < 5) {
				if (carvel > 1.0+1e-4) edge= MEDIUM_FAST;
				else if (carvel < 0.5-1e-4) edge= MEDIUM_SLOW;
				else edge= MEDIUM_MED;
			}
			else
				edge= (carvel >= Dvc_ModelParams::VEL_MAX-1e-4) ? FAR_MAX : FAR_NOMAX;

			/*if(particle->scenario_id==418 && blockIdx.x+threadIdx.y==0){
				float dist=__int_as_float(mindist[threadIdx.x]);
				printf("mindist, carvel= %f %f Trace edge %d\n",dist,carvel,edge);
			}*/

			current_node_[threadIdx.x]=obs_edges_[edge*graph_size_+current_node_[threadIdx.x]];
		}
		__syncthreads();
		if(terminated[threadIdx.x])
		{
			break;
		}
	}
	//use default value for leaf positions
	if(threadIdx.y==0)
	{
		if(!terminal)
		{
			Dvc_ValuedAction va = DvcParticleLowerBound_Value_(0,(Dvc_State*)particle);
			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+1) * va.value;
			/*if( blockIdx.x==0 && particle->scenario_id==418 )
				printf("terminal,va.value, depth, init_depth=%d %f %d %d \n"
						,terminal,Accum_Value,depth,init_depth);*/
		}
	}

	//the value returned here need to be weighted summed to get the real value of the action
	return Dvc_ValuedAction(Action_decision, Accum_Value);
}

/*enum {
		ACT_CUR,
		ACT_ACC,
		ACT_DEC
	};*/

/*DEVICE Dvc_ValuedAction Dvc_PedPomdpParticleLowerBound::Value(
		int scenarioID, Dvc_State * particles) {
	//Dvc_ValuedAction va = DvcModelGetMinRewardAction_();
	//va.value *= 1.0 / (1 - Dvc_Globals::Dvc_Discount(Dvc_config));
	//return va;

	Dvc_PomdpState* state = static_cast<Dvc_PomdpState*>(particles);
	int min_step = numeric_limits<int>::max();
	auto& carpos = path->way_points_[state->car.pos];
	float carvel = state->car.vel;

	// Find mininum num of steps for car-pedestrian collision
	for (int i=0; i<state->num; i++) {
		auto& p = state->peds[i];
		// 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
		int step = int(ceil(Dvc_ModelParams::control_freq
					* max(Dvc_COORD::EuclideanDistance(carpos, p.pos) - 3.25, 0.0)
					/ ((p.vel + carvel))));
		min_step = min(step, min_step);
	}

	//double move_penalty = ped_pomdp_->MovementPenalty(*state);
	float value = Dvc_ModelParams::REWARD_FACTOR_VEL *
						(state->car.vel - Dvc_ModelParams::VEL_MAX) / Dvc_ModelParams::VEL_MAX;

	// Case 1, no pedestrian: Constant car speed
	value = value / (1 - Dvc_Globals::Dvc_Discount(Dvc_config));
	// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
	if (min_step != numeric_limits<int>::max()) {
		//double crash_penalty = ped_pomdp_->CrashPenalty(*state);
		value = (value) * (1 - Dvc_Globals::Dvc_Discount(Dvc_config,min_step))
				/ (1 - Dvc_Globals::Dvc_Discount(Dvc_config));
		value= value + Dvc_ModelParams::CRASH_PENALTY *
				(state->car.vel * state->car.vel + Dvc_ModelParams::REWARD_BASE_CRASH_VEL)
						* Dvc_Globals::Dvc_Discount(Dvc_config,min_step);
	}

	return Dvc_ValuedAction(ACT_CUR,  value);
}*/
