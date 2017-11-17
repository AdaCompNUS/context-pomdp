#include <despot/GPUcore/GPUpolicy.h>
#include <despot/GPUcore/GPUpomdp.h>
#include <unistd.h>

#include <despot/GPUutil/GPUmap.h>

using namespace std;

namespace despot {
//DEVICE Dvc_State* Dvc_particles_copy=NULL;
OBS_TYPE** Dvc_observations_long=NULL;
OBS_TYPE** Dvc_obs_all_a_and_p=NULL;
int** Dvc_obs_int_all_a_and_p=NULL;

bool** Dvc_term_all_a_and_p=NULL;
OBS_TYPE** Hst_obs_all_a_and_p=NULL;
int** Hst_obs_int_all_a_and_p=NULL;

bool** Hst_term_all_a_and_p=NULL;
//DEVICE Dvc_History* Dvc_ParticleHistory=NULL;
DEVICE Dvc_Random* Dvc_random=NULL;
DEVICE int (*DvcPolicyAction_)(int,const Dvc_State* ,Dvc_RandomStreams&, Dvc_History&)=NULL;

DEVICE static double* action_probs_=NULL;
DEVICE static int num_actions_=0;
DEVICE int* Dvc_initial_depth=NULL;


DEVICE Dvc_ValuedAction Dvc_Policy::Value(Dvc_State* particles,
	Dvc_RandomStreams& streams, Dvc_History& Local_history) {	//

	Dvc_State* particle = particles;
	int scenarioID=particle->scenario_id;

	__shared__ int all_terminated[32];

	__shared__ int action[32];
	//__shared__ int obs[60*32];//the data is not used, so its ok to share for all threads

	float Accum_Value=0;

	int init_depth=Local_history.currentSize_;

	//if (streams.Exhausted()
	//		|| (history.Size() - initial_depth_
	//			>= Globals::config.max_policy_sim_len))
	int MaxDepth=Dvc_config->max_policy_sim_len+init_depth;
	//int terminal = false;
	int depth;
	int Action_decision=-1;
	int terminal;
	//if(threadIdx.x+threadIdx.y+blockIdx.x+blockIdx.y==0 && init_depth!=1)
	//	printf("gpu: history at %d, stream at pos %d\n",Local_history.currentSize_, streams.position_);
	for(depth=init_depth;(depth<MaxDepth && !streams.Exhausted());depth++)
	{
		if(threadIdx.y==0/*&& threadIdx.x==0*/)
		{
			all_terminated[threadIdx.x]=true;
		}

		int local_action=DvcPolicyAction_(/*scenarioID,Dvc_particles_copy*//*0*/scenarioID,particle, streams, Local_history);
		if(threadIdx.y==0)
		{
			/*if(streams.Entry(scenarioID)==0.383227 && local_action!=1)
				printf("step 32, wrong action at initial depth\n");*/
			action[threadIdx.x] = local_action;
			if(depth==init_depth/* && threadIdx.x==0*/)
				Action_decision=action[threadIdx.x];
		}
		__syncthreads();

		float reward/*=0*/;

		if(DvcModelStep_)
		{
			OBS_TYPE obs/*=-1*/;
			terminal = DvcModelStep_(*particle, streams.Entry(scenarioID), action[threadIdx.x], reward, obs);
		}
		else
		{
			terminal = DvcModelStepIntObs_(*particle, streams.Entry(scenarioID), action[threadIdx.x], reward,NULL/* obs+threadIdx.x*60*/);
		}
		if(threadIdx.y==0)
		{
			atomicAnd(&all_terminated[threadIdx.x],terminal);

			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth+0)* reward/* * particle->weight*/;
		}
		streams.Advance();

		__syncthreads();
		if(all_terminated[threadIdx.x])
		{
			break;
		}

	}
	/*use default value for leaf positions*/
	if(threadIdx.y==0)
	{
		if(!terminal)
		{
			Dvc_ValuedAction va = DvcParticleLowerBound_Value_(/*scenarioID,Dvc_particles_copy*/0,particle);
			/*if(particle->scenario_id==52 && blockIdx.x==0 && threadIdx.y==0){
				printf("Rollout bottom: bottom_value=%f, last discount depth=%d \n", va.value, depth-init_depth+1);
			}*/
			Accum_Value += Dvc_Globals::Dvc_Discount(Dvc_config,depth-init_depth/*+1*/) * va.value;
		}
	}

	/*the value returned here need to be weighted summed to get the real value of the action */

	return Dvc_ValuedAction(Action_decision, Accum_Value);

}

DEVICE Dvc_RandomPolicy::Dvc_RandomPolicy()
{
	//Action_=&(Action);
	action_probs_=NULL;
	num_actions_=0;
}

DEVICE void Dvc_RandomPolicy::Init(int num_actions)
{
	//Action_=&(Action);
	num_actions_=num_actions;
	action_probs_=NULL;
}

DEVICE int Dvc_RandomPolicy::Action(int scenarioID, const Dvc_State* particles,
	Dvc_RandomStreams& streams, Dvc_History& history) {
	if (action_probs_!= NULL) {
		return Dvc_random->GetCategory(num_actions_,action_probs_, Dvc_random->NextDouble(scenarioID));
	} else {
		return Dvc_random->NextInt(num_actions_, scenarioID );
	}
}
/*DEVICE void Dvc_Policy::Reset() {
}*/

/*DEVICE Dvc_ParticleLowerBound* Dvc_Policy::particle_lower_bound() const {
	return particle_lower_bound_;
}*/
/*
Dvc_ValuedAction Dvc_Policy::Search() {
	Dvc_RandomStreams streams(config->num_scenarios,
		config->search_depth);
	vector<Dvc_State*> particles = belief_->Sample(config->num_scenarios);

	int action = Action(particles, streams, history_);
	double dummy_value = Dvc_Globals::NEG_INFTY;

	for (int i = 0; i < NumParticles; i++)
		DvcModelFree_(particles[i]);

	return Dvc_ValuedAction(action, dummy_value);
}*/
/*

 =============================================================================
 * Dvc_BlindPolicy class
 * =============================================================================

Dvc_BlindPolicy::Dvc_BlindPolicy(const Dvc_DSPOMDP* model, int action, Dvc_ParticleLowerBound*
	bound, Dvc_Belief* belief) :
	Dvc_Policy(model, bound, belief),
	action_(action) {
}

int Dvc_BlindPolicy::Action(const vector<Dvc_State*>& particles, Dvc_RandomStreams& streams,
	Dvc_History& history) const {
	return action_;
}

Dvc_ValuedAction Dvc_BlindPolicy::Search() {
	double dummy_value = Dvc_Globals::NEG_INFTY;
	return Dvc_ValuedAction(action_, dummy_value);
}

void Dvc_BlindPolicy::Update(int action, OBS_TYPE obs) {
}

 =============================================================================
 * Dvc_RandomPolicy class
 * =============================================================================

Dvc_RandomPolicy::Dvc_RandomPolicy(const Dvc_DSPOMDP* model, Dvc_ParticleLowerBound* bound,
	Dvc_Belief* belief) :
	Dvc_Policy(model, bound, belief) {
}

Dvc_RandomPolicy::Dvc_RandomPolicy(const Dvc_DSPOMDP* model,
	const vector<double>& action_probs,
	Dvc_ParticleLowerBound* bound, Dvc_Belief* belief) :
	Dvc_Policy(model, bound, belief),
	action_probs_(action_probs) {
	double sum = 0;
	for (int i = 0; i < action_probs.size(); i++)
		sum += action_probs[i];
	assert(fabs(sum - 1.0) < 1.0e-8);
}

int Dvc_RandomPolicy::Action(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	if (action_probs_.size() > 0) {
		return Random::GetCategory(action_probs_, Random::RANDOM.NextDouble());
	} else {
		return Random::RANDOM.NextInt(model_->NumActions());
	}
}

Dvc_ValuedAction Dvc_RandomPolicy::Search() {
	double dummy_value = Dvc_Globals::NEG_INFTY;
	if (action_probs_.size() > 0) {
		return Dvc_ValuedAction(
			Random::GetCategory(action_probs_, Random::RANDOM.NextDouble()),
			dummy_value);
	} else {
		return Dvc_ValuedAction(Random::RANDOM.NextInt(model_->NumActions()),
			dummy_value);
	}
}

void Dvc_RandomPolicy::Update(int action, OBS_TYPE obs) {
}

 =============================================================================
 * Dvc_MajorityActionPolicy class
 * =============================================================================

Dvc_MajorityActionPolicy::Dvc_MajorityActionPolicy(const Dvc_DSPOMDP* model,
	const Dvc_StatePolicy& policy, Dvc_ParticleLowerBound* bound, Dvc_Belief* belief) :
	Dvc_Policy(model, bound, belief),
	policy_(policy) {
}

int Dvc_MajorityActionPolicy::Action(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	vector<double> frequencies(model_->NumActions());

	for (int i = 0; i < NumParticles; i++) {
		Dvc_State* particle = particles[i];
		int action = policy_.GetAction(*particle);
		frequencies[action] += particle->weight;
	}

	int bestAction = 0;
	double bestWeight = frequencies[0];
	for (int a = 1; a < frequencies.size(); a++) {
		if (bestWeight < frequencies[a]) {
			bestWeight = frequencies[a];
			bestAction = a;
		}
	}

	return bestAction;
}

 =============================================================================
 * Dvc_ModeStatePolicy class
 * =============================================================================

Dvc_ModeStatePolicy::Dvc_ModeStatePolicy(const Dvc_DSPOMDP* model,
	const Dvc_StateIndexer& indexer, const Dvc_StatePolicy& policy,
	Dvc_ParticleLowerBound* bound, Dvc_Belief* belief) :
	Dvc_Policy(model, bound, belief),
	indexer_(indexer),
	policy_(policy) {
	state_probs_.resize(indexer_.NumStates());
}

int Dvc_ModeStatePolicy::Action(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	double maxWeight = 0;
	Dvc_State* mode = NULL;
	for (int i = 0; i < NumParticles; i++) {
		Dvc_State* particle = particles[i];
		int id = indexer_.GetIndex(particle);
		state_probs_[id] += particle->weight;

		if (state_probs_[id] > maxWeight) {
			maxWeight = state_probs_[id];
			mode = particle;
		}
	}

	for (int i = 0; i < NumParticles; i++) {
		state_probs_[indexer_.GetIndex(particles[i])] = 0;
	}

	assert(mode != NULL);
	return policy_.GetAction(*mode);
}

 =============================================================================
 * Dvc_MMAPStatePolicy class
 * =============================================================================

Dvc_MMAPStatePolicy::Dvc_MMAPStatePolicy(const Dvc_DSPOMDP* model,
	const Dvc_MMAPInferencer& inferencer, const Dvc_StatePolicy& policy,
	Dvc_ParticleLowerBound* bound, Dvc_Belief* belief) :
	Dvc_Policy(model, bound, belief),
	inferencer_(inferencer),
	policy_(policy) {
}

int Dvc_MMAPStatePolicy::Action(const vector<Dvc_State*>& particles,
	Dvc_RandomStreams& streams, Dvc_History& history) const {
	return policy_.GetAction(*inferencer_.GetMMAP(particles));
}
*/

} // namespace despot
