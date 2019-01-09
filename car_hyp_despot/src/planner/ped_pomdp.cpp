
#include <limits>
#include <despot/GPUcore/thread_globals.h>
#include <despot/interface/default_policy.h>
#include <despot/core/builtin_policy.h>
#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/builtin_upper_bounds.h>
#include <despot/solver/despot.h>

#include "custom_particle_belief.h"

#include <despot/core/prior.h>

#include "ped_pomdp.h"

#include "neural_prior.h"



#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

static std::map<uint64_t, std::vector<int>> Obs_hash_table;
static PomdpState hashed_state;

class PedPomdpParticleLowerBound : public ParticleLowerBound {
private:
	const PedPomdp* ped_pomdp_;
public:
	PedPomdpParticleLowerBound(const DSPOMDP* model) :
		ParticleLowerBound(model),
		ped_pomdp_(static_cast<const PedPomdp*>(model))
	{
	}

	// IMPORTANT: Check after changing reward function.
	virtual ValuedAction Value(const std::vector<State*>& particles) const {
		PomdpState* state = static_cast<PomdpState*>(particles[0]);
		int min_step = numeric_limits<int>::max();
		auto& carpos = state->car.pos;
		double carvel = state->car.vel;

		// Find mininum num of steps for car-pedestrian collision
		for (int i = 0; i < state->num; i++) {
			auto& p = state->peds[i];
			// 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
			int step = (p.vel + carvel <= 1e-5) ? min_step : int(ceil(ModelParams::control_freq
			           * max(COORD::EuclideanDistance(carpos, p.pos) - /*1.0*/3.25, 0.0)
			           / ((p.vel + carvel))));

			if(DoPrintCPU)
				printf("   step,min_step, p.vel + carvel=%d %d %f\n",step,min_step, p.vel + carvel);
			min_step = min(step, min_step);
		}

		double move_penalty = ped_pomdp_->MovementPenalty(*state);

		// Case 1, no pedestrian: Constant car speed
		double value = move_penalty / (1 - Globals::Discount());
		// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
		if (min_step != numeric_limits<int>::max()) {
			double crash_penalty = ped_pomdp_->CrashPenalty(*state);
			value = (move_penalty) * (1 - Globals::Discount(min_step)) / (1 - Globals::Discount())
			        + crash_penalty * Globals::Discount(min_step);
			if(DoPrintCPU)
				printf("   min_step,crash_penalty, value=%d %f %f\n"
						,min_step, crash_penalty,value);
		}

		if(DoPrintCPU)
			printf("   min_step,num_peds,move_penalty, value=%d %d %f %f\n"
					,min_step,state->num, move_penalty,value);
		return ValuedAction(ped_pomdp_->ACT_CUR, State::Weight(particles) * value);
	}
};

void PedPomdp::InitRVOSetting() {
	use_rvo_in_search = true;
	use_rvo_in_simulation = true;
//	if (use_rvo_in_simulation)
//		ModelParams::LASER_RANGE = 8.0;
//	else
//		ModelParams::LASER_RANGE = 8.0;
}

PedPomdp::PedPomdp(WorldModel &model_) :
	world_model(&model_),
	random_(Random((unsigned) Seeds::Next()))
{
	InitRVOSetting();
}

PedPomdp::PedPomdp() :
	world_model(NULL),
	random_(Random((unsigned) Seeds::Next()))
{
	InitRVOSetting();
}

const std::vector<int>& PedPomdp::ObserveVector(const State& state_) const {
	const PomdpState &state = static_cast<const PomdpState&>(state_);
	static std::vector<int> obs_vec;

	obs_vec.resize(state.num * 2 + 3);

	int i = 0;
	obs_vec[i++] = int(state.car.pos.x / ModelParams::pos_rln);
	obs_vec[i++] = int(state.car.pos.y / ModelParams::pos_rln);

	obs_vec[i++] = int((state.car.vel + 1e-5) / ModelParams::vel_rln); //add some noise to make 1.5/0.003=50

	for (int j = 0; j < state.num; j ++) {
		obs_vec[i++] = int(state.peds[j].pos.x / ModelParams::pos_rln);
		obs_vec[i++] = int(state.peds[j].pos.y / ModelParams::pos_rln);
	}

	return obs_vec;
}

uint64_t PedPomdp::Observe(const State& state) const {
	hash<std::vector<int>> myhash;
	return myhash(ObserveVector(state));
}

std::vector<State*> PedPomdp::ConstructParticles(std::vector<PomdpState> & samples) const {
	int num_particles = samples.size();
	std::vector<State*> particles;
	for (int i = 0; i < samples.size(); i++) {
		PomdpState* particle = static_cast<PomdpState*>(Allocate(-1, 1.0 / num_particles));
		(*particle) = samples[i];
		particle->SetAllocated();
		particle->weight = 1.0 / num_particles;
		particles.push_back(particle);
	}

	return particles;
}

// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpState& state) const { // , int closest_ped, double closest_dist) const {
	// double ped_vel = state.ped[closest_ped].vel;
	return ModelParams::CRASH_PENALTY * (state.car.vel * state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL);
}
// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpStateWorld& state) const { // , int closest_ped, double closest_dist) const {
	// double ped_vel = state.ped[closest_ped].vel;
	return ModelParams::CRASH_PENALTY * (state.car.vel * state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL);
}

// Avoid frequent dec or acc
double PedPomdp::ActionPenalty(int action) const {
//	return (action == ACT_DEC || action == ACT_ACC) ? -0.1 : 0.0;
	return (action == ACT_DEC) ? -0.1 : 0.0;
//	return 0.0;
}

// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpState& state) const {
//	return ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
	return -ModelParams::TIME_REWARD;
}
// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpStateWorld& state) const {
//	return ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
	return -ModelParams::TIME_REWARD;
}

bool PedPomdp::Step(State& state_, double rNum, int action, double& reward, uint64_t& obs) const {
	PomdpState& state = static_cast<PomdpState&>(state_);
	reward = 0.0;

	////// NOTE: Using true random number to make results in different qnodes different ////
	rNum = Random::RANDOM.NextDouble();

//	printf("step: scenario%d \n", state_.scenario_id);
//	printf("action= %d \n",action);
//	PomdpState* pedpomdp_state=static_cast<PomdpState*>(&state_);
//	printf("car_pos= %f,%f",pedpomdp_state->car.pos.x, pedpomdp_state->car.pos.y);
//	printf("car_heading=%f\n",pedpomdp_state->car.heading_dir);
//	printf("car_vel= %f\n",pedpomdp_state->car.vel);
//	for(int i=0;i<3;i++)
//	{
//		printf("ped %d pox_x= %f pos_y=%f goal=%d\n",i,
//				pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y,pedpomdp_state->peds[i].goal);
//	}

	if(FIX_SCENARIO==1 || DESPOT::Print_nodes){
		if(CPUDoPrint && state_.scenario_id==CPUPrintPID){
			printf("(CPU) Before step: scenario%d \n", state_.scenario_id);
			printf("action= %d \n",action);
			PomdpState* pedpomdp_state=static_cast<PomdpState*>(&state_);
			printf("Before step:\n");
			printf("car_pos= %f,%f",pedpomdp_state->car.pos.x, pedpomdp_state->car.pos.y);
			printf("car_heading=%f\n",pedpomdp_state->car.heading_dir);
			printf("car_vel= %f\n",pedpomdp_state->car.vel);
			for(int i=0;i<pedpomdp_state->num;i++)
			{
				printf("ped %d pox_x= %f pos_y=%f\n",i,
						pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
			}
		}
	}
	// Terminate upon reaching goal
	if (world_model->isGlobalGoal(state.car)) {
        reward = ModelParams::GOAL_REWARD;

        logd << "assigning goal reward " << reward << endl;
		return true;
	}

 	// Safety control: collision; Terminate upon collision
    if(state.car.vel > 0.001 && world_model->inCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state);

        logd << "assigning collision reward" << reward << endl;
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(GetAccelerationID(action));

	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);

	// State transition

	//use rNum directly to keep consistent with GPU codes
	if (Globals::config.use_multi_thread_)
	{
		QuickRandom::SetSeed(INIT_QUICKRANDSEED, Globals::MapThread(this_thread::get_id()));
	}
	else
		QuickRandom::SetSeed(INIT_QUICKRANDSEED, 0);

	logd << "[PedPomdp::"<<__FUNCTION__<<"] Refract action"<< endl;
	double acc = GetAcceleration(action);
	double steering = GetSteering(action);

//	cout << "car freq " << world_model->freq << endl;

	world_model->RobStep(state.car, steering, rNum/*random*/);

	world_model->RobVelStep(state.car, acc, rNum/*random*/);


	state.time_stamp = state.time_stamp + 1.0/ModelParams::control_freq;

	if(use_rvo_in_search){
		// Attentive pedestrians
		world_model->RVO2PedStep(state.peds,rNum, state.num, state.car);
		for(int i=0;i<state.num;i++){
			//Distracted pedestrians
			if(state.peds[i].mode==PED_DIS)
				world_model->PedStep(state.peds[i], rNum);
		}
	}
	else{
		for(int i=0;i<state.num;i++)
		{
			world_model->PedStep(state.peds[i], rNum/*random*/);

			assert(state.peds[i].pos.x==state.peds[i].pos.x);//debugging
		}
	}

	if(CPUDoPrint && state.scenario_id==CPUPrintPID){
		if(true){
			PomdpState* pedpomdp_state=static_cast<PomdpState*>(&state_);
			printf("(CPU) After step: scenario=%d \n", pedpomdp_state->scenario_id);
			printf("rand=%f, action=%d \n", rNum, action);

			printf("After step:\n");
			printf("Reward=%f\n",reward);

			printf("car_pos= %f,%f",pedpomdp_state->car.pos.x, pedpomdp_state->car.pos.y);
			printf("car_heading=%f\n",pedpomdp_state->car.heading_dir);
			printf("car vel= %f\n",pedpomdp_state->car.vel);
			for(int i=0;i<pedpomdp_state->num;i++)
			{
				printf("ped %d pox_x= %f pos_y=%f\n",i,
						pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
			}
		}
	}

//	printf("after step \n");
//	printf("car_pos= %f,%f",pedpomdp_state->car.pos.x, pedpomdp_state->car.pos.y);
//	printf("car_heading=%f\n",pedpomdp_state->car.heading_dir);
//	printf("car_vel= %f\n",pedpomdp_state->car.vel);
//	for(int i=0;i<3;i++)
//	{
//		printf("ped %d pox_x= %f pos_y=%f goal=%d\n",i,
//				pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y,pedpomdp_state->peds[i].goal);
//	}

	// Observation
	obs = Observe(state);
	return false;
}

bool PedPomdp::Step(PomdpStateWorld& state, double rNum, int action, double& reward, uint64_t& obs) const {

	reward = 0.0;

	if (world_model->isGlobalGoal(state.car)) {
        reward = ModelParams::GOAL_REWARD;
		return true;
	}

    if(state.car.vel > 0.001 && world_model->inRealCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state);
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(GetAccelerationID(action));

	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);

	// State transition
	Random random(rNum);
	logd << "[PedPomdp::"<<__FUNCTION__<<"] Refract action"<< endl;
	double acc = GetAcceleration(action);
	double steering = GetSteering(action);

	world_model->RobStep(state.car, steering, random);
	world_model->RobVelStep(state.car, acc, random);

	if(use_rvo_in_simulation){
		// Attentive pedestrians
		world_model->RVO2PedStep(state,random);
		// Distracted pedestrians
		for(int i=0;i<state.num;i++){
			if(state.peds[i].mode==PED_DIS)
				world_model->PedStep(state.peds[i], random);
		}
	}
	else{
		for(int i=0;i<state.num;i++)
			world_model->PedStep(state.peds[i], random);
	}
	return false;
}


double PedPomdp::ObsProb(uint64_t obs, const State& s, int action) const {
	return obs == Observe(s);
}

std::vector<std::vector<double>> PedPomdp::GetBeliefVector(const std::vector<State*> particles) const {
	std::vector<std::vector<double>> belief_vec;
	return belief_vec;
}

Belief* PedPomdp::InitialBelief(const State* state, string type) const {

	//Uniform initial distribution
	std::vector<State*> empty; 

	empty.resize(Globals::config.num_scenarios);
	for (int i =0;i< empty.size();i++){
		empty[i] = memory_pool_.Allocate();
		empty[i]->SetAllocated();
		empty[i]->weight = 1.0/empty.size();
	}

	PedPomdpBelief* belief=new PedPomdpBelief(empty , this);

	// const PomdpStateWorld* world_state=static_cast<const PomdpStateWorld*>(state);
	// belief->DeepUpdate(world_state);

	return belief;
}

/// output the probability of the intentions of the pedestrians
void PedPomdp::Statistics(const std::vector<PomdpState*> particles) const {
	return;
	double goal_count[10][10] = {{0}};
	cout << "Current Belief" << endl;
	if (particles.size() == 0)
		return;

	PrintState(*particles[0]);
	PomdpState* state_0 = particles[0];
	for (int i = 0; i < particles.size(); i ++) {
		PomdpState* state = particles[i];
		for (int j = 0; j < state->num; j ++) {
			goal_count[j][state->peds[j].goal] += particles[i]->weight;
		}
	}

	for (int j = 0; j < state_0->num; j ++) {
		cout << "Ped " << j << " Belief is ";
		for (int i = 0; i < world_model->goals.size(); i ++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}
}



ValuedAction PedPomdp::GetBestAction() const {
	return ValuedAction(0,
	                    ModelParams::CRASH_PENALTY * (ModelParams::VEL_MAX * ModelParams::VEL_MAX + ModelParams::REWARD_BASE_CRASH_VEL));
}

class PedPomdpSmartScenarioLowerBound : public DefaultPolicy {
protected:
	const PedPomdp* ped_pomdp_;

public:
	PedPomdpSmartScenarioLowerBound(const DSPOMDP* model, ParticleLowerBound* bound) :
		DefaultPolicy(model, bound),
		ped_pomdp_(static_cast<const PedPomdp*>(model))
	{
	}

	int Action(const std::vector<State*>& particles,
	           RandomStreams& streams, History& history) const {
		return ped_pomdp_->world_model->defaultPolicy(particles);
	}
};


ScenarioLowerBound* PedPomdp::CreateScenarioLowerBound(string name,
        string particle_bound_name) const {
	name = "SMART";
	ScenarioLowerBound* lb;
	if (name == "TRIVIAL") {
		lb = new TrivialParticleLowerBound(this);
	} else if (name == "RANDOM") {
		lb = new RandomPolicy(this, new PedPomdpParticleLowerBound(this));
	} else if (name == "SMART") {
		Globals::config.rollout_type = "INDEPENDENT";
		cout << "[LowerBound] Smart policy independent rollout" << endl;
		lb = new PedPomdpSmartScenarioLowerBound(this, new PedPomdpParticleLowerBound(this));
	} else {
		cerr << "[LowerBound] Unsupported scenario lower bound: " << name << endl;
		exit(0);
	}

	if (Globals::config.useGPU)
		InitGPULowerBound(name, particle_bound_name);

	return lb;
}

double PedPomdp::GetMaxReward() const {
	return 0;
}

class PedPomdpSmartParticleUpperBound : public ParticleUpperBound {
protected:
	const PedPomdp* ped_pomdp_;
public:
	PedPomdpSmartParticleUpperBound(const DSPOMDP* model) :
		ped_pomdp_(static_cast<const PedPomdp*>(model))
	{
	}

	// IMPORTANT: Check after changing reward function.
	double Value(const State& s) const {
		const PomdpState& state = static_cast<const PomdpState&>(s);

		if (ped_pomdp_->world_model->inCollision(state))
			return ped_pomdp_->CrashPenalty(state);

		int min_step = ped_pomdp_->world_model->minStepToGoal(state);
		return ModelParams::GOAL_REWARD * Globals::Discount(min_step);
	}
};


ParticleUpperBound* PedPomdp::CreateParticleUpperBound(string name) const {
	name = "SMART";
	if (name == "TRIVIAL") {
		return new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		return new PedPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "Unsupported particle upper bound: " << name << endl;
		exit(0);
	}
}

ScenarioUpperBound* PedPomdp::CreateScenarioUpperBound(string name,
        string particle_bound_name) const {
	//name = "SMART";
	name = "TRIVIAL";
	ScenarioUpperBound* ub;
	if (name == "TRIVIAL") {
		cout << "[UpperBound] Trivial upper bound" << endl;
		ub =  new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		cout << "[UpperBound] Smart upper bound" << endl;
		ub = new PedPomdpSmartParticleUpperBound(this);
	}
	else {
		cerr << "[UpperBound] Unsupported scenario upper bound: " << name << endl;
		exit(0);
	}
	if (Globals::config.useGPU)
		InitGPUUpperBound(name,	particle_bound_name);
	return ub;
}

void PedPomdp::PrintState(const State& s, string msg, ostream& out) const {

	if (DESPOT::Debug_mode)
		return;

	if (msg == "")
		out << "Search state:\n";
	else
		cout << msg << endl;

	PrintState(s, out);
}


void PedPomdp::PrintState(const State& s, ostream& out) const {

	if (DESPOT::Debug_mode)
		return;

	out << "Address: " << &s << endl;

	if (static_cast<const PomdpStateWorld*> (&s)!=NULL){

		if (static_cast<const PomdpStateWorld*> (&s)->num > ModelParams::N_PED_IN){
			PrintWorldState(static_cast<const PomdpStateWorld&> (s), out);
			return;
		}
	}

	const PomdpState & state = static_cast<const PomdpState&> (s);
	auto& carpos = state.car.pos;

	out << "Distance to goal: " << (world_model->car_goal-carpos).Length() << endl;

	out << "car pos / heading / vel = " << "(" << carpos.x << ", " << carpos.y << ") / "
	    << state.car.heading_dir << " / "
	    << state.car.vel << endl;
	out << state.num << " pedestrians " << endl;
	for (int i = 0; i < state.num; i ++) {
		out << "ped " << i << ": id / pos / vel / goal / dist2car / infront =  " << state.peds[i].id << " / "
		    << "(" << state.peds[i].pos.x << ", " << state.peds[i].pos.y << ") / "
		    << state.peds[i].vel << " / "
		    << state.peds[i].goal << " / "
		    << COORD::EuclideanDistance(state.peds[i].pos, carpos) << "/"
		    << world_model->inFront(state.peds[i].pos, state.car) << endl;
	}
	double min_dist = -1;
	if (state.num > 0)
		min_dist = COORD::EuclideanDistance(carpos, state.peds[0].pos);
	out << "MinDist: " << min_dist << endl;
}

void PedPomdp::PrintWorldState(const PomdpStateWorld& state, ostream& out) const {

	out << "World state:\n";
	auto& carpos = state.car.pos;

	out << "Distance to goal: " << (world_model->car_goal-carpos).Length() << endl;

	out << "car pos / heading / vel = " << "(" << carpos.x << ", " << carpos.y << ") / "
	    << state.car.heading_dir << " / "
	    << state.car.vel << endl;
	out << state.num << " pedestrians " << endl;
	int mindist_id = 0;
	double min_dist = std::numeric_limits<int>::max();

	for (int i = 0; i < state.num; i ++) {
		if (COORD::EuclideanDistance(state.peds[i].pos, carpos) < min_dist)
		{
			min_dist = COORD::EuclideanDistance(state.peds[i].pos, carpos);
			mindist_id = i;
		}
		out << "ped " << i << ": id / pos / vel / goal / dist2car / infront =  " << state.peds[i].id << " / "
		    << "(" << state.peds[i].pos.x << ", " << state.peds[i].pos.y << ") / "
		    << state.peds[i].vel << " / "
		    << state.peds[i].goal << " / "
		    << COORD::EuclideanDistance(state.peds[i].pos, carpos) << "/"
		    << world_model->inFront(state.peds[i].pos, state.car)
		    << " (mode) " << state.peds[i].mode << endl;
	}
	if (state.num > 0)
		min_dist = COORD::EuclideanDistance(carpos, state.peds[/*0*/mindist_id].pos);
	out << "MinDist: " << min_dist << endl;
}
void PedPomdp::PrintObs(const State&state, uint64_t obs, ostream& out) const {
	out << obs << endl;
}

void PedPomdp::PrintAction(int action, ostream& out) const {
	out << action << endl;
}

void PedPomdp::PrintBelief(const Belief& belief, ostream& out ) const {

}

/// output the probability of the intentions of the pedestrians
void PedPomdp::PrintParticles(const vector<State*> particles, ostream& out) const {
	cout << "Particles for planning:" << endl;
	double goal_count[ModelParams::N_PED_IN][10] = {{0}};
	double q_goal_count[ModelParams::N_PED_IN][10] = {{0}}; //without weight, it is q;

	double type_count[ModelParams::N_PED_IN][PED_DIS+1] = {{0}};

	double q_single_weight;
	q_single_weight = 1.0 / particles.size();
	cout << "Current Belief with " << particles.size() << " particles" << endl;
	if (particles.size() == 0)
		return;
	const PomdpState* pomdp_state = static_cast<const PomdpState*>(particles.at(0));

	if(DESPOT::Debug_mode){
		DESPOT::Debug_mode = false;
		PrintState(*pomdp_state);
		DESPOT::Debug_mode = true;
	}

	for (int i = 0; i < particles.size(); i ++) {
		//PomdpState* state = particles[i];
		const PomdpState* pomdp_state = static_cast<const PomdpState*>(particles.at(i));
		for (int j = 0; j < pomdp_state->num; j ++) {
			goal_count[j][pomdp_state->peds[j].goal] += particles[i]->weight;
			q_goal_count[j][pomdp_state->peds[j].goal] += q_single_weight;
			type_count[j][pomdp_state->peds[j].mode] += particles[i]->weight;
		}
	}

	cout << "ped 0 vel: " << pomdp_state->peds[0].vel << endl;

	for (int j = 0; j < 6; j ++) {
		cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
		for (int i = 0; i < world_model->goals.size(); i ++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
		for (int i = 0; i < PED_DIS + 1; i ++) {
			cout << (type_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}

	logd << "<><><> q:" << endl;
	for (int j = 0; j < 6; j ++) {
		logd << "Ped " << pomdp_state->peds[j].id << " Belief is ";
		for (int i = 0; i < world_model->goals.size(); i ++) {
			logd << (q_goal_count[j][i] + 0.0) << " ";
		}
		logd << endl;
	}

	cout << "******** end of scenario belief********" << endl;

}

State* PedPomdp::Allocate(int state_id, double weight) const {
	//num_active_particles ++;
	PomdpState* particle = memory_pool_.Allocate();
	particle->state_id = state_id;
	particle->weight = weight;
	return particle;
}

State* PedPomdp::Copy(const State* particle) const {
	PomdpState* new_particle = memory_pool_.Allocate();
	*new_particle = *static_cast<const PomdpState*>(particle);

	new_particle->SetAllocated();
	return new_particle;
}

void PedPomdp::Free(State* particle) const {
	//num_active_particles --;
	memory_pool_.Free(static_cast<PomdpState*>(particle));
}


int PedPomdp::NumActiveParticles() const {
	return memory_pool_.num_allocated();
}

double PedPomdp::ImportanceScore(PomdpState* state, ACT_TYPE last_action) const
{
	double score = 1.8; //0.3 * 6; 0.3 basic score for each pedestrian
	for(int i=0;i<state->num;i++){
		PedStruct ped = state -> peds[i];
		CarStruct car = state -> car;
		COORD ped_pos = ped.pos;

		const COORD& goal = world_model->goals[ped.goal];
		double move_dw, move_dh;
		if (goal.x == -1 && goal.y == -1) {  //stop intention
			move_dw = 0; //stop intention does not change ped pos, hence movement is 0;
			move_dh = 0;
		}
		else {
			MyVector goal_vec(goal.x - ped_pos.x, goal.y - ped_pos.y);
			double a = goal_vec.GetAngle();
			MyVector move(a, ped.vel*1.0, 0); //movement in unit time
			move_dw = move.dw;
			move_dh = move.dh;
		}

		int count = 0;

		for(int t=1; t<=5; t++){
			ped_pos.x += move_dw;
			ped_pos.y += move_dh;

			Random random(double(state->scenario_id));
			int step = car.vel/ModelParams::control_freq;
			for (int i=0;i<step;i++){
				world_model->RobStep(state->car, GetSteering(last_action), random);
				world_model->RobVelStep(state->car, GetAcceleration(last_action), random);
			}

		   /* double dist = car.vel; // car.vel * 1;
			int nxt = world_model->path.forward(car.pos, dist);
			car.pos = nxt;*/

			double d = COORD::EuclideanDistance(car.pos, ped_pos);

			if(d <= 1 && count < 3) {count ++; score += 4;}
			else if(d <= 2 && count < 3) {count ++; score += 2;}
			else if(d <= 3 && count < 3) {count ++; score += 1;}
		}
	}

	return score;
}

std::vector<double> PedPomdp::ImportanceWeight(std::vector<State*> particles, ACT_TYPE last_action) const
{
	double total_weight = State::Weight(particles);
	double new_total_weight = 0;
	int particles_num = particles.size();

	std::vector<PomdpState*> pomdp_state_particles;

	std::vector <double> importance_weight;

	bool use_is_despot = true;
	if (use_is_despot == false) {
		for (int i = 0; i < particles_num; i++) {
			importance_weight.push_back(particles[i]->weight);
		}
		return importance_weight;
	}

	cout << "use importance sampling ***** " << endl;

	for (int i = 0; i < particles_num; i++) {
		pomdp_state_particles.push_back(static_cast<PomdpState*>(particles[i]));
	}

	for (int i = 0; i < particles_num; i++) {

		importance_weight.push_back(pomdp_state_particles[i]->weight * ImportanceScore(pomdp_state_particles[i], last_action));
		new_total_weight += importance_weight[i];
	}

	//normalize to total_weight
	for (int i = 0; i < particles_num; i++) {
		importance_weight[i] = importance_weight[i] * total_weight / new_total_weight;
		assert(importance_weight[i] > 0);
	}

	return importance_weight;
}


int PedPomdp::NumObservations() const {
	//cout<<__FUNCTION__<<": Obs space too large! INF used instead"<<endl;
	return std::numeric_limits<int>::max();
}
int PedPomdp::ParallelismInStep() const {
	return ModelParams::N_PED_IN;
}
void PedPomdp::ExportState(const State& state, std::ostream& out) const {
	PomdpState cardriveState = static_cast<const PomdpState&>(state);
	ios::fmtflags old_settings = out.flags();

	int Width = 7;
	int Prec = 3;
	out << cardriveState.scenario_id << " ";
	out << cardriveState.weight << " ";
	out << cardriveState.num << " ";
	out << cardriveState.car.heading_dir << " "
			<< cardriveState.car.pos.x<<" "
			<< cardriveState.car.pos.y<<" "
			<< cardriveState.car.vel << " ";
	for (int i = 0; i < ModelParams::N_PED_IN; i++)
		out << cardriveState.peds[i].goal << " " << cardriveState.peds[i].id
		    << " " << cardriveState.peds[i].pos.x << " " << cardriveState.peds[i].pos.y
		    << " " << cardriveState.peds[i].vel << " ";

	out << endl;

	out.flags(old_settings);
}
State* PedPomdp::ImportState(std::istream& in) const {
	PomdpState* cardriveState = memory_pool_.Allocate();

	if (in.good())
	{
		string str;
		while (getline(in, str))
		{
			if (!str.empty())
			{
				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.heading_dir
					>> cardriveState->car.pos.x
					>> cardriveState->car.pos.y
					>> cardriveState->car.vel;
				for (int i = 0; i < ModelParams::N_PED_IN; i++)
					ss >> cardriveState->peds[i].goal >> cardriveState->peds[i].id
					   >> cardriveState->peds[i].pos.x >> cardriveState->peds[i].pos.y
					   >> cardriveState->peds[i].vel;
			}
		}
	}

	return cardriveState;
}
void PedPomdp::ImportStateList(std::vector<State*>& particles, std::istream& in) const {
	if (in.good())
	{
		int PID = 0;
		string str;
		getline(in, str);
		istringstream ss(str);
		int size;
		ss >> size;
		particles.resize(size);
		while (getline(in, str))
		{
			if (!str.empty())
			{
				if (PID >= particles.size())
					cout << "Import particles error: PID>=particles.size()!" << endl;

				PomdpState* cardriveState = memory_pool_.Allocate();

				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.heading_dir
								>> cardriveState->car.pos.x
								>> cardriveState->car.pos.y
								>> cardriveState->car.vel;
				for (int i = 0; i < ModelParams::N_PED_IN; i++)
					ss >> cardriveState->peds[i].goal >> cardriveState->peds[i].id
					   >> cardriveState->peds[i].pos.x >> cardriveState->peds[i].pos.y
					   >> cardriveState->peds[i].vel;
				particles[PID] = cardriveState;
				PID++;

			}
		}
	}
}


OBS_TYPE PedPomdp::StateToIndex(const State* state) const{
	std::hash<std::vector<int>> myhash;
	std::vector<int> obs_vec=ObserveVector(*state);
	OBS_TYPE obs=myhash(obs_vec);
	Obs_hash_table[obs]=obs_vec;

	//cout<<"Hash obs: "<<obs<<endl;

	if(obs<=(OBS_TYPE)140737351976300){
		cout<<"empty obs: "<<obs<<endl;
		return obs;
	}

	return obs;
}


SolverPrior* PedPomdp::CreateSolverPrior(World* world, std::string name, bool update_prior) const{
	SolverPrior* prior=NULL;

	if ( name == "NEURAL") {
		prior= new PedNeuralSolverPrior(this,*world_model);
	}

	cerr << "DEBUG: Getting initial state " << endl;

	const State* init_state=world->GetCurrentState();

	cerr << "DEBUG: Adding initial state " << endl;

	if (init_state != NULL && update_prior){
		prior->Add(-1, init_state);
		State* init_search_state_=CopyForSearch(init_state);//the state is used in search
		prior->Add_in_search(-1, init_search_state_);
		logi << __FUNCTION__ << " add history search state of ts " <<
			static_cast<PomdpState*>(init_search_state_)->time_stamp << endl;
	}

	return prior;
}

State* PedPomdp::CopyForSearch(const State* particle) const {
	//num_active_particles ++;
	PomdpState* new_particle = memory_pool_.Allocate();
	const PomdpStateWorld* world_state = static_cast<const PomdpStateWorld*>(particle);

	new_particle->num=min(ModelParams::N_PED_IN, world_state->num);
	new_particle->car=world_state->car;
	for (int i=0;i<new_particle->num; i++){
		new_particle->peds[i]=world_state->peds[i];
	}
	new_particle->time_stamp = world_state->time_stamp;

	new_particle->SetAllocated();
	return new_particle;
}

double PedPomdp::GetAccelerationID(ACT_TYPE action, bool debug) const{
	return (action%((int)(2*ModelParams::NumAcc+1)));
}

double PedPomdp::GetAcceleration(ACT_TYPE action, bool debug) const{
	double acc_ID=(action%((int)(2*ModelParams::NumAcc+1)));

	return GetAccfromAccID(acc_ID);
//	double normalized_acc=acc_ID/ModelParams::NumAcc;
//	double shifted_acc=normalized_acc-1;
//	if(debug)
//		cout<<"[GetAcceleration] (acc_ID, noirmalized_acc, shifted_acc)="
//		<<"("<<acc_ID<<","<<normalized_acc<<","<<shifted_acc<<")"<<endl;
//	return shifted_acc*ModelParams::AccSpeed;
}


double PedPomdp::GetAccelerationNoramlized(ACT_TYPE action, bool debug) const{
	double acc_ID=(action%((int)(2*ModelParams::NumAcc+1)));

	return GetNormalizeAccfromAccID(acc_ID);

//	double normalized_acc=acc_ID/ModelParams::NumAcc;
//	double shifted_acc=normalized_acc-1;
//	if(debug)
//		cout<<"[GetAcceleration] (acc_ID, noirmalized_acc, shifted_acc)="
//		<<"("<<acc_ID<<","<<normalized_acc<<","<<shifted_acc<<")"<<endl;
//	return shifted_acc;
}

double PedPomdp::GetSteeringID(ACT_TYPE action, bool debug) const{
	return FloorIntRobust(action/(2*ModelParams::NumAcc+1));
}

double PedPomdp::GetSteering(ACT_TYPE action, bool debug) const{
	double steer_ID=FloorIntRobust(action/(2*ModelParams::NumAcc+1));
	double normalized_steer=steer_ID/ModelParams::NumSteerAngle;
	double shifted_steer=normalized_steer-1;

	if(debug)
		cout<<"[GetSteering] (steer_ID, normalized_steer, shifted_steer)="
		<<"("<<steer_ID<<","<<normalized_steer<<","<<shifted_steer<<")"<<endl;
	return shifted_steer*ModelParams::MaxSteerAngle;
}

ACT_TYPE PedPomdp::GetActionID(double steering, double acc, bool debug){
	if(debug){
		cout<<"[GetActionID] steer_ID=" << GetSteerIDfromSteering(steering) <<endl;
		cout<<"[GetActionID] acc_ID=" << GetAccIDfromAcc(acc) <<endl;
	}

	return (ACT_TYPE)(GetSteerIDfromSteering(steering)*ClosestInt(2*ModelParams::NumAcc+1)
			+ GetAccIDfromAcc(acc));
}

ACT_TYPE PedPomdp::GetActionID(int steer_id, int acc_id){
	return (ACT_TYPE)(steer_id * ClosestInt(2*ModelParams::NumAcc+1) + acc_id);
}

double PedPomdp::GetAccfromAccID(int acc){
	switch(acc){
		case ACT_CUR: return 0;
		case ACT_ACC: return ModelParams::AccSpeed;
		case ACT_DEC: return -ModelParams::AccSpeed;
	}
}


double PedPomdp::GetNormalizeAccfromAccID(int acc){
	switch(acc){
		case ACT_CUR: return 0;
		case ACT_ACC: return 1.0;
		case ACT_DEC: return -1.0;
	}
}

int PedPomdp::GetAccIDfromAcc(float acc){
	if (fabs(acc-0)<1e-5)
		return ACT_CUR;
	else if (fabs(acc - ModelParams::AccSpeed) < 1e-5)
		return ACT_ACC;
	else if (fabs(acc - (-ModelParams::AccSpeed)) < 1e-5)
		return ACT_DEC;
}

int PedPomdp::GetSteerIDfromSteering(float steering){
	return ClosestInt((steering/ModelParams::MaxSteerAngle+1)*(ModelParams::NumSteerAngle));
}

