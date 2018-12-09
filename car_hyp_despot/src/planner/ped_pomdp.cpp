#include "ped_pomdp.h"

#include <limits>
#include <despot/GPUcore/thread_globals.h>
#include <despot/interface/default_policy.h>
#include <despot/core/builtin_policy.h>
#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/builtin_upper_bounds.h>
#include <despot/solver/despot.h>

#include "custom_particle_belief.h"

#include <errno.h>
#include <sys/stat.h>


static std::map<uint64_t, std::vector<int>> Obs_hash_table;
static PomdpState hashed_state;

#define ONEOVERSQRT2PI 1.0 / sqrt(2.0 * M_PI)

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
	use_rvo_in_search = false;
	use_rvo_in_simulation = true;
	if (use_rvo_in_simulation)
		ModelParams::LASER_RANGE = 8.0;
	else
		ModelParams::LASER_RANGE = 8.0;
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
	return (action == ACT_DEC || action == ACT_ACC) ? -0.1 : 0.0;
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
		return true;
	}

 	// Safety control: collision; Terminate upon collision
    if(state.car.vel > 0.001 && world_model->inCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state);
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(action);

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

	world_model->RobStep(state.car, steering, rNum/*random*/);

	world_model->RobVelStep(state.car, acc, rNum/*random*/);


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
	reward += ActionPenalty(action);

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

void PedPomdp::PrintState(const State& s, ostream& out) const {

	if (DESPOT::Debug_mode)
		return;

	if (static_cast<const PomdpStateWorld*> (&s)!=NULL){
		PrintWorldState(static_cast<const PomdpStateWorld&> (s), out);
		return;
	}

	const PomdpState & state = static_cast<const PomdpState&> (s);
	auto& carpos = state.car.pos;

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
	double q_single_weight;
	q_single_weight = 1.0 / particles.size();
	cout << "Current Belief" << endl;
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
		}
	}

	for (int j = 0; j < 6; j ++) {
		cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
		for (int i = 0; i < world_model->goals.size(); i ++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}

	cout << "<><><> q:" << endl;
	for (int j = 0; j < 6; j ++) {
		cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
		for (int i = 0; i < world_model->goals.size(); i ++) {
			cout << (q_goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
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

	const State* init_state=world->GetCurrentState();

	if (init_state != NULL && update_prior){
		prior->Add(-1, init_state);
		State* init_search_state_=CopyForSearch(init_state);//the state is used in search
		prior->Add_in_search(-1, init_search_state_);
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

	new_particle->SetAllocated();
	return new_particle;
}


double PedPomdp::GetAcceleration(ACT_TYPE action, bool debug) const{
	double acc_ID=(action%((int)(2*ModelParams::NumAcc+1)));
	double normalized_acc=acc_ID/ModelParams::NumAcc;
	double shifted_acc=normalized_acc-1;
	if(debug)
		cout<<"[GetAcceleration] (acc_ID, noirmalized_acc, shifted_acc)="
		<<"("<<acc_ID<<","<<normalized_acc<<","<<shifted_acc<<")"<<endl;
	return shifted_acc*ModelParams::AccSpeed;
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
		cout<<"[GetActionID] (shifted_steer, normalized_steer, steer_ID)="
				<<"("<<steering/ModelParams::MaxSteerAngle<<","
				<<(steering/ModelParams::MaxSteerAngle+1)<<","
				<<(ACT_TYPE)ClosestInt((steering/ModelParams::MaxSteerAngle+1)*(ModelParams::NumSteerAngle))<<")"<<endl;
		cout<<"[GetActionID] (shifted_acc, noirmalized_acc, acc_ID)="
				<<"("<<acc/ModelParams::AccSpeed<<","
				<<acc/ModelParams::AccSpeed+1<<","
				<<(ACT_TYPE)ClosestInt((acc/ModelParams::AccSpeed+1)*ModelParams::NumAcc)<<")"<<endl;
	}
	return (ACT_TYPE)ClosestInt((steering/ModelParams::MaxSteerAngle+1)*(ModelParams::NumSteerAngle))
			*(ACT_TYPE)ClosestInt(2*ModelParams::NumAcc+1)
			+(ACT_TYPE)ClosestInt((acc/ModelParams::AccSpeed+1)*ModelParams::NumAcc);
}

double PedPomdp::GetAccfromAccID(int acc){
	switch(acc){
		case ACT_CUR: return 0;
		case ACT_ACC: return ModelParams::AccSpeed;
		case ACT_DEC: return -ModelParams::AccSpeed;
	}
}


/*
 * Initialize the prior class and the neural networks
 */
inline PedNeuralSolverPrior::PedNeuralSolverPrior(const DSPOMDP* model,
		WorldModel& world) :
		SolverPrior(model), world_model(world) {

	action_probs_.resize(model->NumActions());

	// TODO: get num_peds_in_NN from ROS param
	num_peds_in_NN = 20;
	// TODO: get num_hist_channels from ROS param
	num_hist_channels = 4;

	// DONE Declare the neural network as a class member, and load it here

	// TODO: Pass the model name through ROS params

	drive_net = torch::jit::load("path to .pt");

	// DONE: The environment map will be received via ROS topic as the OccupancyGrid type
	//		 Data will be stored in raw_map_ (class member)
	//       In the current stage, just use a randomized raw_map_ to develop your code.
	//		 Init raw_map_ including its properties here
	//	     Refer to python codes: bag_2_hdf5.parse_map_data_from_dict
	//		 (map_dict_entry is the raw OccupancyGrid data)

	map_prop_.downsample_ratio = 0.03125;
	map_prop_.resolution = raw_map_.info.resolution;
	map_prop_.origin = COORD(raw_map_.info.origin.position.x, raw_map_.info.origin.position.y);
	map_prop_.dim = (int)(raw_map_.info.height);
	map_prop_.new_dim = (int)(map_prop_.dim * map_prop_.downsample_ratio);
	map_prop_.map_intensity_scale = 1500.0;

	// DONE: Convert the data in raw_map to your desired image data structure;
	// 		 (like a torch tensor);

	map_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);

	int index = 0;
	for (std::vector<int8_t>::const_reverse_iterator iterator = raw_map_.data.rbegin();
		  iterator != raw_map_.data.rend(); ++iterator) {
		int x = (size_t)(index / map_prop_.dim);
		int y = (size_t)(index % map_prop_.dim);
		assert(*iterator != -1);
		map_image_.at<float>(x,y) = (float)(*iterator);
		index++;
	}

	double minVal, maxVal;
	minMaxLoc(map_image_, &minVal, &maxVal);
	map_prop_.map_intensity = maxVal;

	goal_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);

	for( int i=0;i<num_hist_channels;i++){
		map_hist_images_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);
		car_hist_images_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);
	}

	empty_map_tensor_ = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
	map_tensor_ = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
	goal_tensor = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);

	for( int i=0;i<num_hist_channels;i++){
		map_hist_tensor_.push_back(at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));
		car_hist_tensor_.push_back(at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));
	}

	// Car geometry
	vector<cv::Point3f> polygon { Point3f(3.6, 0.95, 1), Point3f(-0.8, 0.95, 1), Point3f(-0.8, -0.95, 1), Point3f(3.6, -0.95, 1)};

}

COORD PedNeuralSolverPrior::point_to_indices(COORD pos, COORD origin, double resolution, int dim) const{
    COORD indices = COORD((pos.x - origin.x) / resolution, (pos.y - origin.y) / resolution);
    if (indices.x < 0 || indices.y < 0 ||
    		indices.x > (dim - 1) || indices.y > (dim - 1))
        return COORD(-1, -1);
    return indices;
}

void PedNeuralSolverPrior::add_in_map(cv::Mat map_image, COORD indices, double map_intensity, double map_intensity_scale){
	if (indices.x == -1 || indices.y == -1)
		return;

	map_image.at<float>((int)round(indices.x), (int)round(indices.y)) = map_intensity * map_intensity_scale;
}

float radians(float degrees){
	return (degrees * M_PI)/180.0;
}

std::vector<COORD> PedNeuralSolverPrior::get_transformed_car(CarStruct car, COORD origin, double resolution){
    float theta = car.heading_dir; // TODO: validate that python code is using [0, 2pi] as the range
    float x = car.pos.x - origin.x;
    float y = car.pos.y - origin.x;
    // transformation matrix: rotate by theta and translate with (x,y)
//    cv::Mat rot_mat( 3, 3, CV_32FC1 );
//
//    double rot_array[] = {cos(theta), -sin(theta), x, sin(theta), cos(theta), y, 0, 0, 1};
//    MatIterator_<double> it, end;
//    int i =0;
//    for( it = rot_mat.begin<double>(), end = rot_mat.end<double>(); it != end; ++it)
//    {
//        *it = rot_array[i];
//        i++;
//    }

    // rotate and scale the car
    vector<COORD> car_polygon;
    for (int i=0; i < car_shape.size(); i++){
//    	Point3f& original = car_shape[i];
//    	Point3f rotated;
    	vector<Point3f> original, rotated;
    	original.push_back(car_shape[i]);
    	rotated.resize(1);
    	cv::transform(original, rotated,
    			cv::Matx33f(cos(theta), -sin(theta), x, sin(theta), cos(theta), y, 0, 0, 1));
    	car_polygon.push_back(COORD(rotated[0].x / resolution, rotated[0].y / resolution));
    }

    // TODO: validate the transformation in test_opencv
    return car_polygon;
}

void fill_car_edges(Mat image, vector<COORD>& points){
	float default_intensity = 1.0;

	for (int i=0;i<points.size();i++){
		int r0, c0, r1, c1;
		r0 = round(points[i].x);
		c0 = round(points[i].y);
		if (i+1 < points.size()){
			r1 = round(points[i+1].x);
			c1 = round(points[i+1].y);
		}
		else{
			r1 = round(points[0].x);
			c1 = round(points[0].y);
		}

		cv::line(image, Point(r0,c0), Point(r1,c1), default_intensity);
	}
}

void rescale_image(cv::Mat& image, at::Tensor& tensor, double downsample_ratio=0.03125){
	Mat tmp = image;
	Mat dst;
    for (int i=0; i < (int)log2(1.0 / downsample_ratio); i++){
        pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
        tmp = dst;
    }

    for(int i=0; i<tmp.rows; i++)
        for(int j=0; j<tmp.cols; j++)
        	tensor[i][j] = tmp.at<float>(i,j);
}

int img_counter = 0;
std::string img_folder="./visualize";

void mkdir_safe(std::string dir){
	if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
	{
	    if( errno == EEXIST ) {
	       // alredy exists
	    } else {
	       // something else
	        std::cout << "cannot create sessionnamefolder error:" << strerror(errno) << std::endl;
	        throw std::runtime_error( strerror(errno) );
	    }
	}
}

void export_image(Mat& image){
	mkdir_safe(img_folder);
	std::ostringstream stringStream;
	stringStream << img_folder << "/" << img_counter << ".jpg";
	std::string img_name = stringStream.str();
	imwrite( img_name , image );
}

void inc_counter(){
	img_counter ++;
}

void reset_counter(){
	img_counter = 0;
}

void PedNeuralSolverPrior::Process_states(const vector<PomdpState*>& hist_states, const vector<int> hist_ids) {
	// DONE: Create num_history copies of the map image, each for a frame of dynamic environment
	// DONE: Get peds from hist_states and put them into the dynamic maps
	//		 Do this for all num_history time steps
	// 		 Refer to python codes: bag_2_hdf5.process_peds, bag_2_hdf5.get_map_indices, bag_2_hdf5.add_to_ped_map
	//		 get_map_indices converts positions to pixel indices
	//		 add_to_ped_map fills the corresponding entry (with some intensity value)
	for (int i = 0; i < hist_states.size(); i++) {
		// clear data in the dynamic map
		int hist_channel = hist_ids[i];

		map_hist_images_[hist_channel].setTo(0.0);
		// get the array of pedestrians (of length ModelParams::N_PED_IN)
		auto& ped_list = hist_states[i]->peds;
		int num_valid_ped = 0;
		for (int ped_id = 0; ped_id < ModelParams::N_PED_IN; ped_id++) {
			// Process each pedestrian
			PedStruct ped = ped_list[ped_id];
			// get position of the ped
			COORD ped_indices = point_to_indices(ped.pos, map_prop_.origin, map_prop_.resolution, map_prop_.dim);
			if (ped_indices.x == -1 or ped_indices.y == -1) // ped out of map
				continue;
			// put the point in the dynamic map
			add_in_map(map_hist_images_[hist_channel], ped_indices, map_prop_.map_intensity, map_prop_.map_intensity_scale);
		}
	}
	// DONE: Allocate 1 goal image (a tensor)
	// DONE: Get path and fill into the goal image
	//       Refer to python codes: bag_2_hdf5.construct_path_data, bag_2_hdf5.fill_image_with_points
	//	     construct_path_data only calculates the pixel indices
	//       fill_image_with_points fills the entries in the images (with some intensity value)
	if (hist_states.size()==1) { // only for current node states
		goal_image_.setTo(0.0);
		Path& path = world_model.path;
		for (int i = 0; i < path.size(); i++) {
			COORD point = path[i];
			// process each point
			COORD indices = point_to_indices(point, map_prop_.origin, map_prop_.resolution, map_prop_.dim);
			if (indices.x == -1 or indices.y == -1) // path point out of map
				continue;
			// put the point in the goal map
			add_in_map(goal_image_,indices, map_prop_.map_intensity, map_prop_.map_intensity_scale);
		}
	}
	// DONE: Allocate num_history history images, each for a frame of car state
	//		 Refer to python codes: bag_2_hdf5.get_transformed_car, fill_car_edges, fill_image_with_points
	// DONE: get_transformed_car apply the current transformation to the car bounding box
	//	     fill_car_edges fill edges of the car shape with dense points
	//		 fill_image_with_points fills the corresponding entries in the images (with some intensity value)
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];
		map_hist_images_[hist_channel].setTo(0.0);

		CarStruct& car = hist_states[i]->car;
		//     car vertices in its local frame
		//      (-0.8, 0.95)---(3.6, 0.95)
		//      |                       |
		//      |                       |
		//      |                       |
		//      (-0.8, -0.95)--(3.6, 0.95)
		// ...
		vector<COORD> transformed_car = get_transformed_car(car, map_prop_.origin, map_prop_.resolution);
		fill_car_edges(car_hist_images_[hist_channel],transformed_car);
	}

	// DONE: Now we have all the high definition images, scale them down to 32x32
	//		 Refer to python codes bag_2_hdf5.rescale_image
	//		 Dependency: OpenCV
	rescale_image(goal_image_, goal_tensor);
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];
		rescale_image(map_hist_images_[hist_channel], map_hist_tensor_[hist_channel]);
		rescale_image(car_hist_images_[hist_channel], car_hist_tensor_[hist_channel]);
	}
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_nodes_input(const std::vector<State*>& vnode_states){
	vector<PomdpState*> cur_state;
	vector<torch::Tensor> output_images;
	cur_state.resize(1);
	vector<int> hist_ids({3}); // use as the last hist step
	for (int i = 0; i < vnode_states.size(); i++){
		cur_state[0]= static_cast<PomdpState*>(vnode_states[i]);

		Process_states(cur_state, hist_ids);

		auto node_nn_input = Combine_images();

		output_images.push_back(node_nn_input);
	}

	return output_images;
}

torch::Tensor PedNeuralSolverPrior::Combine_images(){
	//DONE: Make a tensor with all 9 channels of nn_input_images_ with the 32x32 images here.
			//		 [IMPORTANT] Be cautious on the order of the channels:
			//			config.channel_map = 0  # current step
			//	    	config.channel_map1 = 1  # history steps t-1
			//	    	config.channel_map2 = 2  # history steps t-2
			//	    	config.channel_map3 = 3  # history steps t-3
			//	    	config.channel_goal = 4
			//	    	config.channel_hist1 = 5  # current step
			//	    	config.channel_hist2 = 6  # history steps
			//	    	config.channel_hist3 = 7  # history steps
			//	    	config.channel_hist4 = 8  # history steps

	// DONE: stack them together and return
	torch::Tensor result;
	for (int i = 0; i < num_hist_channels; i++) {
		if (i==0)
			result = map_hist_tensor_[i].unsqueeze(0);
		else
			result = torch::stack({result, map_hist_tensor_[i].unsqueeze(0)}, 0);
	}

	result = torch::stack({result, goal_tensor.unsqueeze(0)}, 0);

	for (int i = 0; i < num_hist_channels; i++) {
		result = torch::stack({result, car_hist_tensor_[i].unsqueeze(0)}, 0);
	}

	if (true){
		for (int i = 0; i < num_hist_channels; i++)
			export_image(map_hist_images_[i]);
		export_image(goal_image_);
		for (int i = 0; i < num_hist_channels; i++)
			export_image(car_hist_images_[i]);
		inc_counter();
	}

	return result;
}

at::Tensor gaussian_probability(at::Tensor &sigma, at::Tensor &mu, at::Tensor &data) {
    // data = data.toType(at::kDouble);
    // sigma = sigma.toType(at::kDouble);
    // mu = mu.toType(at::kDouble);
    // data = data.toType(at::kDouble);
    data = data.unsqueeze(1).expand_as(sigma);
//    std::cout << "data=" << data << std::endl;
//    std::cout << "mu=" << mu  << std::endl;
//    std::cout << "sigma=" << sigma  << std::endl;
//    std::cout << "data - mu=" << data - mu  << std::endl;

    auto exponent = -0.5 * at::pow((data - mu) / sigma, at::Scalar(2));
    std::cout << "exponent=" << exponent << std::endl;
    auto ret = ONEOVERSQRT2PI * (exponent.exp() / sigma);
    std::cout << "ret=" << ret << std::endl;
    return at::prod(ret, 2);
}

at::Tensor gm_pdf(at::Tensor &pi, at::Tensor &sigma,
    at::Tensor &mu, at::Tensor &target) {
    auto prob_double = pi * gaussian_probability(sigma, mu, target);
    auto prob_float = prob_double.toType(at::kFloat);
    std::cout << "prob_float=" << prob_float << std::endl;
    auto safe_sum = at::add(at::sum(prob_float, at::IntList(1)), at::Scalar(0.000001));
    return safe_sum;
}

void PedNeuralSolverPrior::Compute(vector<torch::Tensor>& input_batch, map<OBS_TYPE, despot::VNode*>& vnodes){
	// TODO: Send nn_input_images_ to drive_net, and get the policy and value output
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);

	std::vector<torch::jit::IValue> inputs;

	for (int node_id = 0; node_id< input_batch.size(); node_id++){
		torch::Tensor& entry = input_batch[node_id];
		inputs.push_back(entry.unsqueeze_(0));
	}

	auto drive_net_output = drive_net->forward(inputs).toTuple()->elements();

	auto value_batch = drive_net_output[VALUE].toTensor();
	auto acc_pi_batch = drive_net_output[ACC_PI].toTensor();
	auto acc_mu_batch = drive_net_output[ACC_MU].toTensor();
	auto acc_sigma_batch = drive_net_output[ACC_SIGMA].toTensor();
	auto ang_batch = drive_net_output[ANG].toTensor();

    auto value_double = value_batch.accessor<float, 2>();

	int node_id = -1;
	for (std::map<OBS_TYPE, despot::VNode* >::iterator it = vnodes.begin();
		        it != vnodes.end(); it++) {
		despot::VNode* vnode = it->second;
		node_id ++;

		auto acc_pi = acc_pi_batch[node_id];
		auto acc_mu = acc_mu_batch[node_id];
		auto acc_sigma = acc_sigma_batch[node_id];
		auto ang = ang_batch[node_id];

		// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))

		int num_accs = 2*ModelParams::NumAcc+1;
		at::Tensor acc_candiates = at::ones({num_accs, 1}, at::kFloat);
		for (int acc = 0;  acc < num_accs; acc ++){
			acc_candiates[acc][0] = ped_model->GetAcceleration(acc);
		}

		int num_modes = acc_pi.size(0);

		auto acc_pi_actions = acc_pi.unsqueeze(0).expand({num_accs, num_modes});
		auto acc_mu_actions = acc_mu.unsqueeze(0).expand({num_accs, num_modes, 1});
		auto acc_sigma_actions = acc_sigma.unsqueeze(0).expand({num_accs, num_modes, 1});

		auto acc_probs_Tensor = gm_pdf(acc_pi_actions, acc_sigma_actions, acc_mu_actions, acc_candiates);

        auto steer_probs_Tensor = at::_softmax(ang, 0, false);

	    auto acc_probs_double = acc_probs_Tensor.accessor<float, 1>();
	    auto steer_probs_double = steer_probs_Tensor.accessor<float, 1>();

		// Update the values in the vnode
		for (int action = 0;  action < ped_model->NumActions(); action ++){
			int acc_ID=(action%((int)(2*ModelParams::NumAcc+1)));
			int steerID = FloorIntRobust(action/(2*ModelParams::NumAcc+1));

			float acc_prob = acc_probs_double[acc_ID];
			float steer_prob = steer_probs_double[steerID];

			float joint_prob = acc_prob * steer_prob;
			vnode->prior_action_probs(action, joint_prob);
			// get the steering prob from angs
		}

		vnode->prior_value(value_double[node_id][0]);
	}
}


void PedNeuralSolverPrior::Process_history(int mode){
	int num_history = 0;
	vector<int> hist_ids;
	if (mode == FULL) // Full update of the input channels
		num_history = num_hist_channels;
	else if (mode == PARTIAL) // Partial update of input channels and reuse old channels
		num_history = num_hist_channels - 1;

	for (int i = 0 ; i<num_history ; i++)
		hist_ids.push_back(i);

	// DONE: get the 4 latest history states
	vector<PomdpState*> hist_states;
	int latest=as_history_in_search_.Size()-1;
	for (int t = latest; t > latest - num_history ; t--){// process in reserved time order
		PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
		hist_states.push_back(car_peds_state);
	}

	if (mode == FULL){
		Process_states(hist_states, hist_ids);
	}else if (mode == PARTIAL){
		Process_states(hist_states, hist_ids);
	}
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_history_input(){
	int num_history = num_hist_channels;

	// TODO: get the 4 latest history states
	vector<PomdpState*> hist_states;
	int latest=as_history_in_search_.Size()-1;
	for (int t = latest; t > latest - num_history ; t--){// process in reserved time order
		PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
		hist_states.push_back(car_peds_state);
	}

	vector<int> hist_ids;
	for (int i = 0 ; i<num_history ; i++)
		hist_ids.push_back(i);

	Process_states(hist_states, hist_ids);

	std::vector<torch::Tensor> nn_input;
	nn_input.push_back(Combine_images());
	return nn_input;
}

///*
// * query the policy network to provide prior probability for PUCT
// */
//const vector<double>& PedNeuralSolverPrior::ComputePreference(){
//
//	// TODO: remove this when you finish the coding
//	throw std::runtime_error( "PedNeuralSolverPrior::ComputePreference hasn't been implemented!" );
//	cerr << "" << endl;
//
//	// TODO: Construct input images
//	Process_history(FULL);
//
//	auto nn_input = Combine_images();
//
//	// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))
//	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);
//	for (int action = 0;  action < ped_model->NumActions(); action ++){
//		double accelaration = ped_model->GetAcceleration(action);
//		// TODO: get the probability of acceleration from the Gaussian mixture, and store in action_probs_
//		// Hint: Need to implement Gaussian pdf and calculate mixture
//		// Hint: Refer to Components/mdn.py in the BTS_RL_NN project
//
//	}
//
//	// return the output as vector<double>
//	return action_probs_;
//}
//
///*
// * query the value network to initialize leaf node values
// */
//double PedNeuralSolverPrior::ComputeValue(){
//
//	// TODO: remove this when you finish the coding
//	throw std::runtime_error( "PedNeuralSolverPrior::ComputeValue hasn't been implemented!" );
//
//	// TODO: Construct input images
//	// 		 Here we can reuse existing channels
//	Process_history(PARTIAL);
//
//	// TODO: Send nn_input_images_ to drive_net, and get the value output
//
//	// TODO: return the output as double
//	return 0;
//}
//


