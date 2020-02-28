#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_map>

#include <core/builtin_lower_bounds.h>
#include <core/builtin_policy.h>
#include <core/builtin_upper_bounds.h>
#include <core/prior.h>
#include <GPUcore/thread_globals.h>
#include <interface/default_policy.h>
#include <interface/world.h>
#include <solver/despot.h>
#include <util/seeds.h>
#include <despot/util/logging.h>

#include <GammaParams.h>

#include "config.h"
#include "coord.h"
#include "crowd_belief.h"
#include "utils.h"
#include "default_prior.h"
#include "world_model.h"
#include "context_pomdp.h"
#include "simulator_base.h"

using namespace despot;


double path_look_ahead = 5.0;


static std::map<uint64_t, std::vector<int>> Obs_hash_table;
static PomdpState hashed_state;

class ContextPomdpParticleLowerBound: public ParticleLowerBound {
private:
	const ContextPomdp *ped_pomdp_;
public:
	ContextPomdpParticleLowerBound(const DSPOMDP *model) :
			ParticleLowerBound(model), ped_pomdp_(
					static_cast<const ContextPomdp *>(model)) {
	}

	virtual ValuedAction Value(const std::vector<State *> &particles) const {
		PomdpState *state = static_cast<PomdpState *>(particles[0]);
		int min_step = numeric_limits<int>::max();
		auto &carpos = state->car.pos;
		double carvel = state->car.vel;

		// Find minimum number of steps for car-pedestrian collision
		for (int i = 0; i < state->num; i++) {
			auto &p = state->agents[i];

			if (!ped_pomdp_->world_model.InFront(p.pos, state->car))
				continue;
			int step = min_step;
			if (p.speed + carvel > 1e-5)
				step = int(ceil(
						ModelParams::CONTROL_FREQ *
						max(COORD::EuclideanDistance(carpos, p.pos) - ModelParams::CAR_FRONT, 0.0)
						/ ((p.speed + carvel))));

			if (DoPrintCPU)
				printf("   step,min_step, p.speed + carvel=%d %d %f\n", step,
						min_step, p.speed + carvel);
			min_step = min(step, min_step);
		}

		double value = 0;
		ACT_TYPE default_act;
		int dec_step = round(carvel / ModelParams::ACC_SPEED * ModelParams::CONTROL_FREQ);
		float stay_cost = ModelParams::REWARD_FACTOR_VEL
							* (0.0 - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;

		if (dec_step > min_step) {
			value = ped_pomdp_->CrashPenalty(*state);
		} else {
			// 2. stay forever
			value += stay_cost / (1 - Globals::Discount());
		}

		// 1. Decelerate until collision or full stop
		for (int step = 0; step < min(dec_step, min_step); step++) { // -1.0 is action penalty
			value = -1.0 + stay_cost + value * Globals::Discount();
		}

		// default action, go straight and decelerate
		int steerID = ped_pomdp_->GetSteerIDfromSteering(0);
		default_act = ped_pomdp_->GetActionID(steerID, 2);

		return ValuedAction(default_act, State::Weight(particles) * value);
	}
};

class ContextPomdpSmartScenarioLowerBound: public DefaultPolicy {
protected:
	const ContextPomdp *ped_pomdp_;

public:
	ContextPomdpSmartScenarioLowerBound(const DSPOMDP *model,
			ParticleLowerBound *bound) :
			DefaultPolicy(model, bound), ped_pomdp_(
					static_cast<const ContextPomdp *>(model)) {
	}

	int Action(const std::vector<State *> &particles, RandomStreams &streams,
			History &history) const {
		return ped_pomdp_->world_model.DefaultPolicy(particles);
	}
};

class ContextPomdpSmartParticleUpperBound: public ParticleUpperBound {
protected:
	const ContextPomdp *ped_pomdp_;
public:
	ContextPomdpSmartParticleUpperBound(const DSPOMDP *model) :
			ped_pomdp_(static_cast<const ContextPomdp *>(model)) {
	}

	double Value(const State &s) const {
		const PomdpState &state = static_cast<const PomdpState &>(s);
		int min_step = ped_pomdp_->world_model.MinStepToGoal(state);
		return -ModelParams::TIME_REWARD * min_step
				+ ModelParams::GOAL_REWARD * Globals::Discount(min_step);
	}
};

ContextPomdp::ContextPomdp() : world_model(SimulatorBase::world_model),
		random_(Random((unsigned) Seeds::Next())) {
	InitGammaSetting();
}

SolverPrior *ContextPomdp::CreateSolverPrior(World *world, std::string name,
		bool update_prior) const {
	SolverPrior *prior = NULL;

	if (name == "DEFAULT") {
		prior = new DefaultPrior(this, world_model);
	}

	logv << "DEBUG: Getting initial state " << endl;

	const State *init_state = world->GetCurrentState();

	logv << "DEBUG: Adding initial state " << endl;

	if (init_state != NULL && update_prior) {
		prior->Add(-1, init_state);
		State *init_search_state_ = CopyForSearch(init_state); //the state is used in search
		prior->Add_in_search(-1, init_search_state_);
		logi << __FUNCTION__ << " add history search state of ts "
				<< static_cast<PomdpState *>(init_search_state_)->time_stamp
				<< endl;
	}

	return prior;
}

void ContextPomdp::InitGammaSetting() {
	use_gamma_in_search = true;
	use_gamma_in_simulation = true;
	use_simplified_gamma = false;

	if (use_simplified_gamma) {
		use_gamma_in_search = true;
		use_gamma_in_simulation = true;

		GammaParams::use_polygon = false;
		GammaParams::consider_kinematics = false;
		GammaParams::use_dynamic_att = false;
	}
}

const std::vector<int> &ContextPomdp::ObserveVector(const State &state_) const {
	const PomdpState &state = static_cast<const PomdpState &>(state_);
	static std::vector<int> obs_vec;

	obs_vec.resize(state.num * 2 + 3);

	int i = 0;
	obs_vec[i++] = int(state.car.pos.x / ModelParams::POS_RLN);
	obs_vec[i++] = int(state.car.pos.y / ModelParams::POS_RLN);

	obs_vec[i++] = int((state.car.vel + 1e-5) / ModelParams::VEL_RLN); //add some noise to make 1.5/0.003=50

	for (int j = 0; j < state.num; j++) {
		obs_vec[i++] = int(state.agents[j].pos.x / ModelParams::POS_RLN);
		obs_vec[i++] = int(state.agents[j].pos.y / ModelParams::POS_RLN);
	}

	return obs_vec;
}

uint64_t ContextPomdp::Observe(const State &state) const {
	hash<std::vector<int>> myhash;
	return myhash(ObserveVector(state));
}

std::vector<State *> ContextPomdp::ConstructParticles(
		std::vector<PomdpState> &samples) const {
	int num_particles = samples.size();
	std::vector<State *> particles;
	for (int i = 0; i < samples.size(); i++) {
		PomdpState *particle = static_cast<PomdpState *>(Allocate(-1,
				1.0 / num_particles));
		(*particle) = samples[i];
		particle->SetAllocated();
		particle->weight = 1.0 / num_particles;
		particles.push_back(particle);
	}

	return particles;
}

// Very high cost for collision
double ContextPomdp::CrashPenalty(const PomdpState &state) const // , int closest_ped, double closest_dist) const {
		{
	// double ped_vel = state.agent[closest_ped].vel;
	return ModelParams::CRASH_PENALTY
			* (state.car.vel * state.car.vel
					+ ModelParams::REWARD_BASE_CRASH_VEL);
}
// Very high cost for collision
double ContextPomdp::CrashPenalty(const PomdpStateWorld &state) const // , int closest_ped, double closest_dist) const {
		{
	// double ped_vel = state.agent[closest_ped].vel;
	return ModelParams::CRASH_PENALTY
			* (state.car.vel * state.car.vel
					+ ModelParams::REWARD_BASE_CRASH_VEL);
}

// Avoid frequent dec or acc
double ContextPomdp::ActionPenalty(int action) const {
	return (action == ACT_DEC) ? -0.1 : 0.0;
}

// Less penalty for longer distance travelled
double ContextPomdp::MovementPenalty(const PomdpState &state) const {
	return ModelParams::REWARD_FACTOR_VEL
			* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
}

double ContextPomdp::MovementPenalty(const PomdpState &state,
		float steering) const {
	return ModelParams::REWARD_FACTOR_VEL
			* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
}

// Less penalty for longer distance travelled
double ContextPomdp::MovementPenalty(const PomdpStateWorld &state) const {
	return ModelParams::REWARD_FACTOR_VEL
			* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
}

double ContextPomdp::MovementPenalty(const PomdpStateWorld &state,
		float steering) const {
	return ModelParams::REWARD_FACTOR_VEL
			* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
}

double ContextPomdp::Reward(const State& _state, ACT_TYPE action) const {
	const PomdpState &state = static_cast<const PomdpState &>(_state);
	double reward = 0.0;
	if (world_model.IsGlobalGoal(state.car)) {
		reward = ModelParams::GOAL_REWARD;

		cout << "assigning goal reward " << reward << endl;
		return reward;
	}

	// Safety control: collision; Terminate upon collision
	int col_agent = 0;
	if (state.car.vel > 0.001 && world_model.InCollision(state, col_agent)) /// collision occurs only when car is moving
			{
		reward = CrashPenalty(state);

		cout << "assigning collision reward " << reward << endl;
		return reward;
	}

	// Smoothness control
	double acc_reward = ActionPenalty(GetAccelerationID(action));
	reward += acc_reward;
	cout << "assigning action reward " << acc_reward << endl;

	// Speed control: Encourage higher speed
	double steering = GetSteering(action);
	double move_reward = MovementPenalty(state, steering);
	cout << "assigning move reward " << acc_reward << endl;
	cout << "Scaling factor=" << ModelParams::REWARD_FACTOR_VEL << ", car_vel="
			<< state.car.vel << ", VEL_MAX=" << ModelParams::VEL_MAX << endl;

	reward += move_reward;

	return reward;
}

bool ContextPomdp::Step(State &state_, double rNum, int action, double &reward,
		uint64_t &obs) const {
	PomdpState &state = static_cast<PomdpState &>(state_);
	reward = 0.0;

	////// NOTE: Using true random number to make results in different qnodes different ////
	rNum = Random::RANDOM.NextDouble();

	if (FIX_SCENARIO == 1 || DESPOT::Print_nodes) {
		if (CPUDoPrint && state_.scenario_id == CPUPrintPID) {
			printf("(CPU) Before step: scenario%d \n", state_.scenario_id);
			printf("action= %d \n", action);
			PomdpState *ContextPomdp_state = static_cast<PomdpState *>(&state_);
			printf("Before step:\n");
			printf("car_pos= %f,%f", ContextPomdp_state->car.pos.x,
					ContextPomdp_state->car.pos.y);
			printf("car_heading=%f\n", ContextPomdp_state->car.heading_dir);
			printf("car_vel= %f\n", ContextPomdp_state->car.vel);
			for (int i = 0; i < ContextPomdp_state->num; i++) {
				printf("agent %d pox_x= %f pos_y=%f\n", i,
						ContextPomdp_state->agents[i].pos.x,
						ContextPomdp_state->agents[i].pos.y);
			}
		}
	}
	// Terminate upon reaching goal
	if (world_model.IsGlobalGoal(state.car)) {
		reward = ModelParams::GOAL_REWARD;

		logd << "assigning goal reward " << reward <<
				" car at (" << state.car.pos.x << ", " << state.car.pos.y << ")" << endl;
		logd << "Path start " << world_model.path[0] << endl;
		logd << "Path length " << world_model.path.GetLength() << endl;

		ERR("");
		return true;
	}

	// Safety control: collision; Terminate upon collision
	int col_agent = 0;
	if (state.car.vel > 0.001 && world_model.InCollision(state, col_agent)) /// collision occurs only when car is moving
			{
		reward = CrashPenalty(state);

		logv << "assigning collision reward " << reward << endl;
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(GetAccelerationID(action));

	// Speed control: Encourage higher speed
	double steering = GetSteering(action);

	reward += MovementPenalty(state, steering);

	// State transition
	if (Globals::config.use_multi_thread_) {
		QuickRandom::SetSeed(INIT_QUICKRANDSEED,
				Globals::MapThread(this_thread::get_id()));
	} else
		QuickRandom::SetSeed(INIT_QUICKRANDSEED, 0);

	logv << "[ContextPomdp::" << __FUNCTION__ << "] Refract action" << endl;
	double acc = GetAcceleration(action);

	world_model.RobStep(state.car, steering, rNum);
	world_model.RobVelStep(state.car, acc, rNum);

	state.time_stamp = state.time_stamp + 1.0 / ModelParams::CONTROL_FREQ;

	if (use_gamma_in_search) {
		// Attentive pedestrians
		world_model.GammaAgentStep(state.agents, rNum, state.num, state.car);
		for (int i = 0; i < state.num; i++) {
			//Distracted pedestrians
			if (state.agents[i].mode == AGENT_DIS)
				world_model.AgentStep(state.agents[i], rNum);
		}
	} else {
		for (int i = 0; i < state.num; i++) {
			world_model.AgentStep(state.agents[i], rNum);
			if(isnan(state.agents[i].pos.x))
				ERR("state.agents[i].pos.x is NAN");
		}
	}

	if (CPUDoPrint && state.scenario_id == CPUPrintPID) {
		if (true) {
			PomdpState *ContextPomdp_state = static_cast<PomdpState *>(&state_);
			printf("(CPU) After step: scenario=%d \n",
					ContextPomdp_state->scenario_id);
			printf("rand=%f, action=%d \n", rNum, action);
			printf("After step:\n");
			printf("Reward=%f\n", reward);

			printf("car_pos= %f,%f", ContextPomdp_state->car.pos.x,
					ContextPomdp_state->car.pos.y);
			printf("car_heading=%f\n", ContextPomdp_state->car.heading_dir);
			printf("car vel= %f\n", ContextPomdp_state->car.vel);
			for (int i = 0; i < ContextPomdp_state->num; i++) {
				printf("agent %d pox_x= %f pos_y=%f\n", i,
						ContextPomdp_state->agents[i].pos.x,
						ContextPomdp_state->agents[i].pos.y);
			}
		}
	}

	// Observation
	obs = Observe(state);
	return false;
}

bool ContextPomdp::Step(PomdpStateWorld &state, double rNum, int action,
		double &reward, uint64_t &obs) const {

	reward = 0.0;

	if (world_model.IsGlobalGoal(state.car)) {
		reward = ModelParams::GOAL_REWARD;
		return true;
	}

	if (state.car.vel > 0.001 && world_model.InRealCollision(state, 120.0)) /// collision occurs only when car is moving
			{
		reward = CrashPenalty(state);
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(GetAccelerationID(action));

	// Speed control: Encourage higher speed
	double steering = GetSteering(action);
	reward += MovementPenalty(state, steering);

	// State transition
	Random random(rNum);
	logv << "[ContextPomdp::" << __FUNCTION__ << "] Refract action" << endl;
	double acc = GetAcceleration(action);

	world_model.RobStep(state.car, steering, random);
	world_model.RobVelStep(state.car, acc, random);

	if (use_gamma_in_simulation) {
		// Attentive pedestrians
		double zero_rand = 0.0;
		world_model.GammaAgentStep(state.agents, zero_rand, state.num, state.car);
		// Distracted pedestrians
		for (int i = 0; i < state.num; i++) {
			if (state.agents[i].mode == AGENT_DIS)
				world_model.AgentStep(state.agents[i], random);
		}
	} else {
		for (int i = 0; i < state.num; i++)
			world_model.AgentStep(state.agents[i], random);
	}
	return false;
}

void ContextPomdp::ForwardAndVisualize(const State *sample, int step) const {
	PomdpState *next_state = static_cast<PomdpState *>(Copy(sample));

	for (int i = 0; i < step; i++) {
		// forward
		next_state = PredictAgents(next_state);

		// print
		PrintStateCar(*next_state, string_sprintf("predicted_car_%d", i));
		PrintStateAgents(*next_state, string_sprintf("predicted_agents_%d", i));
	}
}

PomdpState* ContextPomdp::PredictAgents(const PomdpState *ped_state, int acc) const {
	PomdpState* predicted_state = static_cast<PomdpState*>(Copy(ped_state));

	double steer_to_path = world_model.GetSteerToPath(
			predicted_state->car);
	ACT_TYPE action = GetActionID(GetSteerIDfromSteering(steer_to_path), acc);

	OBS_TYPE dummy_obs;
	double dummy_reward;

	double rNum = Random::RANDOM.NextDouble();
	bool terminal = Step(*predicted_state, rNum, action, dummy_reward, dummy_obs);

	if (terminal)
		logi << "[PredictAgents] Reach terminal state" << endl;

	return predicted_state;
}

double ContextPomdp::ObsProb(uint64_t obs, const State &s, int action) const {
	return obs == Observe(s);
}

Belief *ContextPomdp::InitialBelief(const State *state, string type) const {

	//Uniform initial distribution
	CrowdBelief *belief = new CrowdBelief(this);
	return belief;
}

/// output the probability of the intentions of the pedestrians
ValuedAction ContextPomdp::GetBestAction() const {
	return ValuedAction(0,
			ModelParams::CRASH_PENALTY
					* (ModelParams::VEL_MAX * ModelParams::VEL_MAX
							+ ModelParams::REWARD_BASE_CRASH_VEL));
}

double ContextPomdp::GetMaxReward() const {
	return 0;
}

ScenarioLowerBound *ContextPomdp::CreateScenarioLowerBound(string name,
		string particle_bound_name) const {
	name = "SMART";
	ScenarioLowerBound *lb;
	if (name == "TRIVIAL") {
		lb = new TrivialParticleLowerBound(this);
	} else if (name == "RANDOM") {
		lb = new RandomPolicy(this, new ContextPomdpParticleLowerBound(this));
	} else if (name == "SMART") {
		Globals::config.rollout_type = "INDEPENDENT";
		cout << "[LowerBound] Smart policy independent rollout" << endl;
		lb = new ContextPomdpSmartScenarioLowerBound(this,
				new ContextPomdpParticleLowerBound(this));
	} else {
		cerr << "[LowerBound] Unsupported scenario lower bound: " << name
				<< endl;
		exit(0);
	}

	return lb;
}

ParticleUpperBound *ContextPomdp::CreateParticleUpperBound(string name) const {
	name = "SMART";
	if (name == "TRIVIAL") {
		return new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		return new ContextPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "Unsupported particle upper bound: " << name << endl;
		exit(0);
	}
}

ScenarioUpperBound *ContextPomdp::CreateScenarioUpperBound(string name,
		string particle_bound_name) const {
	// name = "SMART";
	name = "TRIVIAL";
	ScenarioUpperBound *ub;
	if (name == "TRIVIAL") {
		cout << "[UpperBound] Trivial upper bound" << endl;
		ub = new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		cout << "[UpperBound] Smart upper bound" << endl;
		ub = new ContextPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "[UpperBound] Unsupported scenario upper bound: " << name
				<< endl;
		exit(0);
	}
	return ub;
}

void ContextPomdp::Statistics(const std::vector<PomdpState *> particles) const {
	return;
	double goal_count[10][10] = { { 0 } };
	cout << "Current Belief" << endl;
	if (particles.size() == 0)
		return;

	PrintState(*particles[0]);
	PomdpState *state_0 = particles[0];
	for (int i = 0; i < particles.size(); i++) {
		PomdpState *state = particles[i];
		for (int j = 0; j < state->num; j++) {
			goal_count[j][state->agents[j].intention] += particles[i]->weight;
		}
	}

	for (int j = 0; j < state_0->num; j++) {
		cout << "Ped " << j << " Belief is ";
		for (int i = 0; i < world_model.GetNumIntentions(state_0->agents[j].id);
				i++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}
}

void ContextPomdp::PrintState(const State &s, string msg, ostream &out) const {

	if (DESPOT::Debug_mode)
		return;

	if (msg == "")
		out << "Search state:\n";
	else
		cout << msg << endl;

	PrintState(s, out);
}

void ContextPomdp::PrintState(const State &s, ostream &out) const {

	if (DESPOT::Debug_mode)
		return;

	out << "Address: " << &s << endl;

	if (static_cast<const PomdpStateWorld *>(&s) != NULL) {

		if (static_cast<const PomdpStateWorld *>(&s)->num
				> ModelParams::N_PED_IN) {
			PrintWorldState(static_cast<const PomdpStateWorld &>(s), out);
			return;
		}
	}

	const PomdpState &state = static_cast<const PomdpState &>(s);
	auto &carpos = state.car.pos;

	out << "car pos / heading / vel = " << "(" << carpos.x << ", " << carpos.y
			<< ") / " << state.car.heading_dir << " / " << state.car.vel
			<< " car dim " << ModelParams::CAR_WIDTH << " "
			<< ModelParams::CAR_FRONT * 2 << endl;
	out << state.num << " pedestrians " << endl;
	for (int i = 0; i < state.num; i++) {
		out << "agent " << i
				<< ": id / pos / speed / vel / intention / dist2car / infront =  "
				<< state.agents[i].id << " / " << "(" << state.agents[i].pos.x
				<< ", " << state.agents[i].pos.y << ") / "
				<< state.agents[i].speed << " / " << "("
				<< state.agents[i].vel.x << ", " << state.agents[i].vel.y
				<< ") / " << state.agents[i].intention << " / "
				<< COORD::EuclideanDistance(state.agents[i].pos, carpos)
				<< " / " << world_model.InFront(state.agents[i].pos, state.car)
				<< " (mode) " << state.agents[i].mode << " (type) "
				<< state.agents[i].type << " (bb) "
				<< state.agents[i].bb_extent_x << " "
				<< state.agents[i].bb_extent_y << " (cross) "
				<< state.agents[i].cross_dir << " (heading) "
				<< state.agents[i].heading_dir << endl;
	}

	double min_dist = -1;
	if (state.num > 0)
		min_dist = COORD::EuclideanDistance(carpos, state.agents[0].pos);
	out << "MinDist: " << min_dist << endl;

	ValidateState(state, __FUNCTION__);
}

void ContextPomdp::PrintStateCar(const State &s, std::string msg,
		ostream &out) const {
	const PomdpState &state = static_cast<const PomdpState &>(s);
	out << msg << " ";
	out << state.car.pos.x << " " << state.car.pos.y << " "
			<< state.car.heading_dir << endl;
}

PomdpState last_state;
void ContextPomdp::PrintStateAgents(const State &s, std::string msg,
		ostream &out) const {

	const PomdpState &state = static_cast<const PomdpState &>(s);

	out << msg << " ";
	for (int i = 0; i < state.num; i++) {
		out << state.agents[i].pos.x << " " << state.agents[i].pos.y << " "
				<< state.agents[i].heading_dir << " "
				<< state.agents[i].bb_extent_x << " "
				<< state.agents[i].bb_extent_y << " ";
	}
	out << endl;

//	out << "vel ";
//	for (int i = 0; i < state.num; i++) {
//		out << COORD::EuclideanDistance(last_state.agents[i].pos, state.agents[i].pos) * ModelParams::CONTROL_FREQ << " ";
//	}
//	out << endl;
//
//	out << "cur_vel intention ";
//	for (int i = 0; i < state.num; i++) {
//		out << world_model.IsCurVelIntention(state.agents[i].intention, state.agents[i].id) << " ";
//	}
//	out << endl;
//
//	out << "step intention ";
//	for (int i = 0; i < state.num; i++) {
//		out << world_model.IsStopIntention(state.agents[i].intention, state.agents[i].id) << " ";
//	}
//	out << endl;

	last_state = state;
}

void ContextPomdp::PrintWorldState(const PomdpStateWorld &state,
		ostream &out) const {
	out << "World state:\n";
	auto &carpos = state.car.pos;
	out << "car pos / heading / vel = " << "(" << carpos.x << ", " << carpos.y
			<< ") / " << state.car.heading_dir << " / " << state.car.vel
			<< " car dim " << ModelParams::CAR_WIDTH << " "
			<< ModelParams::CAR_FRONT * 2 << endl;
	out << state.num << " pedestrians " << endl;

	double min_dist = -1;
	int mindist_id = 0;

	for (int i = 0; i < state.num; i++) {
		if (COORD::EuclideanDistance(state.agents[i].pos, carpos) < min_dist) {
			min_dist = COORD::EuclideanDistance(state.agents[i].pos, carpos);
			mindist_id = i;
		}

		string intention_str = "";
		if (world_model.IsCurVelIntention(state.agents[i].intention, state.agents[i].id))
			intention_str = "cur_vel";
		else if (world_model.IsStopIntention(state.agents[i].intention, state.agents[i].id))
			intention_str = "stop";
		else
			intention_str = "path_" + to_string(state.agents[i].intention);

		string mode_str = "";
		if (state.agents[i].mode == AGENT_DIS)
			mode_str = "dis";
		else
			mode_str = "att";

		out << "agent " << i
				<< ": id / pos / speed / vel / intention / dist2car / infront =  "
				<< state.agents[i].id << " / " << "(" << state.agents[i].pos.x
				<< ", " << state.agents[i].pos.y << ") / "
				<< state.agents[i].speed << " / " << "("
				<< state.agents[i].vel.x << ", " << state.agents[i].vel.y
				<< ") / " << intention_str << " / "
				<< COORD::EuclideanDistance(state.agents[i].pos, carpos)
				<< " / " << world_model.InFront(state.agents[i].pos, state.car)
				<< " (mode) " << mode_str << " (type) "
				<< state.agents[i].type << " (bb) "
				<< state.agents[i].bb_extent_x << " "
				<< state.agents[i].bb_extent_y << " (cross) "
				<< state.agents[i].cross_dir << " (heading) "
				<< state.agents[i].heading_dir << endl;
	}

	if (state.num > 0)
		min_dist = COORD::EuclideanDistance(carpos,
				state.agents[mindist_id].pos);

	out << "MinDist: " << min_dist << endl;
}

void ContextPomdp::PrintObs(const State &state, uint64_t obs, ostream &out) const {
	out << obs << endl;
}

void ContextPomdp::PrintAction(int action, ostream &out) const {
	out << action << endl;
}

void ContextPomdp::PrintBelief(const Belief &belief, ostream &out) const {

}

/// output the probability of the intentions of the pedestrians
void ContextPomdp::PrintParticles(const vector<State *> particles,
		ostream &out) const {
	cout << "Particles for planning:" << endl;
	double goal_count[ModelParams::N_PED_IN][10] = { { 0 } };
	double q_goal_count[ModelParams::N_PED_IN][10] = { { 0 } }; //without weight, it is q;

	double type_count[ModelParams::N_PED_IN][AGENT_DIS + 1] = { { 0 } };

	double q_single_weight;
	q_single_weight = 1.0 / particles.size();
	cout << "Current Belief with " << particles.size() << " particles" << endl;
	if (particles.size() == 0)
		return;
	const PomdpState *pomdp_state =
			static_cast<const PomdpState *>(particles.at(0));

	if (DESPOT::Debug_mode) {
		DESPOT::Debug_mode = false;
		PrintState(*pomdp_state);
		DESPOT::Debug_mode = true;
	}

	for (int i = 0; i < particles.size(); i++) {
		const PomdpState *pomdp_state =
				static_cast<const PomdpState *>(particles.at(i));
		for (int j = 0; j < pomdp_state->num; j++) {
			goal_count[j][pomdp_state->agents[j].intention] +=
					particles[i]->weight;
			q_goal_count[j][pomdp_state->agents[j].intention] +=
					q_single_weight;
			type_count[j][pomdp_state->agents[j].mode] += particles[i]->weight;
		}
	}

	cout << "agent 0 vel: " << pomdp_state->agents[0].speed << endl;

	for (int j = 0; j < 6; j++) {
		cout << "Ped " << pomdp_state->agents[j].id << " Belief is ";
		for (int i = 0;
				i < world_model.GetNumIntentions(pomdp_state->agents[j].id);
				i++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
		for (int i = 0; i < AGENT_DIS + 1; i++) {
			cout << (type_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}

	logv << "<><><> q:" << endl;
	for (int j = 0; j < 6; j++) {
		logv << "Ped " << pomdp_state->agents[j].id << " Belief is ";
		for (int i = 0;
				i < world_model.GetNumIntentions(pomdp_state->agents[j].id);
				i++) {
			logv << (q_goal_count[j][i] + 0.0) << " ";
		}
		logv << endl;
	}
}

State *ContextPomdp::Allocate(int state_id, double weight) const {
	//num_active_particles ++;
	PomdpState *particle = memory_pool_.Allocate();
	particle->state_id = state_id;
	particle->weight = weight;
	return particle;
}

State *ContextPomdp::Copy(const State *particle) const {
	PomdpState *new_particle = memory_pool_.Allocate();
	*new_particle = *static_cast<const PomdpState *>(particle);

	new_particle->SetAllocated();
	return new_particle;
}

State *ContextPomdp::CopyForSearch(const State *particle) const {
	PomdpState *new_particle = memory_pool_.Allocate();
	const PomdpStateWorld *world_state =
			static_cast<const PomdpStateWorld *>(particle);

	new_particle->num = min(ModelParams::N_PED_IN, world_state->num);
	new_particle->car = world_state->car;
	for (int i = 0; i < new_particle->num; i++) {
		new_particle->agents[i] = world_state->agents[i];
	}
	new_particle->time_stamp = world_state->time_stamp;
	new_particle->SetAllocated();
	return new_particle;
}

void ContextPomdp::Free(State *particle) const {
	//num_active_particles --;
	memory_pool_.Free(static_cast<PomdpState *>(particle));
}

int ContextPomdp::NumActiveParticles() const {
	return memory_pool_.num_allocated();
}

int ContextPomdp::NumObservations() const {
	return std::numeric_limits<int>::max();
}
int ContextPomdp::ParallelismInStep() const {
	return ModelParams::N_PED_IN;
}
void ContextPomdp::ExportState(const State &state, std::ostream &out) const {
	PomdpState cardriveState = static_cast<const PomdpState &>(state);
	ios::fmtflags old_settings = out.flags();

	int Width = 7;
	int Prec = 3;
	out << cardriveState.scenario_id << " ";
	out << cardriveState.weight << " ";
	out << cardriveState.num << " ";
	out << cardriveState.car.heading_dir << " " << cardriveState.car.pos.x
			<< " " << cardriveState.car.pos.y << " " << cardriveState.car.vel
			<< " ";
	for (int i = 0; i < ModelParams::N_PED_IN; i++)
		out << cardriveState.agents[i].intention << " "
				<< cardriveState.agents[i].id << " "
				<< cardriveState.agents[i].pos.x << " "
				<< cardriveState.agents[i].pos.y << " "
				<< cardriveState.agents[i].speed << " ";

	out << endl;

	out.flags(old_settings);
}
State *ContextPomdp::ImportState(std::istream &in) const {
	PomdpState *cardriveState = memory_pool_.Allocate();

	if (in.good()) {
		string str;
		while (getline(in, str)) {
			if (!str.empty()) {
				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.heading_dir >> cardriveState->car.pos.x
						>> cardriveState->car.pos.y >> cardriveState->car.vel;
				for (int i = 0; i < ModelParams::N_PED_IN; i++)
					ss >> cardriveState->agents[i].intention
							>> cardriveState->agents[i].id
							>> cardriveState->agents[i].pos.x
							>> cardriveState->agents[i].pos.y
							>> cardriveState->agents[i].speed;
			}
		}
	}

	return cardriveState;
}

void ContextPomdp::ImportStateList(std::vector<State *> &particles,
		std::istream &in) const {
	if (in.good()) {
		int PID = 0;
		string str;
		getline(in, str);
		istringstream ss(str);
		int size;
		ss >> size;
		particles.resize(size);
		while (getline(in, str)) {
			if (!str.empty()) {
				if (PID >= particles.size())
					cout << "Import particles error: PID>=particles.size()!"
							<< endl;

				PomdpState *cardriveState = memory_pool_.Allocate();

				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.heading_dir >> cardriveState->car.pos.x
						>> cardriveState->car.pos.y >> cardriveState->car.vel;
				for (int i = 0; i < ModelParams::N_PED_IN; i++)
					ss >> cardriveState->agents[i].intention
							>> cardriveState->agents[i].id
							>> cardriveState->agents[i].pos.x
							>> cardriveState->agents[i].pos.y
							>> cardriveState->agents[i].speed;
				particles[PID] = cardriveState;
				PID++;

			}
		}
	}
}

bool ContextPomdp::ValidateState(const PomdpState &state, const char *msg) const {

	for (int i = 0; i < state.num; i++) {
		auto &agent = state.agents[i];

		if (agent.type >= AgentType::num_values) {
			ERR(string_sprintf("non-initialized type in state: %d", agent.type));
		}

		if (agent.speed == -1) {
			ERR("non-initialized speed in state");
		}
	}
}

OBS_TYPE ContextPomdp::StateToIndex(const State *state) const {
	std::hash<std::vector<int>> myhash;
	std::vector<int> obs_vec = ObserveVector(*state);
	OBS_TYPE obs = myhash(obs_vec);
	Obs_hash_table[obs] = obs_vec;

	if (obs <= (OBS_TYPE) 140737351976300) {
		cout << "empty obs: " << obs << endl;
		return obs;
	}

	return obs;
}

double ContextPomdp::GetAccelerationID(ACT_TYPE action, bool debug) const {
	return (action % ((int) (2 * ModelParams::NUM_ACC + 1)));
}

double ContextPomdp::GetAcceleration(ACT_TYPE action, bool debug) const {
	double acc_ID = (action % ((int) (2 * ModelParams::NUM_ACC + 1)));
	return GetAccfromAccID(acc_ID);
}

double ContextPomdp::GetAccelerationNoramlized(ACT_TYPE action, bool debug) const {
	double acc_ID = (action % ((int) (2 * ModelParams::NUM_ACC + 1)));
	return GetNormalizeAccfromAccID(acc_ID);
}

double ContextPomdp::GetSteeringID(ACT_TYPE action, bool debug) const {
	return FloorIntRobust(action / (2 * ModelParams::NUM_ACC + 1));
}

double ContextPomdp::GetSteering(ACT_TYPE action, bool debug) const {
	double steer_ID = FloorIntRobust(action / (2 * ModelParams::NUM_ACC + 1));
	double normalized_steer = steer_ID / ModelParams::NUM_STEER_ANGLE;
	double shifted_steer = normalized_steer - 1;

	if (debug)
		cout << "[GetSteering] (steer_ID, normalized_steer, shifted_steer)="
				<< "(" << steer_ID << "," << normalized_steer << ","
				<< shifted_steer << ")" << endl;
	return shifted_steer * ModelParams::MAX_STEER_ANGLE;
}

double ContextPomdp::GetSteeringNoramlized(ACT_TYPE action, bool debug) const {
	double steer_ID = FloorIntRobust(action / (2 * ModelParams::NUM_ACC + 1));
	return steer_ID / ModelParams::NUM_STEER_ANGLE - 1;
}

ACT_TYPE ContextPomdp::GetActionID(double steering, double acc, bool debug) {
	if (debug) {
		cout << "[GetActionID] steer_ID=" << GetSteerIDfromSteering(steering)
				<< endl;
		cout << "[GetActionID] acc_ID=" << GetAccIDfromAcc(acc) << endl;
	}

	return (ACT_TYPE) (GetSteerIDfromSteering(steering)
			* ClosestInt(2 * ModelParams::NUM_ACC + 1) + GetAccIDfromAcc(acc));
}

ACT_TYPE ContextPomdp::GetActionID(int steer_id, int acc_id) {
	return (ACT_TYPE) (steer_id * ClosestInt(2 * ModelParams::NUM_ACC + 1)
			+ acc_id);
}

double ContextPomdp::GetAccfromAccID(int acc) {
	switch (acc) {
	case ACT_CUR:
		return 0;
	case ACT_ACC:
		return ModelParams::ACC_SPEED;
	case ACT_DEC:
		return -ModelParams::ACC_SPEED;
	}
}

double ContextPomdp::GetNormalizeAccfromAccID(int acc) {
	switch (acc) {
	case ACT_CUR:
		return 0;
	case ACT_ACC:
		return 1.0;
	case ACT_DEC:
		return -1.0;
	}
}

int ContextPomdp::GetAccIDfromAcc(float acc) {
	if (fabs(acc - 0) < 1e-5)
		return ACT_CUR;
	else if (fabs(acc - ModelParams::ACC_SPEED) < 1e-5)
		return ACT_ACC;
	else if (fabs(acc - (-ModelParams::ACC_SPEED)) < 1e-5)
		return ACT_DEC;
}

int ContextPomdp::GetSteerIDfromSteering(float steering) {
	return ClosestInt(
			(steering / ModelParams::MAX_STEER_ANGLE + 1)
					* (ModelParams::NUM_STEER_ANGLE));
}

void ContextPomdp::PrintStateIDs(const State& s) {
	const PomdpState& curr_state = static_cast<const PomdpState&>(s);

	cout << "Sampled peds: ";
	for (int i = 0; i < curr_state.num; i++)
		cout << curr_state.agents[i].id << " ";
	cout << endl;
}

void ContextPomdp::CheckPreCollision(const State* s) {
	const PomdpState* curr_state = static_cast<const PomdpState*>(s);

	int collision_peds_id;

	if (curr_state->car.vel > 0.001
			&& world_model.InCollision(*curr_state, collision_peds_id)) {
		cout << "--------------------------- pre-collision ----------------------------"
			<< endl;
		cout << "pre-col ped: " << collision_peds_id << endl;
	}
}

