#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <numeric>

#include <core/globals.h>
#include <solver/despot.h>
#include <despot/util/logging.h>

#include "config.h"
#include "coord.h"
#include "param.h"
#include "crowd_belief.h"
#include "context_pomdp.h"
#include "world_model.h"
#include "simulator_base.h"


int use_att_mode = 2;		// onlu use att mode
vector<State*> particles_;

HiddenStateBelief::HiddenStateBelief(int num_intentions, int num_modes) {
	Resize(num_intentions, num_modes);
}

void HiddenStateBelief::Reset() {
	for (std::vector<double>& intention_probs: probs_) {
		std::fill(intention_probs.begin(), intention_probs.end(),
				1.0/intention_probs.size()/probs_.size());
	}
}

void HiddenStateBelief::Resize(int new_intentions) {
	int mode_size = probs_.size();
	Resize(new_intentions, mode_size);
}

void HiddenStateBelief::Resize(int new_intentions, int new_modes) {
	probs_.resize(new_modes);
	for (std::vector<double>& intention_probs: probs_) {
		intention_probs.resize(new_intentions);
		std::fill(intention_probs.begin(), intention_probs.end(), 1.0/new_intentions/new_modes);
	}
}

double gaussian_prob(double x, double stddev) {
	double a = 1.0 / stddev / sqrt(2 * M_PI);
	double b = -x * x / 2.0 / (stddev * stddev);
	return a * exp(b);
}

double TransitionLikelihood(const COORD& past_pos, const COORD& cur_pos, const COORD& pred_pos) {
	const double K = 0.001;

	double goal_dist = (pred_pos - past_pos).Length();
	double move_dist = (cur_pos - past_pos).Length();

	double angle_error = COORD::Angle(past_pos, cur_pos, pred_pos, 0.0);
	double angle_prob = gaussian_prob(angle_error,
			ModelParams::NOISE_GOAL_ANGLE) + K;

	double dist_error = move_dist - goal_dist;
	double dist_prob = gaussian_prob(dist_error,
			ModelParams::NOISE_PED_VEL / ModelParams::CONTROL_FREQ) + K;

	if (isnan(angle_prob) || isnan(dist_prob))
		ERR("Get transition likelihood as NAN");

	return angle_prob * dist_prob;
}

void HiddenStateBelief::Update(WorldModel& model, AgentStruct& past_agent, const AgentStruct& cur_agent,
		int intention_id, int mode_id) {

	AgentStruct predicted_agent = past_agent;
	if (past_agent.type == AGENT_ATT)
		model.GammaAgentStep(predicted_agent, intention_id);
	else
		model.AgentStepPath(predicted_agent);

	double likelihood = TransitionLikelihood(past_agent.pos, cur_agent.pos, predicted_agent.pos);

	probs_[mode_id][intention_id] = likelihood * probs_[mode_id][intention_id];
}

void HiddenStateBelief::Normalize() {
	double total_prob = 0;
	for (auto& intention_probs: probs_) {
		total_prob += std::accumulate(intention_probs.begin(), intention_probs.end(), 0.0);
	}

	if(total_prob == 0)
		ERR("total_prob == 0");

	for (auto& intention_probs: probs_) {
		std::transform(intention_probs.begin(), intention_probs.end(), intention_probs.begin(),
		               std::bind(std::multiplies<double>(), std::placeholders::_1, 1.0/total_prob));
	}
}

void HiddenStateBelief::Sample(int& intention_id, int& mode_id) {
	this->Normalize();

	double r = Random::RANDOM.NextDouble();

	if (use_att_mode == 0) {
		for (int mode = 0; mode < probs_.size(); mode++) {
			auto& goal_probs = probs_[mode];
			for (int intention = 0; intention < goal_probs.size(); intention++) {
				r -= probs_[mode][intention];
				if (r <= 0.001) {
					intention_id = intention;
					mode_id = mode;
					break;
				}
			}
			if (r <= 0.001)
				break;
		}
	} else {
		int mode = (use_att_mode <= 1) ? AGENT_DIS : AGENT_ATT;
		auto& goal_probs = probs_[mode];
		double total_prob = std::accumulate(goal_probs.begin(), goal_probs.end(), 0.0);
		for (int intention = 0; intention < goal_probs.size(); intention++) {
			r -= probs_[mode][intention] / total_prob;
			if (r <= 0.001) {
				intention_id = intention;
				mode_id = mode;
				break;
			}
		}
	}

	if (r > 0.001)
		ERR("Sampling belief failed");
}

void AgentBelief::Reset(int new_intentions) {
	if (new_intentions != belief_->size(1))
		belief_->Resize(new_intentions);
	else
		belief_->Reset();
}

void AgentBelief::Sample(int& intention, int& mode) const {
	belief_->Sample(intention, mode);
}

void AgentBelief::Update(WorldModel& model, const AgentStruct& cur_agent, int num_intentions) {
	if (num_intentions != belief_->size(1))
		belief_->Resize(num_intentions);

	AgentStruct past_agent = observable_;
	for (int mode = 0; mode < belief_->size(0); mode++)
		for (int intention = 0; intention < belief_->size(1); intention++) {
			belief_->Update(model, past_agent, cur_agent, intention, mode);
		}
	observable_ = cur_agent;
	time_stamp = Globals::ElapsedTime();
}

CrowdBelief::CrowdBelief(const DSPOMDP* model): Belief(model),
		world_model_(SimulatorBase::world_model){
}

CrowdBelief::CrowdBelief(const DSPOMDP* model, History history,
		std::map<double, AgentBelief*> sorted_belief_) : Belief(model),
	sorted_belief_(sorted_belief_),	world_model_(SimulatorBase::world_model) {
}

CrowdBelief::~CrowdBelief() {
	for (std::map<double,AgentBelief*>::iterator it=sorted_belief_.begin();
			it!=sorted_belief_.end(); ++it) {
		AgentBelief* b = it->second;
		delete b;
	}
}

Belief* CrowdBelief::MakeCopy() const {
	return new CrowdBelief(model_, history_, sorted_belief_);
}


std::vector<State*> CrowdBelief::Sample(int num) const {
	logi << "Sample particles from belief" << endl;
	const ContextPomdp* context_pomdp = static_cast<const ContextPomdp*>(model_);

	if (DESPOT::Debug_mode)
		std::srand(0);

	for (auto* particle: particles_) {
		if (particle->IsAllocated())
			model_->Free(particle);
	}
	particles_.resize(num);

	for (int i = 0; i < num; i++) {
		particles_[i] = model_->Allocate(i, 1.0/num);
	}

	for(auto* particle: particles_) {
		PomdpState* state = static_cast<PomdpState*>(particle);
		state->car = car_;
		state->num = min<int>(ModelParams::N_PED_IN, sorted_belief_.size());

		cout << "state->num=" << state->num <<
				", sorted_belief_.size()=" << sorted_belief_.size() << endl;

		state->time_stamp = Globals::ElapsedTime();

		int agent_id = 0;
		for (std::map<double,AgentBelief*>::const_iterator it=sorted_belief_.begin(); it!=sorted_belief_.end(); ++it) {
			if (agent_id < state->num) {
				AgentBelief* b = it->second;
				state->agents[agent_id] = b->observable_;
				b->Sample(state->agents[agent_id].intention, state->agents[agent_id].mode);

				logd << "[Sample] agent " << state->agents[agent_id].id <<
						" num_intentions=" <<
						world_model_.GetNumIntentions(state->agents[agent_id].id) << endl;

				world_model_.ValidateIntention(state->agents[agent_id].id, state->agents[agent_id].intention,
						__FUNCTION__, __LINE__);

				agent_id++;
			} else
				break;
		}
	}

	if (abs(State::Weight(particles_) - 1.0) > 1e-6) {
		cerr << "[CrowdBelief::CrowdBelief] Particle weights sum to "
				<< State::Weight(particles_) << " instead of 1" << endl;
		ERR("particle sampling error");
	}

	random_shuffle(particles_.begin(), particles_.end());

	return particles_;
}

void CrowdBelief::Update(ACT_TYPE action, OBS_TYPE obs) {
	ERR("this update function is deprecated");
}

void CrowdBelief::Update(ACT_TYPE action, const State* state) {
	const PomdpStateWorld* observed = static_cast<const PomdpStateWorld*>(state);

	logd << "[CrowdBelief::Update] " << "observed->num=" << observed->num << endl;
	std::map<int, const AgentStruct*> src_agent_map;
	for (int i=0;i < observed->num; i++) {
		src_agent_map[observed->agents[i].id] = &(observed->agents[i]);
	}

	// new container for belief update
	std::map<int,AgentBelief*> indexed_belief;
	for (std::map<double,AgentBelief*>::iterator it=sorted_belief_.begin(); it!=sorted_belief_.end(); ++it) {
		AgentBelief* agent_belief = it->second;
		indexed_belief[agent_belief->observable_.id] = agent_belief;
	}
	logd << "[CrowdBelief::Update] " << "indexed_belief.size()=" << indexed_belief.size() << endl;

	// update belief
	int n = 0;
	car_ = observed->car;
	world_model_.PrepareAttentiveAgentMeanDirs(state);
	for (auto it=src_agent_map.begin(); it!=src_agent_map.end(); ++it) {
		int id = it->first;
		const AgentStruct& agent = *(it->second);
		auto it1 = indexed_belief.find(id);
		int num_intentions = world_model_.GetNumIntentions(agent.id);
		logd << "[Update] agent " << agent.id << " num_intentions=" << num_intentions << endl;
		if (it1 != indexed_belief.end()) { // existing agents
			AgentBelief* agent_belief = it1->second;
			if (world_model_.NeedBeliefReset(id))
				agent_belief->Reset(num_intentions);
			agent_belief->Update(world_model_, agent, num_intentions);
		} else { // new agent
			indexed_belief[id] = new AgentBelief(num_intentions, PED_MODES::NUM_AGENT_TYPES);
			indexed_belief[id]->observable_ = agent;
		}
	}
	logd << "[CrowdBelief::Update] " << "indexed_belief.size()=" << indexed_belief.size() << endl;

	// remove out-dated agents
	double time_stamp = Globals::ElapsedTime();
    std::map<int,AgentBelief*>::iterator it = indexed_belief.begin();
	while (it != indexed_belief.end()){
		AgentBelief* agent_belief = it->second;
		if (agent_belief->OutDated(time_stamp)) {
			logd << "[CrowdBelief::Update] " << "agent disappear" << endl; 
			delete agent_belief;
			it = indexed_belief.erase(it);
		} else 
			++it;
	}

	// agents disappeared less than 2 seconds
	for (auto it=indexed_belief.begin(); it!=indexed_belief.end(); ++it) {
		AgentBelief* agent_belief = it->second;
		if (world_model_.NumPaths(agent_belief->observable_.id) == 0) {
			logd << "[CrowdBelief::Update] " << "cur time_stamp = " << time_stamp
					<< ", belief time_stamp="<< agent_belief->time_stamp << endl;
			agent_belief->Reset(world_model_.GetNumIntentions(agent_belief->observable_.id));
		}
	}

	logd << "[CrowdBelief::Update] " << "indexed_belief.size()=" << indexed_belief.size() << endl;

	// regenerate ordered belief
	sorted_belief_.clear();
	for (std::map<int,AgentBelief*>::iterator it=indexed_belief.begin(); it!=indexed_belief.end(); ++it) {
		int id = it->first;
		AgentBelief* agent_belief = it->second;
		double dist_to_car = COORD::EuclideanDistance(agent_belief->observable_.pos, car_.pos);
		sorted_belief_[dist_to_car] = agent_belief;
	}

	logd << "[CrowdBelief::Update] " << "sorted_belief_.size()=" << sorted_belief_.size() << endl;
}
