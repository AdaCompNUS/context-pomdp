#ifndef WORLD_H
#define WORLD_H

#include "globals.h"
#include "model.h"

/* This class maintains the current state of the world and steps it forward
 * whenever the agent takes a real action and receives a real observation.
 */
template<typename T>
class World {
 public:
  World(unsigned seed, const Model<T>& model)
    : state_(model.GetStartState()),
      initial_state_(state_),
      seed_(seed),
      initial_seed_(seed),
      model_(model),
      total_reward_(0),
			total_undiscounted_reward_(0),
      n_steps_(0) 
  {}

  // Returns the total discounted reward
  double TotalReward();

	double TotalUndiscountedReward();

  // Advances the current state of the world.
  void Step(int action, uint64_t& obs, double& reward);
  void StepMultiple(int action, vector<uint64_t>& obss, vector<double>& rewards);

  // Resets the world to have the same initial state and seed so that
  // a sequence of updates can be reproduced exactly.
  void Reset();

	void SetStartStates(vector<T>& states) {
		states_ = states;
		cout << "Initial States:" << endl;
		for(int i=0; i<states_.size(); i++) {
			model_.PrintState(states_[i]);
		}
	}
  T GetState()
  {
  	return state_;
  }

 private:
  T state_;
	vector<T> states_;
  const T initial_state_;
  unsigned seed_;
  const unsigned initial_seed_;
  const Model<T>& model_;
  double total_reward_;
  double total_undiscounted_reward_;
  int n_steps_;
};

template<typename T>
void World<T>::Reset() {
  state_ = initial_state_;
  seed_ = initial_seed_;
  total_reward_ = 0;
  n_steps_ = 0;
}

template<typename T>
double World<T>::TotalReward() {
  return total_reward_;
}

template<typename T>
double World<T>::TotalUndiscountedReward() {
  return total_undiscounted_reward_;
}

template<typename T>
void World<T>::StepMultiple(int action, vector<uint64_t>& obss, vector<double>& rewards) {
	double random_num = (double)rand_r(&seed_) / RAND_MAX;

	cout << "Before stepping:" << endl;
	for(int i=0; i<states_.size(); i++) {
		model_.PrintState(states_[i]);
	}
	model_.StepMultiple(states_, random_num, action, rewards, obss);

	cout << "Action = " << action << endl;
	for(int i=0; i<states_.size(); i++) {
		cout << "Pedestrian " << i << endl;
		cout << "State = \n"; 
		model_.PrintState(states_[i]);
		cout << "Observation: "; 
		model_.PrintObs(obss[i]);
		cout << endl;
	}
	double reward = 0;
	for(int i=0; i<rewards.size(); i++)
		reward += rewards[i];
  cout << "Reward = " << reward << endl;
  total_reward_ += pow(Globals::config.discount, n_steps_++) * reward;
	total_undiscounted_reward_ += reward;
}

template<typename T>
void World<T>::Step(int action, uint64_t& obs, double& reward) {
  double random_num = (double)rand_r(&seed_) / RAND_MAX;
  model_.Step(state_, random_num, action, reward, obs);
  cout << "Action = " << action << endl;
  cout << "State = \n"; model_.PrintState(state_);
  cout << "Observation: "; model_.PrintObs(obs); cout << endl;
  cout << "Reward = " << reward << endl;
  total_reward_ += pow(Globals::config.discount, n_steps_++) * reward;
	total_undiscounted_reward_ += reward;
}

#endif
