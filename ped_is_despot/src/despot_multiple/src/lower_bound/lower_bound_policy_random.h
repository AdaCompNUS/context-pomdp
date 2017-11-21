#ifndef LOWER_BOUND_POLICY_RANDOM_H
#define LOWER_BOUND_POLICY_RANDOM_H

#include "lower_bound/lower_bound_policy.h"

/* This class refines PolicyLowerBound. An action for a particle set is
 * selected in the following manner: a random state is first selected from the 
 * particle set. Then a set of actions is generated for the state, and a random
 * action is selected from the set. The set of generated actions depends on the
 * "knowledge" level that can be specified in the model parameters file
 * (default is 2). In level 1, it is the set of legal actions for the 
 * chosen state, which by default is the entire action space (can be
 * overridden). In level 2, it is a set of preferred actions, which being 
 * problem-specific, is delegated to the model.
 *
 * Space complexity: O(Number of actions generated for the state)
 * Time complexity (per query): O(Number of particles * Globals::config.search_depth)
 */
template<typename T>
class RandomPolicyLowerBound : public PolicyLowerBound<T> {
 public:
  RandomPolicyLowerBound(const RandomStreams& streams, 
                         int knowledge,
                         unsigned action_root_seed) 
      : PolicyLowerBound<T>(streams),
        knowledge_(knowledge),
        action_root_seed_(action_root_seed)
  {}

  int Action(const vector<Particle<T>*>& particles,
             const Model<T>& model,
             const History& history) const;

  static string Name() { return "random"; }

 private:
  int knowledge_;
	mutable unsigned action_root_seed_;
};

template<typename T>
int RandomPolicyLowerBound<T>::Action(
    const vector<Particle<T>*>& particles,
    const Model<T>& model,
    const History& history) const {
//	if (1 == 1) return 2;

   return model.DefaultPolicy(particles);
  const T& state = 
    particles[rand_r(&action_root_seed_) % particles.size()]->state;

	if (knowledge_ >= 2) {
		shared_ptr<vector<int>> actions = model.GeneratePreferred(state, history);
		if (!(actions->empty())) 
			return (*actions)[rand_r(&action_root_seed_) % actions->size()];
	}

	if (knowledge_ >= 1) {
		shared_ptr<vector<int>> actions = model.GenerateLegal(state, history);
		if (!(actions->empty()))
			return (*actions)[rand_r(&action_root_seed_) % actions->size()];
	}

	return rand_r(&action_root_seed_) % model.NumActions(); 
}

#endif
