#ifndef LOWER_BOUND_POLICY_MODE
#define LOWER_BOUND_POLICY_MODE

#include "lower_bound/lower_bound_policy.h"

/* This class refines PolicyLowerBound to define the best action of a particle
 * set as the best action for the mode of the set. As the computation of the 
 * best action for the mode is problem-specific, it is further delegated to the
 * model.
 *
 * Implementation note: The problem state must define implicit conversions 
 * to int because this module uses state-indexed tables of size |S| to 
 * compute the mode in linear time. Therefore this module is unsuitable 
 * for problems with large state spaces.
 *
 * Space complexity: O(|S|)
 * Time complexity (per query): O(Number of particles * Globals::config.search_depth)
 */
template<typename T>
class ModePolicyLowerBound : public PolicyLowerBound<T> {
 public:
  ModePolicyLowerBound(const RandomStreams& streams, int num_states)
      : PolicyLowerBound<T>(streams),
        state_weight_(num_states)
  {}

  // Computes the mode and calls the model to return the best action for it.
  int Action(const vector<Particle<T>*>& particles,
             const Model<T>& model,
             const History& history) const;

  static string Name() { return "mode"; }

 private:
  mutable vector<double> state_weight_;
};

template<typename T>
int ModePolicyLowerBound<T>::Action(
    const vector<Particle<T>*>& particles,
    const Model<T>& model,
    const History& history) const {
  T mode = 0;
  for (auto p: particles) {
    state_weight_[p->state] += p->wt;
    if (state_weight_[p->state] > state_weight_[mode])
      mode = p->state;
  }
  for (auto p: particles)
    state_weight_[p->state] = 0;

  return model.LowerBoundAction(mode);
}

#endif
