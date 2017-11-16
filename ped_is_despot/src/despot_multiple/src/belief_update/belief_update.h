#ifndef BELIEF_UPDATE_H 
#define BELIEF_UPDATE_H

#include "globals.h"
#include "model.h"
#include "particle.h"

// This is the interface implemented by all belief-update strategies.
template<typename T>
class BeliefUpdate {
 public:
  BeliefUpdate(unsigned belief_update_seed, const Model<T>& model) :
    num_updates_(0),
    belief_update_seed_(belief_update_seed),
    model_(model),
    init_belief_update_seed_(belief_update_seed) {
	}

  virtual ~BeliefUpdate() {}

  // Samples N elements uniformly at random from a particle set.
  vector<Particle<T>*> Sample(const vector<Particle<T>*>& pool, int N);

  // Given a set of particles, updates their weights to sum to 1
  void Normalize(vector<Particle<T>*>& particles) const;

  // Updates a belief (@particles) given an action and observation. The
  // resulting belief will have @N particles.
  vector<Particle<T>*> Update(const vector<Particle<T>*>& particles,
                              int N, 
                              int act, 
                              T obs_state,T obs_state_old) {
    vector<Particle<T>*> ans = UpdateImpl(particles, N, act, obs_state,obs_state_old);
    num_updates_++;
    return ans;
  }

  // Resets the updater to use the same seed as the one it started with. Useful
  // when a simulation needs to be repeated exactly.
  void Reset() { 
    belief_update_seed_ = init_belief_update_seed_;
    num_updates_ = 0;
  }

 /*temply set the seed to be public,since it is needed by the particles resampler*/

 
 protected:
  // Drop particles lighter than this
  static constexpr double PARTICLE_WT_THRESHOLD = 1e-20;

  int num_updates_;

  const Model<T>& model_;
 unsigned belief_update_seed_;
 private:
  // Define this in your implementation.
  virtual vector<Particle<T>*> UpdateImpl(const vector<Particle<T>*>& particles,
                                          int N, 
                                          int act, 
                                           T obs_state,T obs_state_old) = 0;

  const unsigned init_belief_update_seed_;
};

template<typename T>
constexpr double BeliefUpdate<T>::PARTICLE_WT_THRESHOLD;

template<typename T>
vector<Particle<T>*> BeliefUpdate<T>::Sample(
    const vector<Particle<T>*>& pool, int N) {
  vector<Particle<T>*> ans;

  // Ensure particle weights sum to 1
  double sum_without_last = 0;
  auto end_particle = pool.end() - 1;
  for (auto it = pool.begin(); it != end_particle; it++)
    sum_without_last += (*it)->wt;
  double end_weight = 1 - sum_without_last;

  // Divide the cumulative frequency into N equally-spaced parts
  int num_sampled = 0;
  double r = (rand_r(&belief_update_seed_) / (double)RAND_MAX) / N;
  auto curr_particle = pool.begin();
  double cum_sum = curr_particle == end_particle 
                   ? end_weight 
                   : (*curr_particle)->wt;
  while (num_sampled < N) {
    while (cum_sum < r) {
      curr_particle++;
      cum_sum += curr_particle == end_particle 
                 ? end_weight 
                 : (*curr_particle)->wt;
    }
    Particle<T>* new_particle = model_.Copy(*curr_particle);
    new_particle->id = num_sampled++;
    new_particle->wt = 1.0 / N;
    ans.push_back(new_particle);
    r += 1.0 / N;
  }
  return ans;
}

template<typename T>
void BeliefUpdate<T>::Normalize(vector<Particle<T>*>& particles) const {
  double prob_sum = 0;
  for (auto it: particles)
    prob_sum += it->wt;
  for (auto it: particles)
    it->wt /= prob_sum;
}

#endif
