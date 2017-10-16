#ifndef UPPER_BOUND_STOCHASTIC_H
#define UPPER_BOUND_STOCHASTIC_H

#include "upper_bound/upper_bound.h"

/* This class computes a generic upper bound for stochastic models. The space
 * complexity is O(#particles * search depth * |S|), time complexity for
 * pre-processing is O(#particles * search depth * |S| * |A|) and time
 * complexity for queries is O(#particles).
 *
 * The upper bound for a set of particles is the average of the upper bounds
 * of the particles. The upper bound for a particle is computed as the best
 * possible sequence of actions that can lead to a terminal state using the
 * random number sequence for that particle.
 */
template<typename T>
class UpperBoundStochastic : public IUpperBound<T> {
 public:
  UpperBoundStochastic(const RandomStreams& streams, const Model<T>& model);

  double UpperBound(const vector<Particle<T>*>& particles,
                    const Model<T>& model,
                    int stream_position,
                    History& history) const;

 private:
  // Helper to compute the upper bound for a single particle at a given 
  // position in the stream.
  double UpperBound(const Particle<T>& p, int stream_position,
                    const Model<T>& model);

  // Index: [particle id][stream position][state]
  // The last index, state, is required because a given particle at a given 
  // depth may end up in one of several states depending on the sequence of 
  // actions that led it there.
  vector<vector<vector<double>>> upper_bound_memo_;
};

template<typename T>
UpperBoundStochastic<T>::UpperBoundStochastic(
    const RandomStreams& streams, const Model<T>& model)
    : IUpperBound<T>(streams) {
  upper_bound_memo_.resize(Globals::config.n_particles);
  for (auto& it : upper_bound_memo_) {
    it.resize(Globals::config.search_depth + 1); // +1: we store the edge case as well
    for (auto& it2 : it) {
      it2.resize(model.NumStates());
      fill(it2.begin(), it2.end(), -Globals::INF);
    }
  }
  cerr << "Computing upper bound\n";
  for (int i = 0; i < Globals::config.n_particles; i++) {
    if (i % 10 == 0)
      cerr << i << " streams done..\n";
    for (int j = 0; j <= Globals::config.search_depth; j++)
      for (int k = 0; k < model.NumStates(); k++)
        if (upper_bound_memo_[i][j][k] == -Globals::INF)
          UpperBound(Particle<T>(k, i, 0/*dummy*/), j, model);
  }
}

template<typename T>
double UpperBoundStochastic<T>::UpperBound(
    const vector<Particle<T>*>& particles,
    const Model<T>& model,
    int stream_position,
    History& history) const {
  double ws = 0, total_cost = 0;
  for (auto p: particles) {
    ws += p->wt;
    total_cost += p->wt * upper_bound_memo_[p->id][stream_position][p->state];
  }
  return total_cost / ws;
}

template<typename T>
double UpperBoundStochastic<T>::UpperBound(const Particle<T>& p, 
                                           int stream_position,
                                           const Model<T>& model) {
  auto& ref = upper_bound_memo_[p.id][stream_position][p.state];
  if (ref != -Globals::INF)
    return ref;
  if (stream_position >= Globals::config.search_depth) {
    ref = model.FringeUpperBound(p.state);
    return ref;
  }
  if (model.IsTerminal(p.state)) {
    ref = 0;
    return 0;
  }
  double r;
  auto p_copy = model.Copy(&p);
  for (int a = 0; a < model.NumActions(); a++) {
    model.Step(p_copy->state, this->streams_.Entry(p.id, stream_position), a, r);
    double maybe_better = r + Globals::config.discount *
                          UpperBound(*p_copy, stream_position + 1, model);
    if (maybe_better > ref)
      ref = maybe_better;
    *p_copy = p;
  }

  model.Free(p_copy);

  return ref;
}

#endif
